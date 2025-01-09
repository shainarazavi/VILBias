import pandas as pd
import signal
import time  
from PIL import Image
import io
from transformers import AutoModel, AutoTokenizer
import torch
import os
from tqdm import tqdm 
import re


tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-2_6", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "openbmb/MiniCPM-V-2_6",
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
)
model_name = 'cpm'


def truncate_content(content, max_words=125):
    words = content.split()
    if len(words) > max_words:
        truncated_content = ' '.join(words[:max_words])
        return truncated_content
    return content


def parse_to_dict(input_data):
    input_data = re.sub(r'```\w*\s*', '', input_data)
    input_data = input_data.strip()
    input_data = input_data.strip('{}')
    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
    matches = re.findall(pattern, input_data)
    result = {key.strip(): value.strip() for key, value in matches}
    return result


def log_missing_unique_id(unique_id):
    csv_filename = '/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/missing_unique_ids.csv'
    
    if os.path.exists(csv_filename):
        existing_ids = pd.read_csv(csv_filename)['missing_unique_id'].tolist()
    else:
        existing_ids = []

    if unique_id not in existing_ids:
        new_entry = pd.DataFrame({'missing_unique_id': [unique_id]})
        new_entry.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)


def label_and_reason(unique_id, content, date_published):
    print("processing: ", unique_id)
    truncated_content = truncate_content(content)
    context = f"""
        Headline: {truncated_content}
        Date: {date_published}
    """

    full_prompt = f"""
        <|user|>\n<|image_1|>
        Analyze the given image and the following context, if present, from a news article and perform the following task. 
        The response you give should be a python dictionary where the keys are "{model_name}_description", "{model_name}_bias_analysis", "{model_name}_label". 

        In the value for "{model_name}_description": Describe the image in detail including its content, such as key subjects (people, objects, setting, actions, or events). Mention notable features such as color schemes, facial expressions, or background elements if relevant in just 1 line.\n

        In the value for "{model_name}_bias_analysis": Analyze the image and the provided context for potential biases and summarize your analysis in 1 line, referencing specific elements from the image and/or context.
        Consider the following factors, and any other relevant points, in your analysis:
        - Why might this particular image have been chosen? Would using another image significantly change the message/tone?
        - How are people, places, or things depicted? Are they shown in a positive, negative, or neutral light?
        - Could the image evoke strong emotions in viewers that may sway opinions? What type of emotions?
        - Does the image reinforce stereotypes or simplify a complex issue?
        - Look at how the image is cropped or framed. What’s included versus excluded? Note the angle, perspective, and composition.
        - Examine whether the image seems digitally altered or staged versus candid.
        - Consider how the image influences readers' interpretation of the headline.

        In the value for "{model_name}_label", answer: Is the image "Likely" or "Unlikely" to be biased? the answer should be one of those keywords only. 

        The response format should be the following: {{"{model_name}_description": [INSERT DESCRIPTION HERE], "{model_name}_bias_analysis": [INSERT BIAS ANALYSIS HERE], "{model_name}_label": [INSERT LABEL HERE]}}
        
        Context:
        {context}
        <|end|>\n<|assistant|>\n
    """
    try: 
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_path = None
        for ext in image_extensions:
            img_file = os.path.join('/fs01/projects/NMB-Plus/Caesar/Datasets/images_merged', f"{unique_id}{ext}")
            if os.path.exists(img_file):
                image_path = img_file
                break
        
        compressed_image = None
        if image_path is None:
            print(f"No image found for unique_id: {unique_id}")
            log_missing_unique_id(unique_id)
            return
        else:
            print("image found")
            compressed_image_data = compress_image(image_path)
            compressed_image = Image.open(io.BytesIO(compressed_image_data))
    except Exception as e:
        print("Error opening and compressing image: ", unique_id)
    try:
        
        with torch.no_grad():
            msgs = [{'role': 'user', 'content': [compressed_image, full_prompt]}]
            response = model.chat(
                    image=None,
                    msgs=msgs,
                    tokenizer=tokenizer
                )

            print(response)

            result_row = {
                f'{model_name}_description': '', 
                f'{model_name}_bias_analysis': '',
                f'{model_name}_label': ''
            }

            result = parse_to_dict(response)

            if result and result[f'{model_name}_description'] and result[f'{model_name}_bias_analysis'] and result[f'{model_name}_label'] and (result[f'{model_name}_label'].lower() == "likely" or result[f'{model_name}_label'].lower() == "unlikely"):
                result[f'{model_name}_label'] = result[f'{model_name}_label'].lower().capitalize()
                result_row = result

        print("Single analysis result:", result_row)
        return result_row
    except Exception as e:
        print(f"An error occurred while processing {unique_id}: {str(e)}")


def compress_image(img_path):
    """Compresses image at img_path and returns compressed binary data."""
    with Image.open(img_path) as image:
        with io.BytesIO() as image_bytes:
            if image.format == 'JPEG':
                image.save(image_bytes, format='JPEG', quality=75)
            elif image.format == 'PNG':
                image.save(image_bytes, format='PNG', optimize=True)
            else:
                raise ValueError("Unsupported image format. Please provide a JPG or PNG image.")
            
            compressed_image_data = image_bytes.getvalue()

    return compressed_image_data  


def handle_signal(signal_num, frame):
    print(f"\nSignal {signal_num} received. Saving progress before exiting.")
    raise KeyboardInterrupt if signal_num in {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1} else SystemExit


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP, signal.SIGUSR1):
    signal.signal(sig, handle_signal)


def process_csv(batch_size=100):
    """Processes each row in the CSV row by row and passes each row to label_and_reason for inference."""

    data = pd.read_csv("/fs01/projects/NMB-Plus/Caesar/Datasets/text_merged.csv")

    processed_file = f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv"

    if os.path.exists(processed_file):
        processed_data = pd.read_csv(processed_file, usecols=['unique_id'])  
        processed_indices = set(processed_data['unique_id'].values)  
    else:
        processed_indices = set()

    print("Processed Unique ID Numbers: ", len(processed_indices))

    data = data[~data['unique_id'].isin(processed_indices)]

    results_list = []

    try:
        for index, row in tqdm(data.iterrows(), total=len(data), desc='Processing'):
            unique_id = row.get('unique_id')
            title = row.get('title', '').replace('\n', ' ').replace('\r', ' ').strip() if row.get('title') else ''

            date_published = row.get('date_published', '')
            
            start_time = time.time()
            analysis_result = label_and_reason(unique_id, title, date_published)
            
            if analysis_result:
                if analysis_result[f'{model_name}_description'] and analysis_result[f'{model_name}_bias_analysis'] and analysis_result[f'{model_name}_label']:
                    result = {**row.to_dict(), **analysis_result}
                    results_list.append(result)
            else:
                continue

            elapsed_time = time.time() - start_time
            print(f"Processed unique_id: {unique_id}, Time taken: {elapsed_time:.2f} seconds.")

            print("length of results_list: ", len(results_list))
            if len(results_list) >= batch_size:
                print("saving to csv the results_list that is greater than batch_size!")
                pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv"), index=False)
                results_list.clear()

    except (KeyboardInterrupt, SystemExit):  
        print("\nProcess interrupted or suspended! Saving current progress...")
        if results_list:
            pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv"), index=False)
            results_list.clear()
    finally:
        if results_list:
            pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv"), index=False)
            print(f"Final results saved to /fs01/projects/NMB-Plus/Caesar/Datasets/Images/CSVs/{model_name}.csv.")


if __name__ == "__main__":
    # label_and_reason("63a7cff0fe", "We need our farmland,' regional council told amid protest over Wilmot land deal - CBC.ca", "2024-06-20 16:13:00+00:00")
    process_csv()