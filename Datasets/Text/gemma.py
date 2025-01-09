import pandas as pd
import re
import signal
import time  
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from tqdm import tqdm  

tokenizer = AutoTokenizer.from_pretrained("/model-weights/gemma-2-9b-it/")
model = AutoModelForCausalLM.from_pretrained("/model-weights/gemma-2-9b-it/", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda")
model_name = 'gemma'


def label_and_reason(content):
    full_prompt = f"""Assess the text below for potential disinformation (try finding Deliberately misleading or biased information) by identifying the presence of rhetorical techniques listed.
                If you find any of the listed rhetorical techniques, then the article is likely disinformation; if not, it is likely not disinformation.
                Provide three separate assessments with 'Likely' or 'Unlikely' followed by one-line long concise reasoning on why you chose 'Likely' or 'Unlikely' for each without any further explanation.
                
                Rhetorical Techniques Checklist:
                - Emotional Appeal: Uses language that intentionally invokes extreme emotions like fear or anger, aiming to distract from lack of factual backing.
                - Exaggeration and Hyperbole: Makes claims that are unsupported by evidence, or presents normal situations as extraordinary to manipulate perceptions.
                - Bias and Subjectivity: Presents information in a way that unreasonably favors one perspective, omitting key facts that might provide balance.
                - Repetition: Uses repeated messaging of specific points or misleading statements to embed a biased viewpoint in the reader's mind.
                - Specific Word Choices: Employs emotionally charged or misleading terms to sway opinions subtly, often in a manipulative manner.
                - Appeals to Authority: References authorities who lack relevant expertise or cites sources that do not have the credentials to be considered authoritative in the context.
                - Lack of Verifiable Sources: Relies on sources that either cannot be verified or do not exist, suggesting a fabrication of information.
                - Logical Fallacies: Engages in flawed reasoning such as circular reasoning, strawman arguments, or ad hominem attacks that undermine logical debate.
                - Conspiracy Theories: Propagates theories that lack proof and often contain elements of paranoia or implausible scenarios as facts.
                - Inconsistencies and Factual Errors: Contains multiple contradictions or factual inaccuracies that are easily disprovable, indicating a lack of concern for truth.
                - Selective Omission: Deliberately leaves out crucial information that is essential for a fair understanding of the topic, skewing perception.
                - Manipulative Framing: Frames issues in a way that leaves out alternative perspectives or possible explanations, focusing only on aspects that support a biased narrative.
                
                Response format required: 
                1. [Likely/Unlikely] [Reasoning], 2. [Likely/Unlikely] [Reasoning], 3. [Likely/Unlikely] [Reasoning]

                Text: {content}"""
    try:
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(**inputs, temperature=0.2, max_new_tokens=500)  
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def convert_analysis_to_dict(row, analysis_str):
    results = row.to_dict()
    results[f'{model_name}'] = ''

    analysis_str = analysis_str.replace('\n', ' ').replace('\r', ' ').strip()

    normalized_str = analysis_str.lower()

    pattern = r'[\[\(\{\<]\s*\b(likely|unlikely)\b\s*[\]\)\}\>]|\b(likely|unlikely)\b'
    matches = re.findall(pattern, normalized_str)

    label_count = {'Likely': 0, 'Unlikely': 0}
    
    for match in matches:
        label = match[0].strip().capitalize() if match[0] else match[1].strip().capitalize()
        if label in label_count:
            label_count[label] += 1

    if label_count['Likely'] > label_count['Unlikely']:
        results[f'{model_name}'] = 'Likely'
    elif label_count['Unlikely'] > label_count['Likely']:
        results[f'{model_name}'] = 'Unlikely'
    else:
        print(f"No label Assigned for {results['unique_id']}")
        return None

    return results


def handle_signal(signal_num, frame):
    print(f"\nSignal {signal_num} received. Saving progress before exiting.")
    raise KeyboardInterrupt if signal_num in {signal.SIGINT, signal.SIGTERM, signal.SIGUSR1} else SystemExit


for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGTSTP, signal.SIGUSR1):
    signal.signal(sig, handle_signal)


def process_csv(batch_size=100):

    data = pd.read_csv("/fs01/projects/NMB-Plus/Caesar/Datasets/text_merged.csv")

    processed_file = f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"
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
            unique_id = row.get('unique_id', None)
            
            content = row.get('content', '').replace('\n', ' ').replace('\r', ' ').strip()

            if content:
                start_time = time.time()  
                analysis_result = label_and_reason(content)
                
                if analysis_result:
                    result = convert_analysis_to_dict(row, analysis_result)
                    if result:
                        results_list.append(result)

                elapsed_time = time.time() - start_time
                print(f"Processed unique_id: {unique_id}, Time taken: {elapsed_time:.2f} seconds.")


            if len(results_list) >= batch_size:
                pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"), index=False)
                results_list.clear()

    except (KeyboardInterrupt, SystemExit):  
        print("\nProcess interrupted or suspended! Saving current progress...")
        if results_list:
            pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"), index=False)
            results_list.clear()

    
    finally:
        if results_list:
            pd.DataFrame(results_list).to_csv(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv", mode='a', header=not os.path.exists(f"/fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv"), index=False)
            print(f"Final results saved to /fs01/projects/NMB-Plus/Caesar/Datasets/Text/CSVs/{model_name}.csv.")


def test_single(content):
    analysis_result = label_and_reason(content)
    if analysis_result:
        dummy_row = pd.Series({'unique_id': 'test_id', 'content': content})
        result = convert_analysis_to_dict(dummy_row, analysis_result)
        print("Single analysis result:", result)
    else:
        print("Analysis failed for the provided content.")


if __name__ == "__main__":
    # test_content = "Scientists agree that climate change is a hoax created by greedy corporations looking to profit off of government handouts. The media loves to sensationalize this non-issue to distract us from real problems like crime and unemployment. Don't let them fool you—our planet has been perfectly fine for centuries, and it's just a ploy to control our lives!"
    # test_single(test_content)
    process_csv()