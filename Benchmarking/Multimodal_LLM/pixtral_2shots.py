import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
import pandas as pd
import numpy as np
import os
import time
from codecarbon import EmissionsTracker
import logging
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import seaborn as sns
import random
import wandb
import re
from PIL import Image
import io

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

model_name = 'pixtral_2shots'

tracker = EmissionsTracker(project_name=f"{model_name}", output_dir=f"./{model_name}", output_file=f"{model_name}_emissions.csv")

class PixtralClassifier:
    def __init__(self, model_name="/model-weights/pixtral-12b/", max_length=500):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = LlavaForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)

        self.model.to(self.device)
        self.max_length = max_length
        end_time = time.time() - start_time
        print("Time Taken for Initializing Model: ", end_time)

    def truncate_content(self, content, max_words=125):
        words = content.split()
        if len(words) > max_words:
            truncated_content = ' '.join(words[:max_words])
            return truncated_content
        return content
    
    def compress_image(self, img_path):
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

    def label_and_reason(self, image_path, content):
        full_prompt = f"""
        <s>[INST][IMG]
        Example 1:
        Image: [IMG]
        Text: "This is a published version of the Forbes Daily newsletter, you can sign-up to get Forbes Daily in your inbox here.  Good morning,  As companies race to capture the weight loss drug market Goldman Sachs analysts predict will soon be worth $100 billion, one drug has a new advantage when it comes to shedding pounds.  New research shows that tirzepatide, the generic name for Zepbound and Mounjaro, was “associated with significantly greater weight loss” than semaglutide, the generic name for Wegovy and Ozempic, in overweight or obese adults.  The research is a boon for Zepbound, manufactured by pharma giant Eli Lilly, but that doesn’t mean it’s going to replace dominant player Novo Nordisk’s Wegovy anytime soon. Both drugs have been hailed as game changers for treating obesity—and both are in short supply.  FIRST UP  No one has spent more trying to reinstall Donald Trump in the White House than Timothy Mellon: The grandson of banking titan Andrew Mellon has given $76.5 million to Trump groups, Forbes recently reported. But Mellon doesn’t have as vast of a fortune as billionaires like Michael Bloomberg, worth $106.2 billion, or Miriam Adelson, who along with her family is worth $28.5 billion, according to Forbes’ real-time billionaires tracker. He only appears to be worth about $1 billion, based on his estimated inheritance and proceeds from the sale of his railroad company.  BUSINESS + FINANCE  Bitcoin prices briefly fell to a four-month low on Monday, dropping over 6% before recovering to about $57,400 as of Tuesday morning. Bitcoin’s weeks-long downturn has been driven by diminishing interest in cryptocurrency ETFs, uncertainty over monetary policy, and fears of a massive bitcoin selloff from the bankrupt Mt. Gox exchange, which could spark further declines."
        Response: Likely

        Example 2:
        Image: [IMG]
        Text: "Jared Schoenfeld spent over 15 successful years in the sports entertainment industry as an executive. I sat down with him to discuss how he pivoted from a career in sports entertainment to one focusing on mental health and well-being. In 2011, he received global recognition by SportsPro Media as one of the ten young leaders in the sports business, and in 2015, he was nominated for Forbes’ 30 under 30 list. However, amidst his career achievements, later on in his career, Schoenfeld began experiencing symptoms of burnout and anxiety, prompting him to take some time off for self-reflection and ultimately find a new path where he could make a meaningful impact on the lives of others. This led him to founding 4B Advisory in May 2023 after advising Chopra Global and Dr. Deepak Chopra for a few years, for which he is grateful.  His personal experiences highlight the importance of self-care and well-being for Schoenfeld. In 2018, he experienced his first and only panic attack, which occurred during a meeting. Overwhelmed by shortness of breath and palpitations, he went to his best friend and lead advisor’s apartment, David Oestreicher, who has also played a critical role in launching 4B.  “I was overly tired and would be nervous in settings that I was never nervous in before – I was overthinking everything,” says Schoenfeld. Towards the latter part of his career in sports entertainment, he developed anxiety symptoms and burnout, which he attributes to long work hours, dealing with some personal matters, all while neglecting his health. “It was a blend of working seven days a week not taking care of my physical and mental health, not eating well, working around the clock, and dealing with personal and family challenges all at once” notes Schoenfeld."
        Response: Unlikely

        Analyze the given image and the text below for potential disinformation.
        If you find any of the bias or disinformation in either the image or the text, then the article is likely to be bias or disinformation; if the image or the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
        If either the image or the text contain any bias or disinformation, please respond with 'Likely'. Otherwise, if both the image and the text do not contain any bias or disinformation, please respond with 'Unlikely'.

        Response format required: 
        [Likely/Unlikely]

        Text: {self.truncate_content(content)}
        [/INST]
        """
        print("image path is.... ", image_path)
        image_path = "/fs01/projects/NMB-Plus/Caesar/Datasets/" + image_path
        try:
            compressed_image = None
            image1 = None
            image2 = None
            if image_path is None:
                return
            else:
                print("image found")
                compressed_image_data = self.compress_image(image_path)
                compressed_image = Image.open(io.BytesIO(compressed_image_data)) 
                image1_data = self.compress_image("/fs01/projects/NMB-Plus/Caesar/Datasets/images/9c4593df6d.jpg") 
                image1 = Image.open(io.BytesIO(image1_data))            
                image2_data = self.compress_image("/fs01/projects/NMB-Plus/Caesar/Datasets/images/51b79a40cc.jpg")
                image2 = Image.open(io.BytesIO(image2_data))
        except Exception as e:
            print(f"An error while compressing the image: {str(e)}")
            return None
        
        try:
            inputs = self.processor(text=full_prompt, images=[compressed_image, image1, image2], return_tensors="pt").to("cuda:0")
            
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, max_new_tokens=50, temperature=0.2, eos_token_id=self.processor.tokenizer.eos_token_id)
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
                response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print(response)
                return response
        except Exception as e:
            print(f"An error occurred while processing : {str(e)}")

            
    def eval(self, image_paths, text, true_vals):
        """Test the trained model"""

        self.model.eval()
        preds, targets = [], []
        tracker.start()
        for i, content in enumerate(text):
            response = self.label_and_reason(image_paths[i], content)
            if response:
                analysis_str = response.replace('\n', ' ').replace('\r', ' ').strip()
                normalized_str = analysis_str.lower()
                pattern = r'[\[\(\{\<]\s*\b(unlikely|likely)\b\s*[\]\)\}\>]|\b(unlikely|likely)\b'
                label = re.search(pattern, normalized_str)
                if label:
                    label = label.group().title()
                print(i, label)
                if label == 'Likely' or label == 'Unlikely':
                    preds.append(label)
                    targets.append(true_vals[i])
                    print(len(preds))

        tracker.stop()
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, support = precision_recall_fscore_support(targets, preds, average='weighted')
        cm = confusion_matrix(targets, preds)
        report = classification_report(targets, preds, target_names=['Likely', 'Unlikely'], output_dict=True)

        print("\nClassification Results:")
        print("-----------------------")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall F1: {f1:.4f}")
        tracker.stop()
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, support = precision_recall_fscore_support(targets, preds, average='weighted')
        cm = confusion_matrix(targets, preds)
        report = classification_report(targets, preds, target_names=['Likely', 'Unlikely'], output_dict=True)

        likely_precision = report['Likely']['precision']
        likely_recall = report['Likely']['recall']
        likely_f1 = report['Likely']['f1-score']
        
        unlikely_precision = report['Unlikely']['precision']
        unlikely_recall = report['Unlikely']['recall']
        unlikely_f1 = report['Unlikely']['f1-score']

        print("\nPer-Class Metrics:")
        print(f"Likely Class - Precision: {likely_precision:.4f}, Recall: {likely_recall:.4f}, F1: {likely_f1:.4f}")
        print(f"Unlikely Class - Precision: {unlikely_precision:.4f}, Recall: {unlikely_recall:.4f}, F1: {unlikely_f1:.4f}")

        print("\nClassification Results:")
        print("-----------------------")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Overall Precision: {precision:.4f}")
        print(f"Overall Recall: {recall:.4f}")
        print(f"Overall F1: {f1:.4f}")

        return accuracy, precision, recall, f1, support, report

    def predict(self, text):
        """Predict a label after training the model"""
        print("predicting the label for given input")
        self.model.eval()
        preds = []
        for content in text:
            response = self.label_and_reason(content)
            analysis_str = response.replace('\n', ' ').replace('\r', ' ').strip()
            normalized_str = analysis_str.lower()

            pattern = r'[\[\(\{\<]\s*\b(unlikely|likely)\b\s*[\]\)\}\>]|\b(unlikely|likely)\b'
            label = re.search(pattern, normalized_str).group()
            if label:
                label = label.title()
            preds.append(label)
        return preds


if __name__ == "__main__":
    test_data = pd.read_csv('/fs01/projects/NMB-Plus/Caesar/Benchmarking/sample_test.csv')
    
    test_images = test_data['image'].tolist()
    test_inputs = test_data['content'].tolist()
    test_targets = test_data['multimodal_label'].tolist()

    classifier = PixtralClassifier(max_length=1024)

    print("\nTest Set Evaluation:")
    classifier.eval(test_images, test_inputs, test_targets)

    test_sample = test_inputs[:5]
    predicted_labels = classifier.predict(test_sample)
    results_df = pd.DataFrame({
        'Text': test_sample,
        'Predicted Label': predicted_labels,
        'True Label': [label for label in test_targets[:5]]
    })
    print("\nSample Predictions:")
    print(results_df)