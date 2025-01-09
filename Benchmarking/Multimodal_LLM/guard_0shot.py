import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
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

model_name = 'guard_0shot'

tracker = EmissionsTracker(project_name=f"{model_name}", output_dir=f"./{model_name}", output_file=f"{model_name}_emissions.csv")


class LlamaClassifier:
    def __init__(self, model_name="meta-llama/Llama-Guard-3-11B-Vision", max_length=500):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)

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
        Analyze the given image and the text below for potential disinformation.
        If you find any of the bias or disinformation in either the image or the text, then the article is likely to be bias or disinformation; if the image or the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
        If either the image or the text contain any bias or disinformation, please respond with 'Likely'. Otherwise, if both the image and the text do not contain any bias or disinformation, please respond with 'Unlikely'.

        Response format required: 
        [Likely/Unlikely]

        Text: {self.truncate_content(content)}
        """
        print("image path is.... ", image_path)
        image_path = "/fs01/projects/NMB-Plus/Caesar/Datasets/" + image_path
        try:
            compressed_image = None
            if image_path is None:
                return
            else:
                print("image found")
                compressed_image_data = self.compress_image(image_path)
                compressed_image = Image.open(io.BytesIO(compressed_image_data))            
        except Exception as e:
            print(f"An error while compressing the image: {str(e)}")
            return None
        
        try:            
            with torch.no_grad():
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": full_prompt
                            },
                            {
                                "type": "image",
                            },
                        ],
                    }
                ]

                input_prompt = self.processor.apply_chat_template(conversation, return_tensors="pt")

                inputs = self.processor(text=input_prompt, images=compressed_image, return_tensors="pt").to("cuda:0")

                prompt_len = len(inputs['input_ids'][0])
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=0,
                )

                generated_tokens = output[:, prompt_len:]
                label = self.processor.decode(generated_tokens[0])
                return label
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
                cleaned = response.strip().replace('\n', '').replace(' ', '')
            
                first_word = cleaned.split('<|eot_id|>')[0]
                label = None
                if not first_word:
                    label = None
                elif first_word == 'safe':
                    label = 'Unlikely'
                else:
                    label = 'Likely'
                    print("likely increased by one")

                if label == 'Likely' or label == 'Unlikely':
                    preds.append(label)
                    targets.append(true_vals[i])
                    print(len(preds))

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

    classifier = LlamaClassifier(max_length=1024)

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