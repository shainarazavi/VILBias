import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
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

model_name = 'llava_2shots'

tracker = EmissionsTracker(project_name=f"{model_name}", output_dir=f"./{model_name}", output_file=f"{model_name}_emissions.csv")


class LlaVaClassifier:
    def __init__(self, model_name="llava-hf/llava-v1.6-mistral-7b-hf", max_length=500):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LlavaNextProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, trust_remote_code=True)

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
        [INST]
        Here are two examples of analyzing images and text for potential disinformation:

        Example 1:
        <image1>
        Text: "Studies show that vaccines cause autism in children."
        Analysis: Likely (This statement is a common piece of disinformation not supported by scientific evidence.)

        Example 2:
        <image2>
        Text: "New evidence suggests the Earth might actually be flat."
        Analysis: Likely (This claim contradicts well-established scientific facts and is a known conspiracy theory.)

        Now, analyze the given image and the text below for potential disinformation.
        If you find any of the bias or disinformation in either the image or the text, then the article is likely to be bias or disinformation; if the image or the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
        If either the image or the text contain any bias or disinformation, please respond with 'Likely'. Otherwise, if both the image and the text do not contain any bias or disinformation, please respond with 'Unlikely'.

        Response format required: 
        [Likely/Unlikely]

        Text: {self.truncate_content(content)}
        [/INST]
        """
        image_path = "/fs01/projects/NMB-Plus/Caesar/Datasets/" + image_path
        try:
            compressed_image = None
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
            with torch.no_grad():
                conversation = [
                    {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": full_prompt},
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "image"},
                        ],
                    },
                ]
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                print("before inputs")
                inputs = self.processor(images=[image1, image2, compressed_image], text=prompt, return_tensors="pt").to("cuda:0")
                print("before generate")
                generate_ids = self.model.generate(**inputs, max_new_tokens=self.max_length, temperature=0.2)
                print("after generate")
                generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
                print("before decode")
                response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                print("after decode")
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
                label = re.search(pattern, normalized_str).group()
                if label:
                    label = label.title()
                print(i, label)
                if label == 'Likely' or label == 'Unlikely':
                    preds.append(label)
                    targets.append(true_vals[i])
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

    classifier = LlaVaClassifier(max_length=1024)

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