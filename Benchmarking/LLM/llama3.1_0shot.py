import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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

torch.cuda.empty_cache()
torch.cuda.synchronize()

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "llama3.1_0shot"

tracker = EmissionsTracker(project_name=f"{model_name}", output_dir=f"./{model_name}", output_file=f"{model_name}_emissions.csv")

class LlamaClassifier:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_length=500):
        """Initialize model Class"""
        start_time = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.max_length = max_length
        end_time = time.time() - start_time
        print("Time Taken for Initializing Model: ", end_time)

    def label_and_reason(self, content):
        full_prompt = f"""
        <|begin_of_text|>[INST]
        Assess the text below for potential disinformation.
        If you find any of the bias or disinformation, then the article is likely to be bias or disinformation; if the text doesn't contain any of that, it is unlikely to contain bias or disinformation.
        Provide whether the text is 'Likely' or 'Unlikely' to be biased or disinformative without any further explanation.
        
        Response format required: 
        [Likely/Unlikely]

        Text: {content}
        [/INST]<|end_of_text|>
        """
        try:
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, temperature=0.2, max_new_tokens=self.max_length)  
                            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
            
    def eval(self, text, true_vals):
        """Test the trained model"""

        self.model.eval()
        preds, targets = [], []
        tracker.start()
        for i, content in enumerate(text):
            response = self.label_and_reason(content)
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


        return accuracy, precision, recall, f1, support
        
    def predict(self, text):
        """Predict a label after training the model"""
        print("predicting the label for given input")
        self.model.eval()
        preds = []
        for content in text:
            response = self.label_and_reason(content)
            if response:
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
    
    test_inputs = test_data['content'].tolist()
    test_targets = test_data['text_label'].tolist()

    classifier = LlamaClassifier(max_length=1024)

    print("\nTest Set Evaluation:")
    classifier.eval(test_inputs, test_targets)

    test_sample = test_inputs[:5]
    predicted_labels = classifier.predict(test_sample)
    results_df = pd.DataFrame({
        'Text': test_sample,
        'Predicted Label': predicted_labels,
        'True Label': [label for label in test_targets[:5]]
    })
    print("\nSample Predictions:")
    print(results_df)