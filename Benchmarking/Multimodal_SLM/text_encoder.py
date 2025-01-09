import pandas as pd
import torch
import os
from transformers import BertTokenizerFast, BertModel, CLIPProcessor, CLIPModel
from torch.nn.functional import softmax


class TextEncoder:
    def __init__(self, bert_model='bert-base-uncased', clip_model='openai/clip-vit-base-patch32'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        self.bert_model = BertModel.from_pretrained(bert_model)
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_model.to(self.device)
        self.bert_model.to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)


    def extract_text_features(self, text):
        # Preprocess the text
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            clip_text_features = self.clip_model.get_text_features(**inputs)

        # Preprocess the text for BERT
        bert_inputs = self.bert_tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            bert_outputs = self.bert_model(**bert_inputs)
            bert_text_embedding = bert_outputs.last_hidden_state[:, 0, :]  # Take the [CLS] token embedding

        # Normalize the BERT text embedding
        bert_text_embedding = bert_text_embedding / bert_text_embedding.norm(p=2, dim=-1, keepdim=True)

        # Concatenate BERT and CLIP features
        concatenated_features = torch.cat((bert_text_embedding, clip_text_features), dim=1)  # (batch_size, 1276)

        return clip_text_features, concatenated_features

