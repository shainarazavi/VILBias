import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

import os

class ImageEncoder:
    def __init__(self, fine_tune_resnet=False, clip_model="openai/clip-vit-base-patch32") -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fine_tune_resnet = fine_tune_resnet
        self.clip_model = clip_model
        self.resnet_model = models.resnet50(pretrained=True)
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model.to(self.device)
        self.resnet_model.to(self.device)


    # Load ResNet-50
    def get_resnet50_model(self):
        resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer
        resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
        
        if not self.fine_tune_resnet:
            # Freeze all layers if not fine-tuning
            for param in resnet50.parameters():
                param.requires_grad = False
        
        return resnet50

    # Image preprocessing function for ResNet
    def preprocess_resnet(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return preprocess(image)

    # Function to extract features from an image
    def extract_image_features(self, image):
        # Get ResNet-50 model with or without fine-tuning
        resnet50 = self.get_resnet50_model(self.fine_tune_resnet)
        
        # Preprocess the image for ResNet
        image_resnet = self.preprocess_resnet(image)
        image_resnet = image_resnet.unsqueeze(0)  # Add batch dimension

        # Extract ResNet features
        with torch.no_grad():
            resnet_features = resnet50(image_resnet)
            resnet_features = torch.flatten(resnet_features, 1)  # Flatten to (batch_size, 2048)

        # Preprocess the image for CLIP
        inputs = self.clip_processor(images=image, return_tensors="pt")
        
        # Extract CLIP features
        with torch.no_grad():
            clip_features = self.clip_model.get_image_features(**inputs)
            clip_features = clip_features / clip_features.norm(p=2, dim=-1, keepdim=True)  # Normalize features

        # Concatenate ResNet and CLIP features
        concatenated_features = torch.cat((resnet_features, clip_features), dim=1)  # (batch_size, 2560)

        return clip_features, concatenated_features



img_file_dir = os.path.join("/projects/NMB-Plus/consolidated_data/", "images")
cont_file_dir = os.path.join("/projects/NMB-Plus/consolidated_data/", "cleaned_data.csv")




# Example usage
from PIL import Image

# Load an example image
image_path = os.path.join(img_file_dir, "b6f9ebec03.jpg")
image = Image.open(image_path)

# Extract and concatenate features with or without ResNet fine-tuning
img_enc = ImageEncoder(fine_tune_resnet=False)
_, concatenated_features = img_enc.extract_image_features(image)
print(concatenated_features.shape)  # Output: torch.Size([1, 2560])
