
import numpy as np
import torch
import timm
from torchvision import transforms
from PIL import Image
import timm.data

class FeatureExtractor:
    def __init__(self, model_name, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.batch_size = batch_size

        # Load model
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)  # num_classes=0 removes final classification layer
        self.model.to(self.device)
        self.model.eval()
        
        # Get the appropriate transformation for the model
        self.transform = timm.data.transforms_factory.create_transform(input_size=self.model.default_cfg['input_size'])

    def load_and_transform_images(self, file_list):
        images = []
        for file_path in file_list:
            image = Image.open(file_path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        return torch.stack(images)

    def extract_features(self, file_list):
        all_features = []
        for i in range(0, len(file_list), self.batch_size):
            batch_files = file_list[i:i+self.batch_size]
            images = self.load_and_transform_images(batch_files)
            images = images.to(self.device)
            
            with torch.no_grad():
                features = self.model(images)
            
            all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
