import json
import torch
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Classifier:
    def __init__(self, model_name="efficientnet-b0", mapping_path="category_mapping.json", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNet.from_pretrained(model_name).to(self.device).eval()
        input_size = EfficientNet.get_image_size(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        with open(mapping_path) as f:
            self.mapping = json.load(f)

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top_idx = int(torch.argmax(probs).item())
            confidence = float(probs[top_idx].item())
        entry = self.mapping.get(str(top_idx), {"class_name": "unknown", "category": None})
        return {
            "class_name": entry["class_name"],
            "category": entry["category"],
            "confidence": confidence,
        }