import json
import torch
import open_clip
from PIL import Image

DEFAULT_TEMPLATE = "a photo of {}"


class ClipClassifier:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", prompts_file=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.categories = []
        self.prompts = []
        prompt_map = json.load(open(prompts_file)) if prompts_file else None
        if not prompt_map:
            raise ValueError("prompts_file with category->prompt list is required")
        for cat, phrases in prompt_map.items():
            self.categories.append(cat)
            self.prompts.append(phrases if isinstance(phrases, list) else [phrases])

        self._build_text_features()

    def _build_text_features(self):
        all_feats = []
        for phrases in self.prompts:
            tokens = self.tokenizer(phrases).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_feats.append(feats.mean(dim=0))
        self.text_features = torch.stack(all_feats)
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(x)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ self.text_features.T)[0]
            probs = sims.softmax(dim=0)
            top_idx = int(torch.argmax(probs).item())
        return {
            "category": self.categories[top_idx],
            "confidence": float(probs[top_idx].item()),
        }