import torch
import torch.nn as nn
import torch.nn.functional as F
import clip  # pip install git+https://github.com/openai/CLIP.git
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"


class MultiHeadGeoCLIP(nn.Module):
    def __init__(
        self,
        n_s3: int = 3,
        n_s16: int = 16,
        n_s365: int = 365,
        clip_model_name: str = "ViT-B/32",
        freeze_clip: bool = True,
    ):
        super().__init__()

        # 1) CLIP backbone
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad = False

        feat_dim = self.clip_model.visual.output_dim  # 512 for ViT-B/32

        # 2) Classification heads
        self.head_s3 = nn.Linear(feat_dim, n_s3)
        self.head_s16 = nn.Linear(feat_dim, n_s16)
        self.head_s365 = nn.Linear(feat_dim, n_s365)

    def encode_image(self, images):
        # images should already be passed through self.preprocess
        feats = self.clip_model.encode_image(images)
        return feats.float()  # avoid Half/Float mismatch

    def forward(self, images):
        feats = self.encode_image(images)  # [B, D]

        out = {
            "logits_s3": self.head_s3(feats),        # [B, n_s3]
            "logits_s16": self.head_s16(feats),      # [B, n_s16]
            "logits_s365": self.head_s365(feats),    # [B, n_s365]
            "feats": feats,                          # [B, D]
        }
        return out

class ResNet18MultiHead(nn.Module):
    def __init__(
        self,
        n_s3: int = 3,
        n_s16: int = 16,
        n_s365: int = 365,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # --- Backbone ---
        # Use ImageNet-pretrained resnet18 as feature extractor
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Get feature dimension of final FC layer, then remove it
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # backbone -> feature vector

        # Optionally freeze backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # --- Heads ---
        self.head_s3 = nn.Linear(in_features, n_s3)
        self.head_s16 = nn.Linear(in_features, n_s16)
        self.head_s365 = nn.Linear(in_features, n_s365)
    
    def encode_image(self, x):
        """
        x: (B, 3, H, W), normalized as usual for ImageNet/ResNet
        returns:
            feats: (B, D) feature vectors from backbone
        """
        feats = self.backbone(x)  # (B, in_features)
        return feats

    def forward(self, x):
        """
        x: (B, 3, H, W), normalized as usual for ImageNet/ResNet
        returns:
            logits_s3:   (B, n_s3)
            logits_s16:  (B, n_s16)
            logits_s365: (B, n_s365)
        """
        feats = self.backbone(x)  # (B, in_features)

        logits_s3 = self.head_s3(feats)
        logits_s16 = self.head_s16(feats)
        logits_s365 = self.head_s365(feats)

        return {
            "logits_s3": logits_s3,
            "logits_s16": logits_s16,
            "logits_s365": logits_s365,
            "feats": feats,  # in case you still want to use the shared embedding
        }