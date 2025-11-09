"""
Hybrid DeepLabV3+ × TransUNet Segmentation Model
Best performance: Dice 88.63%, IoU 79.62%
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.atrous1  = nn.Conv2d(in_ch, out_ch, 1)
        self.atrous6  = nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6)
        self.atrous12 = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12)
        self.atrous18 = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18)
        self.pool     = nn.AdaptiveAvgPool2d(1)
        self.pool_conv= nn.Conv2d(in_ch, out_ch, 1)
        self.fuse     = nn.Conv2d(out_ch*5, out_ch, 1)

    def forward(self, x):
        size = x.shape[2:]
        gp = self.pool(x)
        gp = F.interpolate(self.pool_conv(gp), size=size, mode='bilinear', align_corners=False)
        x1 = self.atrous1(x); x2 = self.atrous6(x); x3 = self.atrous12(x); x4 = self.atrous18(x)
        out = torch.cat([x1,x2,x3,x4,gp], dim=1)
        return self.fuse(out)


class TransformerDecoder(nn.Module):
    def __init__(self, dim, num_heads=8, ff_dim=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=False)
        self.ffn  = nn.Sequential(nn.Linear(dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, dim))
        self.n1   = nn.LayerNorm(dim)
        self.n2   = nn.LayerNorm(dim)

    def forward(self, x_seq):  # [HW, B, C]
        a, _ = self.attn(x_seq, x_seq, x_seq)
        x = self.n1(x_seq + a)
        f = self.ffn(x)
        return self.n2(x + f)


class DecoderSharpen(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class DeepLabV3Plus_TransUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2
        self.enc1 = backbone.layer1  # /4
        self.enc2 = backbone.layer2  # /8
        self.enc3 = backbone.layer3  # /16
        self.enc4 = backbone.layer4  # /32 (2048c)

        self.aspp   = ASPP(2048, 256)
        self.trans  = TransformerDecoder(256)
        self.sharp  = DecoderSharpen(256)
        self.skip1x1= nn.Conv2d(256, 256, 1)
        self.head   = nn.Conv2d(256, num_classes, 1)

    def encoder(self, x):
        x = self.stem(x)
        skip = self.enc1(x)
        x = self.enc2(skip)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.aspp(x)
        return x, skip

    def forward(self, x_in):
        in_h, in_w = x_in.shape[2], x_in.shape[3]
        x, skip = self.encoder(x_in)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2,0,1)           # [HW,B,C]
        x = self.trans(x).permute(1,2,0).contiguous().view(b, c, h, w)
        x = self.sharp(x)
        skip_up = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
        x = x + self.skip1x1(skip_up)
        x = self.head(x)
        x = F.interpolate(x, size=(in_h, in_w), mode="bilinear", align_corners=False)  # match targets
        return x


def load_segmentation_model(model_path=None, device='cpu'):
    """
    Load the Hybrid DeepLabV3+ × TransUNet segmentation model.
    
    Args:
        model_path: Path to model weights (.pt file). If None, returns untrained model.
        device: 'cpu' or 'cuda'
    
    Returns:
        Loaded model in evaluation mode
    """
    model = DeepLabV3Plus_TransUNet(num_classes=1)
    
    if model_path and os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded segmentation model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}. Using untrained model.")
    else:
        print("Warning: No model weights found. Using untrained model.")
    
    model.to(device)
    model.eval()
    return model


def segment_image(model, image_tensor, threshold=0.5, device='cpu'):
    """
    Segment an image to find melasma regions.
    
    Args:
        model: Segmentation model
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        threshold: Threshold for binarizing mask
        device: 'cpu' or 'cuda'
    
    Returns:
        Binary mask (numpy array) where melasma regions are white (255)
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).float()
        mask = (mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
    return mask

