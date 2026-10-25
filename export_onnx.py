"""Export the PatchCore feature extractor to ONNX.

WideResNet-50 (ImageNet) -> layer2 + layer3 -> 3x3 local aggregation -> upsample layer3
-> concat -> channel reduction.

    input  "image"   float32 [1, 3, 224, 224]   (ImageNet-normalised RGB)
    output "patches" float32 [1, 784, DIM]      (28x28 patch descriptors)

Requires: torch>=2.5, torchvision
Usage:    python export_onnx.py [--dim 384] [--out backbone.onnx]
"""
import argparse

import torch
import torch.nn.functional as F
import torchvision


class PatchFeatures(torch.nn.Module):
    def __init__(self, dim: int, pretrained: bool):
        super().__init__()
        assert 1536 % dim == 0, "dim must divide 1536 (512 + 1024 channels)"
        net = torchvision.models.wide_resnet50_2(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        self.stem = torch.nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool, net.layer1
        )
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.dim = dim

    def forward(self, x):
        f2 = self.layer2(self.stem(x))  # [1,  512, 28, 28]  mid-level, finer
        f3 = self.layer3(f2)  # [1, 1024, 14, 14]  mid-level, coarser
        f2 = F.avg_pool2d(f2, 3, 1, 1)  # local neighbourhood aggregation
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], dim=1)  # [1, 1536, 28, 28]
        b, c, h, w = f.shape
        f = f.permute(0, 2, 3, 1).reshape(b, h * w, self.dim, c // self.dim)
        return f.mean(-1)  # average neighbouring channels -> [1, 784, dim]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--out", default="backbone.onnx")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights (smoke tests only)")
    args = ap.parse_args()

    model = PatchFeatures(args.dim, not args.no_pretrained).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 224, 224),
        args.out,
        input_names=["image"],
        output_names=["patches"],
        opset_version=17,
        dynamo=False,
    )
    print(f"wrote {args.out}")
