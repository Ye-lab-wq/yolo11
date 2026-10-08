import torch
import torch.nn as nn
import torch.nn.functional as F


def _normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight, mean, std)
    if getattr(module, "bias", None) is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    """Author-uploaded implementation: despite its name, this is static bilinear interpolation."""

    def __init__(self, in_channels, scale=2, groups=4):
        super().__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)


class DySampleOfficial(nn.Module):
    """Faithful LP/PL DySample implementation from the ICCV 2023 reference algorithm."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        if style not in {"lp", "pl"}:
            raise ValueError(f"unsupported DySample style: {style}")
        if style == "pl" and (in_channels < scale**2 or in_channels % scale**2):
            raise ValueError("PL DySample requires channels divisible by scale**2")
        if in_channels < groups or in_channels % groups:
            raise ValueError("DySample requires channels divisible by groups")
        offset_channels = 2 * groups if style == "pl" else 2 * groups * scale**2
        offset_in_channels = in_channels // scale**2 if style == "pl" else in_channels
        self.offset = nn.Conv2d(offset_in_channels, offset_channels, 1)
        _normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(offset_in_channels, offset_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid(h, h, indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def sample(self, x, offset):
        batch, _, height, width = offset.shape
        offset = offset.view(batch, 2, -1, height, width)
        coords_h = torch.arange(height, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(width, device=x.device, dtype=x.dtype) + 0.5
        coords = (
            torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
        )
        normalizer = torch.tensor([width, height], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(batch, -1, height, width), self.scale)
            .view(batch, 2, -1, self.scale * height, self.scale * width)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(batch * self.groups, -1, height, width),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(batch, -1, self.scale * height, self.scale * width)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        shuffled = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(shuffled) * self.scope(shuffled).sigmoid(), self.scale) * 0.5
        else:
            offset = F.pixel_unshuffle(self.offset(shuffled), self.scale) * 0.25
        return self.sample(x, offset + self.init_pos)

    def forward(self, x):
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)
