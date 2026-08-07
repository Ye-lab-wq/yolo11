import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 改为从具体的底层文件导入，避免通过 __init__.py 导致循环引用
from .conv import RepConv  #meda2
from .conv import Conv
from .block import C3k2, Bottleneck # 视你是否用了这些基础块
# 如果需要 Detect，它通常在 head.py
# from .head import Detect
from .dysample import DySample  # 👈 这一行是核心！
# ==================== ADown ====================
# 注意：ADown 已在 ultralytics.nn.modules 中内置，我们直接复用，但为了统一注册，这里也写一份
class ADown(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c = c2 // 2
        # 分支 1：卷积自带 stride=2，尺寸减半
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        # 分支 2：卷积 stride=1，不改变尺寸
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        # 先进行平均池化处理边缘
        x = F.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)  # 按通道切分成两半

        # 处理分支 1
        x1 = self.cv1(x1)  # 得到 16x16 (假设输入是 32)

        # ✅ 修正分支 2：增加一个 MaxPool 进行下采样，使尺寸也变成 16x16
        x2 = F.max_pool2d(x2, 3, 2, 1)  # kernel=3, stride=2, padding=1
        x2 = self.cv2(x2)

        # 现在两者都是 16x16，可以 cat 了
        return torch.cat((x1, x2), 1)

# ==================== MSEF ====================
# ==================== MSEF ====================
# class EdgeEnhancer(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels, 1)
#         self.bn = nn.BatchNorm2d(channels)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         smooth = F.avg_pool2d(x, 3, stride=1, padding=1)
#         # 计算高频边缘注意力权重 (值在 0~1 之间)
#         attention_weight = self.sigmoid(self.bn(self.conv(x - smooth)))
#
#         # ✅ 核心修复 1：将强行相加 (x + edge) 改为注意力乘法 (x * attention_weight)
#         # 这样起到了掩码过滤的作用，避免了给深层网络强加持续的正向偏置导致爆炸
#         return x * attention_weight
#
#
# class MSEF(nn.Module):
#     """Multi-Scale Edge Fusion with fixed kernel average pooling (Safe Version)"""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)  # 隐藏层通道数
#         self.cv_local = Conv(c1, self.c, 3, 1, 1)  # 提取局部特征 X_local
#
#         # ✅ 核心修复 2：使用固定核尺寸的池化，彻底抛弃 AdaptiveAvgPool + 极端插值
#         # 这种方式保证了特征图尺寸(H,W)自始至终不变，从根本上杜绝了缩放带来的梯度累积爆炸
#         self.pool_sizes = [3, 5, 7]
#         self.poolings = nn.ModuleList([
#             nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2) for k in self.pool_sizes
#         ])
#
#         # 每个分支的 1x1 卷积，将平滑特征映射到隐藏通道
#         self.branch_convs = nn.ModuleList([
#             Conv(c1, self.c, 1, 1, 0) for _ in self.pool_sizes
#         ])
#
#         # 边缘增强器
#         self.edge_enhancers = nn.ModuleList([
#             EdgeEnhancer(self.c) for _ in self.pool_sizes
#         ])
#
#         # 最终融合卷积
#         self.cv_final = Conv(self.c * (1 + len(self.pool_sizes)), c2, 1)
#
#     def forward(self, x):
#         # 1. 提取局部特征 X_local
#         local = self.cv_local(x)  # (B, self.c, H, W)
#
#         feats = [local]
#         for pool, conv, enhancer in zip(self.poolings, self.branch_convs, self.edge_enhancers):
#             # 2. 平均池化，剥离低频信息 (由于加了 padding，此时 H,W 依然保持不变)
#             smoothed = pool(x)
#             # 3. 映射通道
#             smoothed = conv(smoothed)
#             # 4. 计算残差 (提取高频边缘信息)
#             residual = local - smoothed
#             # 5. 经过边缘增强器过滤
#             edge_weight = enhancer(residual)
#             feats.append(edge_weight)
#
#         # 6. 特征融合
#         out = torch.cat(feats, dim=1)
#         out = self.cv_final(out)
#         return out
class EdgeEnhancer(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.conv = Conv(c1, c1, 1)  # 使用 YOLO 的 Conv 模块，自动处理 BN 和激活

    def forward(self, x):
        x_smooth = F.avg_pool2d(x, 3, stride=1, padding=1)
        edge = torch.sigmoid(self.conv(x - x_smooth))
        return x + edge

# class MSEF(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)
#         self.cv_local = Conv(c1, self.c, 3, 1, 1)
#         self.scales = [3, 6, 9, 12]
#         self.poolings = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in self.scales])
#         self.edge_enhancers = nn.ModuleList([EdgeEnhancer(self.c) for _ in self.scales])
#         self.cv_final = Conv(self.c * (len(self.scales) + 1), c2, 1)
#
#     def forward(self, x):
#         _, _, h, w = x.shape
#         local = self.cv_local(x)
#         feats = [local]
#         for pool, enhancer in zip(self.poolings, self.edge_enhancers):
#             pooled = pool(local)
#             up = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
#             feats.append(enhancer(up))
#         out = torch.cat(feats, 1)
#         return self.cv_final(out)
class MSEF(nn.Module):
    """Multi-Scale Edge Fusion with fixed kernel average pooling"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)                     # hidden channels
        self.cv_local = Conv(c1, self.c, 3, 1, 1)  # 提取局部特征 X_local

        # 固定核尺寸的平均池化（用于剥离低频拖影）
        self.pool_sizes = [3, 5, 7]              # 可调整，论文中用了 3×3,5×5
        self.poolings = nn.ModuleList([
            nn.AvgPool2d(kernel_size=k, stride=1, padding=k//2) for k in self.pool_sizes
        ])
        # 每个分支的 1x1 卷积，将平滑特征映射到隐藏通道
        self.branch_convs = nn.ModuleList([
            Conv(c1, self.c, 1, 1, 0) for _ in self.pool_sizes
        ])
        # 边缘增强器（可选，用于增强残差）
        self.edge_enhancers = nn.ModuleList([
            EdgeEnhancer(self.c) for _ in self.pool_sizes
        ])
        # 最终融合卷积
        self.cv_final = Conv(self.c * (1 + len(self.pool_sizes)), c2, 1)

    def forward(self, x):
        _, _, h, w = x.shape
        # 1. 提取局部特征 X_local
        local = self.cv_local(x)                 # (B, self.c, H, W)

        feats = [local]
        for pool, conv, enhancer in zip(self.poolings, self.branch_convs, self.edge_enhancers):
            # 2. 平均池化，剥离低频信息 → X_pool_k (保持 H,W)
            smoothed = pool(x)                   # (B, c1, H, W)
            # 3. 映射到隐藏通道
            smoothed = conv(smoothed)            # (B, self.c, H, W)
            # 4. 计算残差 ΔXk = X_local - X_pool_k
            residual = local - smoothed
            # 5. 边缘增强（论文中的公式后处理）
            edge_weight = enhancer(residual)     # (B, self.c, H, W)
            feats.append(edge_weight)

        # 6. 特征融合
        out = torch.cat(feats, dim=1)            # (B, self.c*(1+len), H, W)
        out = self.cv_final(out)                 # (B, c2, H, W)
        return out

# ==================== ELSN Shared Head ====================
class ELSNHead(nn.Module):
    """共享检测头（简化版，只实现参数共享，不改变输出结构）"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc
        self.ch = ch  # 多尺度输入通道列表
        # 共享卷积层（适用于所有尺度）
        self.shared_conv = nn.Sequential(
            nn.Conv2d(ch[0], ch[0], 3, padding=1, groups=ch[0]),  # DWConv
            nn.Conv2d(ch[0], ch[0], 1),                            # PWConv
            nn.SiLU(inplace=True)
        )
        # 为每个尺度学习的缩放因子
        self.scale_factors = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(len(ch))])
        # 每个尺度的分类和回归分支（参数少，可独立）
        self.cls_convs = nn.ModuleList([nn.Conv2d(ch[i], nc, 1) for i in range(len(ch))])
        self.reg_convs = nn.ModuleList([nn.Conv2d(ch[i], 4, 1) for i in range(len(ch))])

    def forward(self, xs):
        out = []
        for i, x in enumerate(xs):
            x = self.shared_conv(x)
            cls = self.cls_convs[i](x)
            reg = self.reg_convs[i](x)
            out.append(torch.cat([reg, cls], dim=1) * self.scale_factors[i])
        return out

# --- EMA (Efficient Multi-Scale Attention) ---
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d(1)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g, c//g, h, w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (x11 @ x12 + x21 @ x22).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

# --- RepNCSPELAN (YOLOv9/11 进化版结构) ---
class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=1):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(nn.Sequential(Conv(c3 // 2, c4, 3, 1), Conv(c4, c4, 3, 1)) if n == 1 else
                                 nn.Sequential(nn.Sequential(Conv(c3 // 2, c4, 3, 1), Conv(c4, c4, 3, 1)),
                                 *[nn.Sequential(Conv(c4, c4, 3, 1), Conv(c4, c4, 3, 1)) for _ in range(n - 1)]))
        self.cv3 = nn.Sequential(nn.Sequential(Conv(c4, c4, 3, 1), Conv(c4, c4, 3, 1)) if n == 1 else
                                 nn.Sequential(nn.Sequential(Conv(c4, c4, 3, 1), Conv(c4, c4, 3, 1)),
                                 *[nn.Sequential(Conv(c4, c4, 3, 1), Conv(c4, c4, 3, 1)) for _ in range(n - 1)]))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m := self.cv2(y[-1]), self.cv3(m)))
        return self.cv4(torch.cat(y, 1))