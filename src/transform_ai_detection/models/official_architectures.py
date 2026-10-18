"""Official paper architectures for online pretrained checkpoints.

Contains faithful PyTorch implementations of:
1. OfficialPSCCNet (proteus1991/PSCC-Net: HRNet-W18-Small-v2 + NLCDetection + DetectionHead)
2. OfficialTruForNet (grip-unina/TruFor: Dual-SegFormer mit_b2 + DnCNN Noiseprint++ + MLPDecoder)
3. OfficialSparseViTNet (scu-zjz/SparseViT: SparseViT_Mul with CBlock/SABlock + Multiple head)
4. OfficialIMLViTNet (SunnyHaze/IML-ViT: Window-Attention ViT-B + SimpleFeaturePyramid + PredictHead)
5. OfficialCoDENet (siriusPRX/CoDE: AC_CoDE with EfficientNet-B0 + Gaussian policy/value heads)
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath, Mlp as TimmMlp, to_2tuple, trunc_normal_
except ImportError:
    from timm.models.layers import DropPath, Mlp as TimmMlp, to_2tuple, trunc_normal_


# =============================================================================
# 1. Official PSCC-Net (proteus1991/PSCC-Net)
# =============================================================================

BN_MOMENTUM_PSCC = 0.01


def _pscc_conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _PSCCBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = _pscc_conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM_PSCC)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = _pscc_conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM_PSCC)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _PSCCBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM_PSCC)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM_PSCC)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM_PSCC)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class _PSCCHighResolutionModule(nn.Module):
    def __init__(
        self,
        num_branches: int,
        blocks: Any,
        num_blocks: List[int],
        num_inchannels: List[int],
        num_channels: List[int],
        fuse_method: str,
        multi_scale_output: bool = True,
    ) -> None:
        super().__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=False)

    def _make_one_branch(self, branch_index: int, block: Any, num_blocks: List[int], num_channels: List[int], stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM_PSCC),
            )
        layers = [block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample)]
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)

    def _make_branches(self, num_branches: int, block: Any, num_blocks: List[int], num_channels: List[int]) -> nn.ModuleList:
        return nn.ModuleList([self._make_one_branch(i, block, num_blocks, num_channels) for i in range(num_branches)])

    def _make_fuse_layers(self) -> Optional[nn.ModuleList]:
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM_PSCC),
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM_PSCC),
                            ))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM_PSCC),
                                nn.ReLU(inplace=False),
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self) -> List[int]:
        return self.num_inchannels

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[x[i].shape[-2], x[i].shape[-1]],
                        mode="bilinear",
                        align_corners=True,
                    )
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


_PSCC_BLOCKS_DICT = {"BASIC": _PSCCBasicBlock, "BOTTLENECK": _PSCCBottleneck}

_PSCC_HRNET_CFG = {
    "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "NUM_BLOCKS": [2], "NUM_CHANNELS": [64], "BLOCK": "BOTTLENECK", "FUSE_METHOD": "SUM"},
    "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [18, 36], "BLOCK": "BASIC", "FUSE_METHOD": "SUM"},
    "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [18, 36, 72], "BLOCK": "BASIC", "FUSE_METHOD": "SUM"},
    "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [18, 36, 72, 144], "BLOCK": "BASIC", "FUSE_METHOD": "SUM"},
}


class PSCCHRNetBackbone(nn.Module):
    """Official HRNet-W18-Small-v2 feature extractor from PSCC-Net."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM_PSCC)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM_PSCC)
        self.relu = nn.ReLU(inplace=False)

        self.stage1_cfg = _PSCC_HRNET_CFG["STAGE1"]
        num_channels = self.stage1_cfg["NUM_CHANNELS"][0]
        block = _PSCC_BLOCKS_DICT[self.stage1_cfg["BLOCK"]]
        num_blocks = self.stage1_cfg["NUM_BLOCKS"][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = _PSCC_HRNET_CFG["STAGE2"]
        num_channels = [c * _PSCC_BLOCKS_DICT[self.stage2_cfg["BLOCK"]].expansion for c in self.stage2_cfg["NUM_CHANNELS"]]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = _PSCC_HRNET_CFG["STAGE3"]
        num_channels = [c * _PSCC_BLOCKS_DICT[self.stage3_cfg["BLOCK"]].expansion for c in self.stage3_cfg["NUM_CHANNELS"]]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = _PSCC_HRNET_CFG["STAGE4"]
        num_channels = [c * _PSCC_BLOCKS_DICT[self.stage4_cfg["BLOCK"]].expansion for c in self.stage4_cfg["NUM_CHANNELS"]]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

    def _make_transition_layer(self, num_channels_pre_layer: List[int], num_channels_cur_layer: List[int]) -> nn.ModuleList:
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)
        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM_PSCC),
                        nn.ReLU(inplace=False),
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM_PSCC),
                        nn.ReLU(inplace=False),
                    ))
                transition_layers.append(nn.Sequential(*conv3x3s))
        return nn.ModuleList(transition_layers)

    def _make_layer(self, block: Any, inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM_PSCC),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config: Dict[str, Any], num_inchannels: List[int], multi_scale_output: bool = True) -> Tuple[nn.Sequential, List[int]]:
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = _PSCC_BLOCKS_DICT[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]
        modules = []
        for i in range(num_modules):
            reset_multi_scale_output = False if (not multi_scale_output and i == num_modules - 1) else True
            modules.append(_PSCCHighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.layer1(x)

        x_list = [self.transition1[i](x) if self.transition1[i] is not None else x for i in range(self.stage2_cfg["NUM_BRANCHES"])]
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[i] if i < self.stage2_cfg["NUM_BRANCHES"] else y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[i] if i < self.stage3_cfg["NUM_BRANCHES"] else y_list[-1]))
            else:
                x_list.append(y_list[i])
        return self.stage4(x_list)


class _PSCCNonLocalMask(nn.Module):
    def __init__(self, in_channels: int, reduce_scale: int) -> None:
        super().__init__()
        self.r = reduce_scale
        self.ic = in_channels * self.r * self.r
        self.mc = self.ic
        self.g = nn.Conv2d(self.ic, self.ic, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.ic, self.mc, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.ic, self.mc, kernel_size=1, stride=1, padding=0)
        self.W_s = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.W_c = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.gamma_s = nn.Parameter(torch.ones(1))
        self.gamma_c = nn.Parameter(torch.ones(1))
        self.getmask = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x1 = x.reshape(b, self.ic, h // self.r, w // self.r)
        g_x = self.g(x1).view(b, self.ic, -1).permute(0, 2, 1)
        theta_x = self.theta(x1).view(b, self.mc, -1)
        theta_x_s = theta_x.permute(0, 2, 1)
        theta_x_c = theta_x
        phi_x = self.phi(x1).view(b, self.mc, -1)
        phi_x_s = phi_x
        phi_x_c = phi_x.permute(0, 2, 1)

        f_s_div = F.softmax(torch.matmul(theta_x_s, phi_x_s), dim=-1)
        f_c_div = F.softmax(torch.matmul(theta_x_c, phi_x_c), dim=-1)

        y_s = torch.matmul(f_s_div, g_x).permute(0, 2, 1).contiguous().view(b, c, h, w)
        y_c = torch.matmul(g_x, f_c_div).view(b, c, h, w)
        z = x + self.gamma_s * self.W_s(y_s) + self.gamma_c * self.W_c(y_c)
        mask = torch.sigmoid(self.getmask(z.clone()))
        return mask, z


class PSCCNLCDetection(nn.Module):
    """Official NLCDetection progressive spatio-channel correlation head."""

    def __init__(self, crop_size: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.crop_size = list(crop_size)
        feat1_num, feat2_num, feat3_num, feat4_num = [18, 36, 72, 144]
        self.getmask4 = _PSCCNonLocalMask(feat4_num, 1)
        self.getmask3 = _PSCCNonLocalMask(feat3_num, 2)
        self.getmask2 = _PSCCNonLocalMask(feat2_num, 2)
        self.getmask1 = _PSCCNonLocalMask(feat1_num, 4)

    def forward(self, feat: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s1, s2, s3, s4 = feat
        if list(s1.shape[2:]) != self.crop_size:
            s1 = F.interpolate(s1, size=self.crop_size, mode="bilinear", align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode="bilinear", align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode="bilinear", align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode="bilinear", align_corners=True)

        mask4, _ = self.getmask4(s4)
        mask4U = F.interpolate(mask4, size=s3.size()[2:], mode="bilinear", align_corners=True)
        s3 = s3 * mask4U
        mask3, _ = self.getmask3(s3)
        mask3U = F.interpolate(mask3, size=s2.size()[2:], mode="bilinear", align_corners=True)
        s2 = s2 * mask3U
        mask2, _ = self.getmask2(s2)
        mask2U = F.interpolate(mask2, size=s1.size()[2:], mode="bilinear", align_corners=True)
        s1 = s1 * mask2U
        mask1, _ = self.getmask1(s1)
        return mask1, mask2, mask3, mask4


class _PSCCDetBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM_PSCC)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM_PSCC)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM_PSCC)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class PSCCDetectionHead(nn.Module):
    """Official DetectionHead from PSCC-Net."""

    def __init__(self, crop_size: Tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        self.crop_size = list(crop_size)
        pre_stage_channels = [18, 36, 72, 144]
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)
        self.classifier = nn.Sequential(
            nn.Linear(128, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2),
        )

    def _make_layer(self, block: Any, inplanes: int, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM_PSCC),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_head(self, pre_stage_channels: List[int]) -> Tuple[nn.ModuleList, nn.ModuleList, nn.Sequential]:
        head_block = _PSCCDetBottleneck
        head_channels = pre_stage_channels
        incre_modules = nn.ModuleList([
            self._make_layer(head_block, channels, head_channels[i], 1, stride=1)
            for i, channels in enumerate(pre_stage_channels)
        ])
        downsamp_modules = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(head_channels[i] * head_block.expansion, head_channels[i + 1] * head_block.expansion, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(head_channels[i + 1] * head_block.expansion, momentum=BN_MOMENTUM_PSCC),
                nn.ReLU(inplace=True),
            )
            for i in range(len(pre_stage_channels) - 1)
        ])
        final_layer = nn.Sequential(
            nn.Conv2d(head_channels[3] * head_block.expansion, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM_PSCC),
            nn.ReLU(inplace=True),
        )
        return incre_modules, downsamp_modules, final_layer

    def forward(self, feat: List[torch.Tensor]) -> torch.Tensor:
        s1, s2, s3, s4 = feat
        if list(s1.shape[2:]) != self.crop_size:
            s1 = F.interpolate(s1, size=self.crop_size, mode="bilinear", align_corners=True)
            s2 = F.interpolate(s2, size=[i // 2 for i in self.crop_size], mode="bilinear", align_corners=True)
            s3 = F.interpolate(s3, size=[i // 4 for i in self.crop_size], mode="bilinear", align_corners=True)
            s4 = F.interpolate(s4, size=[i // 8 for i in self.crop_size], mode="bilinear", align_corners=True)
        y_list = [s1, s2, s3, s4]
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)
        y = self.final_layer(y)
        y = F.avg_pool2d(y, kernel_size=y.size()[2:]).view(y.size(0), -1)
        return self.classifier(y)


class OfficialPSCCNet(nn.Module):
    """Complete official PSCC-Net combining FENet (HRNet), SegNet (NLCDetection), and ClsNet."""

    def __init__(self) -> None:
        super().__init__()
        self.FENet = PSCCHRNetBackbone()
        self.SegNet = PSCCNLCDetection(crop_size=(256, 256))
        self.ClsNet = PSCCDetectionHead(crop_size=(256, 256))

    def load_official_components(self, components: Dict[str, Dict[str, Any]]) -> None:
        self.FENet.load_state_dict(components["FENet"], strict=True)
        self.SegNet.load_state_dict(components["SegNet"], strict=True)
        if "ClsNet" in components:
            self.ClsNet.load_state_dict(components["ClsNet"], strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        feat = self.FENet(x)
        pred_mask = self.SegNet(feat)[0]
        pred_mask = F.interpolate(pred_mask, size=(h, w), mode="bilinear", align_corners=True)
        return pred_mask


# =============================================================================
# 2. Official TruFor (grip-unina/TruFor)
# =============================================================================

def _trufor_make_dncnn(num_levels: int = 17, out_channel: int = 1) -> nn.Module:
    layers: List[Tuple[str, nn.Module]] = []
    for i in range(num_levels):
        in_ch = 3 if i == 0 else 64
        out_ch = out_channel if i == num_levels - 1 else 64
        use_bn = (0 < i < num_levels - 1)
        sub: List[Tuple[str, nn.Module]] = [
            ("conv", nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=1, padding=1, bias=not use_bn))
        ]
        if use_bn:
            sub.append(("bn", nn.BatchNorm2d(out_ch, momentum=0.1, eps=1e-5)))
        if i < num_levels - 1:
            sub.append(("act", nn.ReLU(inplace=True)))
        layers.append((f"level{i + 1:d}", nn.Sequential(OrderedDict(sub))))

    class _DnCNN(nn.Module):
        def __init__(self, seq: nn.Sequential) -> None:
            super().__init__()
            self.net = seq

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    return _DnCNN(nn.Sequential(OrderedDict(layers)))


class _TruForChannelWeights(nn.Module):
    def __init__(self, dim: int, reduction: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = x1.shape
        x = torch.cat((x1, x2), dim=1)
        y = torch.cat((self.avg_pool(x).view(B, self.dim * 2), self.max_pool(x).view(B, self.dim * 2)), dim=1)
        return self.mlp(y).reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)


class _TruForSpatialWeights(nn.Module):
    def __init__(self, dim: int, reduction: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // reduction, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        return self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)


class _TruForFRM(nn.Module):
    def __init__(self, dim: int, reduction: int = 1, lambda_c: float = 0.5, lambda_s: float = 0.5) -> None:
        super().__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = _TruForChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = _TruForSpatialWeights(dim=dim, reduction=reduction)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cw = self.channel_weights(x1, x2)
        sw = self.spatial_weights(x1, x2)
        return (
            x1 + self.lambda_c * cw[1] * x2 + self.lambda_s * sw[1] * x2,
            x2 + self.lambda_c * cw[0] * x1 + self.lambda_s * sw[0] * x1,
        )


class _TruForCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Optional[float] = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        ctx1 = ((k1.transpose(-2, -1) @ v1) * self.scale).softmax(dim=-2)
        ctx2 = ((k2.transpose(-2, -1) @ v2) * self.scale).softmax(dim=-2)
        return (
            (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous(),
            (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous(),
        )


class _TruForCrossPath(nn.Module):
    def __init__(self, dim: int, reduction: int = 1, num_heads: int = 8, norm_layer: Any = nn.LayerNorm) -> None:
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = _TruForCrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        return (
            self.norm1(x1 + self.end_proj1(torch.cat((y1, v1), dim=-1))),
            self.norm2(x2 + self.end_proj2(torch.cat((y2, v2), dim=-1))),
        )


class _TruForChannelEmbed(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 1, norm_layer: Any = nn.BatchNorm2d) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels // reduction, out_channels // reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        return self.norm(self.residual(x) + self.channel_embed(x))


class _TruForFFM(nn.Module):
    def __init__(self, dim: int, reduction: int = 1, num_heads: int = 8, norm_layer: Any = nn.BatchNorm2d) -> None:
        super().__init__()
        self.cross = _TruForCrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = _TruForChannelEmbed(in_channels=dim * 2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x1.shape
        x1, x2 = self.cross(x1.flatten(2).transpose(1, 2), x2.flatten(2).transpose(1, 2))
        return self.channel_emb(torch.cat((x1, x2), dim=-1), H, W)


class _TruForDWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        return self.dwconv(x).flatten(2).transpose(1, 2)


class _TruForMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Any = nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _TruForDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.dwconv(self.fc1(x), H, W)))))


class _TruForAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Optional[float] = None, attn_drop: float = 0.0, proj_drop: float = 0.0, sr_ratio: int = 1) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = self.norm(self.sr(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, -1).permute(0, 2, 1))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


class _TruForBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, qk_scale: Optional[float] = None, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, act_layer: Any = nn.GELU, norm_layer: Any = nn.LayerNorm, sr_ratio: int = 1) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _TruForAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _TruForMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        return x + self.drop_path(self.mlp(self.norm2(x), H, W))


class _TruForOverlapPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        p_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=p_size, stride=stride, padding=(p_size[0] // 2, p_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), H, W


class _TruForMitB2(nn.Module):
    def __init__(self, norm_fuse: Any = nn.BatchNorm2d) -> None:
        super().__init__()
        embed_dims = [64, 128, 320, 512]
        num_heads = [1, 2, 5, 8]
        mlp_ratios = [4, 4, 4, 4]
        depths = [3, 4, 6, 3]
        sr_ratios = [8, 4, 2, 1]
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = _TruForOverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        self.patch_embed2 = _TruForOverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = _TruForOverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = _TruForOverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.extra_patch_embed1 = _TruForOverlapPatchEmbed(patch_size=7, stride=4, in_chans=3, embed_dim=embed_dims[0])
        self.extra_patch_embed2 = _TruForOverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.extra_patch_embed3 = _TruForOverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.extra_patch_embed4 = _TruForOverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([_TruForBlock(embed_dims[0], num_heads[0], mlp_ratios[0], True, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.extra_block1 = nn.ModuleList([_TruForBlock(embed_dims[0], num_heads[0], mlp_ratios[0], True, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.extra_norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([_TruForBlock(embed_dims[1], num_heads[1], mlp_ratios[1], True, drop_path=dpr[cur], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.extra_block2 = nn.ModuleList([_TruForBlock(embed_dims[1], num_heads[1], mlp_ratios[1], True, drop_path=dpr[cur + 1], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for _ in range(depths[1])])
        self.extra_norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([_TruForBlock(embed_dims[2], num_heads[2], mlp_ratios[2], True, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.extra_block3 = nn.ModuleList([_TruForBlock(embed_dims[2], num_heads[2], mlp_ratios[2], True, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.extra_norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([_TruForBlock(embed_dims[3], num_heads[3], mlp_ratios[3], True, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.extra_block4 = nn.ModuleList([_TruForBlock(embed_dims[3], num_heads[3], mlp_ratios[3], True, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.extra_norm4 = norm_layer(embed_dims[3])

        self.FRMs = nn.ModuleList([_TruForFRM(dim=d, reduction=1) for d in embed_dims])
        self.FFMs = nn.ModuleList([_TruForFFM(dim=embed_dims[i], reduction=1, num_heads=num_heads[i], norm_layer=norm_fuse) for i in range(4)])

    def forward(self, x_rgb: torch.Tensor, x_e: torch.Tensor) -> List[torch.Tensor]:
        B = x_rgb.shape[0]
        outs = []
        stages = [
            (self.patch_embed1, self.extra_patch_embed1, self.block1, self.extra_block1, self.norm1, self.extra_norm1, 0),
            (self.patch_embed2, self.extra_patch_embed2, self.block2, self.extra_block2, self.norm2, self.extra_norm2, 1),
            (self.patch_embed3, self.extra_patch_embed3, self.block3, self.extra_block3, self.norm3, self.extra_norm3, 2),
            (self.patch_embed4, self.extra_patch_embed4, self.block4, self.extra_block4, self.norm4, self.extra_norm4, 3),
        ]
        for pe, epe, blks, eblks, nrm, enrm, idx in stages:
            x_rgb, H, W = pe(x_rgb)
            x_e, _, _ = epe(x_e)
            for blk in blks:
                x_rgb = blk(x_rgb, H, W)
            for blk in eblks:
                x_e = blk(x_e, H, W)
            x_rgb = nrm(x_rgb).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_e = enrm(x_e).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x_rgb, x_e = self.FRMs[idx](x_rgb, x_e)
            outs.append(self.FFMs[idx](x_rgb, x_e))
        return outs


class _TruForMLP(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.flatten(2).transpose(1, 2))


class _TruForDecoderHead(nn.Module):
    def __init__(self, in_channels: List[int] = [64, 128, 320, 512], num_classes: int = 2, embed_dim: int = 512, norm_layer: Any = nn.BatchNorm2d) -> None:
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c4 = _TruForMLP(c4, embed_dim)
        self.linear_c3 = _TruForMLP(c3, embed_dim)
        self.linear_c2 = _TruForMLP(c2, embed_dim)
        self.linear_c1 = _TruForMLP(c1, embed_dim)
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]
        _c4 = F.interpolate(self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3]), size=c1.size()[2:], mode="bilinear", align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3]), size=c1.size()[2:], mode="bilinear", align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3]), size=c1.size()[2:], mode="bilinear", align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        return self.linear_pred(self.dropout(x))


class OfficialTruForNet(nn.Module):
    """Official TruFor architecture (myEncoderDecoder with mit_b2, DnCNN, and MLPDecoder)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = _TruForMitB2(norm_fuse=nn.BatchNorm2d)
        self.decode_head = _TruForDecoderHead(in_channels=[64, 128, 320, 512], num_classes=2, embed_dim=512)
        self.decode_head_conf = _TruForDecoderHead(in_channels=[64, 128, 320, 512], num_classes=1, embed_dim=512)
        self.detection = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
        )
        self.dncnn = _trufor_make_dncnn(num_levels=17, out_channel=1)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        _, _, h, w = rgb.shape
        modal_x = self.dncnn(rgb).repeat(1, 3, 1, 1)
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb.device).view(1, 3, 1, 1)
        rgb_norm = (rgb - mean) / std
        feats = self.backbone(rgb_norm, modal_x)
        logits = self.decode_head(feats)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return F.softmax(logits, dim=1)[:, 1:2]


# =============================================================================
# 3. Official SparseViT (scu-zjz/SparseViT)
# =============================================================================

class _SparseViTDWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, C = x.shape
        return self.dwconv(x.transpose(1, 2).view(B, C, H, W)).flatten(2).transpose(1, 2)


class _SparseViTMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Any = nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _SparseViTDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.dwconv(self.fc1(x), H, W)))))


class _SparseViTCMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: Any = nn.GELU, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _SparseViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, qk_scale: Optional[float] = None, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, N, C)))


def _sparsevit_alter_sparse(x: torch.Tensor, sparse_size: int = 8) -> Tuple[torch.Tensor, int, int, int]:
    x = x.permute(0, 2, 3, 1)
    B, H, W, C = x.shape
    grid_size = H // sparse_size
    pad_h = (grid_size - H % grid_size) % grid_size
    pad_w = (grid_size - W % grid_size) % grid_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    out = x.reshape(B, Hp // grid_size, grid_size, Wp // grid_size, grid_size, C).permute(0, 1, 3, 2, 4, 5).contiguous()
    out = out.permute(0, 3, 4, 1, 2, 5).contiguous().reshape(-1, sparse_size, sparse_size, C).permute(0, 3, 1, 2)
    return out, H, Hp, C


def _sparsevit_alter_unsparse(x: torch.Tensor, H: int, Hp: int, C: int, sparse_size: int = 8) -> torch.Tensor:
    x = x.permute(0, 2, 3, 1).reshape(-1, Hp // sparse_size, Hp // sparse_size, sparse_size, sparse_size, C)
    x = x.permute(0, 3, 4, 1, 2, 5).contiguous()
    B, bH, bW, win_H, win_W, _ = x.shape
    out = x.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(B, bH * win_H, bW * win_W, C)
    if bH * win_H > H:
        out = out[:, :H, :H, :].contiguous()
    return out.permute(0, 3, 1, 2)


class _SparseViTCBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = _SparseViTCMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _SparseViTSABlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, sparse_size: int = 8, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0, norm_layer: Any = nn.LayerNorm) -> None:
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = _SparseViTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _SparseViTMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        self.sparse_size = sparse_size
        self.gamma_1 = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed(x)
        x_before = x.flatten(2).transpose(1, 2)
        B, N, H, W = x.shape
        x_s, Ho, Hp, C = _sparsevit_alter_sparse(x, self.sparse_size)
        Bf, Nf, Hf, Wf = x_s.shape
        x_s = self.attn(self.norm1(x_s.flatten(2).transpose(1, 2))).transpose(1, 2).reshape(Bf, Nf, Hf, Wf)
        x_s = _sparsevit_alter_unsparse(x_s, Ho, Hp, C, self.sparse_size).flatten(2).transpose(1, 2)
        x_out = x_before + self.drop_path(self.gamma_1 * x_s)
        x_out = x_out + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_out), H, W))
        return x_out.transpose(1, 2).reshape(B, N, H, W)


class _SparseViTPatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 64) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        B, _, H, W = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()


class _SparseViTBackbone(nn.Module):
    def __init__(self, layers: List[int] = [5, 8, 20, 7], img_size: int = 512, embed_dim: List[int] = [64, 128, 320, 512], s_blocks3: List[int] = [8, 4, 2, 1], s_blocks4: List[int] = [2, 1], head_dim: int = 64) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed1 = _SparseViTPatchEmbed(img_size=img_size, patch_size=4, in_chans=3, embed_dim=embed_dim[0])
        self.patch_embed2 = _SparseViTPatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1])
        self.patch_embed3 = _SparseViTPatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2])
        self.patch_embed4 = _SparseViTPatchEmbed(img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3])
        self.pos_drop = nn.Dropout(p=0.0)
        dpr = [x.item() for x in torch.linspace(0, 0.2, sum(layers))]
        num_heads = [d // head_dim for d in embed_dim]

        self.blocks1 = nn.ModuleList([_SparseViTCBlock(dim=embed_dim[0], drop_path=dpr[i]) for i in range(layers[0])])
        self.norm1 = norm_layer(embed_dim[0])
        self.blocks2 = nn.ModuleList([_SparseViTCBlock(dim=embed_dim[1], drop_path=dpr[i + layers[0]]) for i in range(layers[1])])
        self.norm2 = norm_layer(embed_dim[1])
        self.blocks3 = nn.ModuleList([
            _SparseViTSABlock(dim=embed_dim[2], num_heads=num_heads[2], sparse_size=32 // s_blocks3[i // 5], drop_path=dpr[i + layers[0] + layers[1]], norm_layer=norm_layer)
            for i in range(layers[2])
        ])
        self.norm3 = norm_layer(embed_dim[2])
        self.blocks4 = nn.ModuleList([
            _SparseViTSABlock(dim=embed_dim[3], num_heads=num_heads[3], sparse_size=16 // s_blocks4[i // 4], drop_path=dpr[i + layers[0] + layers[1] + layers[2]], norm_layer=norm_layer)
            for i in range(layers[3])
        ])
        self.norm4 = norm_layer(embed_dim[3])
        self.pre_logits = nn.Identity()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        x = self.pos_drop(self.patch_embed1(x))
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for index, blk in enumerate(self.blocks3):
            x = blk(x)
            if (index + 1) % 5 == 0 and (index + 1) // 5 != 4:
                outputs["third" + str((index + 1) // 5)] = x
        outputs["third"] = self.norm3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        x = self.patch_embed4(x)
        for index, blk in enumerate(self.blocks4):
            x = blk(x)
            if (index + 1) % 4 == 0:
                outputs["last" + str((index + 1) // 4)] = x
        outputs["last"] = self.norm4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return outputs


class _SparseViTMultipleHead(nn.Module):
    def __init__(self, embed_dim: int = 512) -> None:
        super().__init__()
        self.gamma1 = nn.Parameter(1e-6 * torch.ones((embed_dim)))
        self.gamma2 = nn.Parameter(1e-6 * torch.ones((embed_dim)))
        self.gamma3 = nn.Parameter(1e-6 * torch.ones((embed_dim)))
        self.gamma4 = nn.Parameter(1e-6 * torch.ones((embed_dim)))
        self.gamma5 = nn.Parameter(1e-6 * torch.ones((embed_dim)))
        self.gamma6 = nn.Parameter(1e-6 * torch.ones((embed_dim)))
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.conv_layer1 = nn.Conv2d(320, 512, kernel_size=1)
        self.conv_layer2 = nn.Conv2d(320, 512, kernel_size=1)
        self.conv_layer3 = nn.Conv2d(320, 512, kernel_size=1)
        self.conv_layer4 = nn.Conv2d(320, 512, kernel_size=1)
        self.conv_last = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4, c5, c6 = features
        c1 = self.conv_layer1(c1)
        c2 = self.conv_layer2(c2)
        c3 = self.conv_layer3(c3)
        c4 = self.conv_layer4(c4)
        b, c, h, w = c1.shape
        c5 = F.interpolate(c5, size=(h, w), mode="bilinear", align_corners=False)
        c6 = F.interpolate(c6, size=(h, w), mode="bilinear", align_corners=False)
        x = (
            self.gamma1 * c1.flatten(2).transpose(1, 2)
            + self.gamma2 * c2.flatten(2).transpose(1, 2)
            + self.gamma3 * c3.flatten(2).transpose(1, 2)
            + self.gamma4 * c4.flatten(2).transpose(1, 2)
            + self.gamma5 * c5.flatten(2).transpose(1, 2)
            + self.gamma6 * c6.flatten(2).transpose(1, 2)
        )
        x = x.transpose(1, 2).reshape(b, c, h, w)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        return self.conv_last(x)


class OfficialSparseViTNet(nn.Module):
    """Official SparseViT_Mul architecture from scu-zjz/SparseViT."""

    def __init__(self, img_size: int = 512) -> None:
        super().__init__()
        self.img_size = img_size
        self.encoder_net = _SparseViTBackbone(img_size=img_size)
        self.lmu = _SparseViTMultipleHead(embed_dim=512)
        self.BCE_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_in = F.interpolate((x - mean) / std, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        feats_dict = self.encoder_net(x_in)
        logits = self.lmu(list(feats_dict.values()))
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


# =============================================================================
# 4. Official IML-ViT (SunnyHaze/IML-ViT)
# =============================================================================

def _imlvit_get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1), size=max_rel_dist, mode="linear").reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def _imlvit_add_decomposed_rel_pos(attn: torch.Tensor, q: torch.Tensor, rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor, q_size: Tuple[int, int], k_size: Tuple[int, int]) -> torch.Tensor:
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _imlvit_get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = _imlvit_get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    return (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)


class _IMLViTAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True, use_rel_pos: bool = False, input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos and input_size is not None:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = _imlvit_add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class _IMLViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True, drop_path: float = 0.0, norm_layer: Any = nn.LayerNorm, use_rel_pos: bool = False, window_size: int = 0, input_size: Optional[Tuple[int, int]] = None) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _IMLViTAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos, input_size=input_size if window_size == 0 else (window_size, window_size))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = TimmMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            B, H, W, C = x.shape
            pad_h = (self.window_size - H % self.window_size) % self.window_size
            pad_w = (self.window_size - W % self.window_size) % self.window_size
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w
            x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
            x = self.attn(x)
            x = x.view(B, Hp // self.window_size, Wp // self.window_size, self.window_size, self.window_size, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
            if Hp > H or Wp > W:
                x = x[:, :H, :W, :].contiguous()
        else:
            x = self.attn(x)
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class _IMLViTPatchEmbed(nn.Module):
    def __init__(self, kernel_size: Tuple[int, int] = (16, 16), stride: Tuple[int, int] = (16, 16), in_chans: int = 3, embed_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class _IMLViTLayerNorm2d(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        return self.weight[:, None, None] * ((x - u) / torch.sqrt(s + self.eps)) + self.bias[:, None, None]


class _IMLViTConv2d(nn.Conv2d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        norm = kwargs.pop("norm", None)
        super().__init__(*args, **kwargs)
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            x = self.norm(x)
        return x


class _IMLViTBackbone(nn.Module):
    def __init__(self, img_size: int = 1024, patch_size: int = 16, embed_dim: int = 768, depth: int = 12, num_heads: int = 12) -> None:
        super().__init__()
        self.patch_embed = _IMLViTPatchEmbed((patch_size, patch_size), (patch_size, patch_size), 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, (224 // patch_size) ** 2 + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, 0.1, depth)]
        win_idx = {0, 1, 3, 4, 6, 7, 9, 10}
        self.blocks = nn.ModuleList([
            _IMLViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=dpr[i],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_rel_pos=True,
                window_size=14 if i in win_idx else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.patch_embed(x)
        h, w = x.shape[1], x.shape[2]
        abs_pos = self.pos_embed[:, 1:]
        size = int(math.sqrt(abs_pos.shape[1]))
        if size != h or size != w:
            abs_pos = F.interpolate(abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2), size=(h, w), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
        else:
            abs_pos = abs_pos.reshape(1, h, w, -1)
        x = x + abs_pos
        for blk in self.blocks:
            x = blk(x)
        return {"last_feat": x.permute(0, 3, 1, 2)}


class _IMLViTSimpleFPN(nn.Module):
    def __init__(self, input_dim: int = 768, out_channels: int = 256, scale_factors: Tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)) -> None:
        super().__init__()
        self.stages = []
        strides = [int(16 // s) for s in scale_factors]
        for idx, scale in enumerate(scale_factors):
            out_dim = input_dim
            if scale == 4.0:
                layers: List[nn.Module] = [
                    nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=2, stride=2),
                    _IMLViTLayerNorm2d(input_dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(input_dim // 2, input_dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = input_dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(input_dim, input_dim // 2, kernel_size=2, stride=2)]
                out_dim = input_dim // 2
            elif scale == 1.0:
                layers = []
            else:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers.extend([
                _IMLViTConv2d(out_dim, out_channels, kernel_size=1, bias=False, norm=_IMLViTLayerNorm2d(out_channels)),
                _IMLViTConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False, norm=_IMLViTLayerNorm2d(out_channels)),
            ])
            seq = nn.Sequential(*layers)
            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", seq)
            self.stages.append(seq)

    def forward(self, x: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        feat = x["last_feat"]
        results = [stage(feat) for stage in self.stages]
        results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))
        return results


class _IMLViTPredictHead(nn.Module):
    def __init__(self, embed_dim: int = 256, norm: str = "BN") -> None:
        super().__init__()
        self.linear_fuse = nn.Conv2d(embed_dim * 5, embed_dim, kernel_size=1)
        self.norm = _IMLViTLayerNorm2d(embed_dim) if norm == "LN" else nn.BatchNorm2d(embed_dim)
        self.dropout = nn.Dropout()
        self.linear_predict = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        h, w = x[0].shape[2], x[0].shape[3]
        upsampled = [F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False) for t in x]
        return self.linear_predict(self.dropout(self.norm(self.linear_fuse(torch.cat(upsampled, dim=1)))))


class OfficialIMLViTNet(nn.Module):
    """Official iml_vit_model architecture from SunnyHaze/IML-ViT."""

    def __init__(self, input_size: int = 1024, predict_head_norm: str = "BN") -> None:
        super().__init__()
        self.input_size = input_size
        self.encoder_net = _IMLViTBackbone(img_size=input_size)
        self.featurePyramid_net = _IMLViTSimpleFPN()
        self.predict_head = _IMLViTPredictHead(embed_dim=256, norm=predict_head_norm)
        self.BCE_loss = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x_in = F.interpolate((x - mean) / std, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        feats = self.featurePyramid_net(self.encoder_net(x_in))
        logits = self.predict_head(feats)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


# =============================================================================
# 5. Official CoDE (siriusPRX/CoDE)
# =============================================================================

class _CoDEEfficientNetEncoder(nn.Module):
    def __init__(self, efn: nn.Module, start: int, end: int) -> None:
        super().__init__()
        self.blocks = efn._blocks[start:end]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class OfficialCoDENet(nn.Module):
    """Official AC_CoDE actor-critic reinforcement learning architecture from siriusPRX/CoDE."""

    def __init__(self, encoder: str = "efficientnet-b4", steps: int = 3) -> None:
        super().__init__()
        from efficientnet_pytorch import EfficientNet

        self.steps = steps
        efn = EfficientNet.from_name(encoder)
        if encoder == "efficientnet-b0":
            filters = [32, 24, 40, 80, 192]
            ends = [3, 5, 8, 15]
        else:
            filters = [48, 32, 56, 112, 272]
            ends = [6, 10, 16, 30]

        self.conv_stem = nn.Sequential(efn._conv_stem, nn.BatchNorm2d(filters[0]))
        self.down1 = _CoDEEfficientNetEncoder(efn, start=0, end=ends[0])
        self.down2 = _CoDEEfficientNetEncoder(efn, start=ends[0], end=ends[1])
        self.down3 = _CoDEEfficientNetEncoder(efn, start=ends[1], end=ends[2])
        self.down4 = _CoDEEfficientNetEncoder(efn, start=ends[2], end=ends[3])

        def _up(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(in_c, out_c, 3, 1, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

        def _merge(c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c * 2, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, 1, 1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            )

        self.p_up1, self.p_merge1 = _up(filters[4], filters[3]), _merge(filters[3])
        self.p_up2, self.p_merge2 = _up(filters[3], filters[2]), _merge(filters[2])
        self.p_up3, self.p_merge3 = _up(filters[2], filters[1]), _merge(filters[1])
        self.p_up4, self.p_merge4 = _up(filters[1], filters[0]), _merge(filters[0])

        self.p_mask_out = nn.Sequential(
            nn.Conv2d(1, filters[0], 1, bias=False), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], 3, 2, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
        )
        self.mu = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters[0], filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], 1, 3, 1, 1), nn.Softsign(),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters[0], filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], 1, 3, 1, 1), nn.Softplus(),
        )

        self.v_up1, self.v_merge1 = _up(filters[4], filters[3]), _merge(filters[3])
        self.v_up2, self.v_merge2 = _up(filters[3], filters[2]), _merge(filters[2])
        self.v_up3, self.v_merge3 = _up(filters[2], filters[1]), _merge(filters[1])
        self.v_up4, self.v_merge4 = _up(filters[1], filters[0]), _merge(filters[0])

        self.v_mask_out = nn.Sequential(
            nn.Conv2d(1, filters[0], 1, bias=False), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], filters[0], 3, 2, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
        )
        self.v_head = nn.Sequential(
            nn.Conv2d(filters[0] * 2, filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(filters[0], 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forgery_forward(self, x_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.conv_stem(x_rgb)
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        p1 = self.p_merge1(torch.cat((self.p_up1(d4), d3), dim=1))
        p2 = self.p_merge2(torch.cat((self.p_up2(p1), d2), dim=1))
        p3 = self.p_merge3(torch.cat((self.p_up3(p2), d1), dim=1))
        p4 = self.p_merge4(torch.cat((self.p_up4(p3), x1), dim=1))

        v1 = self.v_merge1(torch.cat((self.v_up1(d4), d3), dim=1))
        v2 = self.v_merge2(torch.cat((self.v_up2(v1), d2), dim=1))
        v3 = self.v_merge3(torch.cat((self.v_up3(v2), d1), dim=1))
        v4 = self.v_merge4(torch.cat((self.v_up4(v3), x1), dim=1))
        return p4, v4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_in = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
        p4, v4 = self.forgery_forward(x_in)
        mask = torch.zeros((b, 1, 512, 512), dtype=x.dtype, device=x.device)
        for _ in range(self.steps):
            p_m = self.p_mask_out(mask)
            mu = self.mu(torch.cat((p4, p_m), dim=1))
            mask = torch.clamp(mask + 0.5 * mu, 0.0, 1.0)
        return F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)


# =============================================================================
# 6. Official MaskCLIP / SideCLIP (weryw/MaskCLIP CVPR 2025)
# =============================================================================

class _CLIPResBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.ModuleDict({
            "c_fc": nn.Linear(d_model, d_model * 4),
            "c_proj": nn.Linear(d_model * 4, d_model),
        })
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ln = self.ln_1(x)
        attn_out, _ = self.attn(x_ln, x_ln, x_ln, need_weights=False)
        x = x + attn_out
        m = self.mlp["c_fc"](self.ln_2(x))
        m = m * torch.sigmoid(1.702 * m)
        x = x + self.mlp["c_proj"](m)
        return x


class _CLIPTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.resblocks = nn.ModuleList([_CLIPResBlock(width, heads) for _ in range(layers)])


class _CLIPVisual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.zeros(1024))
        self.positional_embedding = nn.Parameter(torch.zeros(257, 1024))
        self.proj = nn.Parameter(torch.zeros(1024, 768))
        self.conv1 = nn.Conv2d(3, 1024, kernel_size=14, stride=14, bias=False)
        self.ln_pre = nn.LayerNorm(1024)
        self.transformer = _CLIPTransformer(1024, 24, 16)
        self.ln_post = nn.LayerNorm(1024)


class _CLIPBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.zeros(77, 768))
        self.text_projection = nn.Parameter(torch.zeros(768, 768))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.visual = _CLIPVisual()
        self.transformer = _CLIPTransformer(768, 12, 12)
        self.token_embedding = nn.Embedding(49408, 768)
        self.ln_final = nn.LayerNorm(768)


class _SideViTAttn(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _SideViTBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SideViTAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.ModuleDict({
            "fc1": nn.Linear(dim, dim * 4),
            "fc2": nn.Linear(dim * 4, dim),
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp["fc2"](F.gelu(self.mlp["fc1"](self.norm2(x))))
        return x


class _SideViTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, 768))
        self.patch_embed = nn.ModuleDict({"proj": nn.Conv2d(3, 768, kernel_size=16, stride=16)})
        self.blocks = nn.ModuleList([_SideViTBlock(768, 12) for _ in range(12)])


class _ChannelLayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _FusionInputProj(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_proj = nn.ModuleList([
            nn.LayerNorm(1024),
            nn.Conv2d(1024, 768, kernel_size=1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, 1024) where N=256 (16x16)
        B, N, C = x.shape
        h = w = int(N ** 0.5)
        x = self.input_proj[0](x).permute(0, 2, 1).reshape(B, C, h, w)
        return self.input_proj[1](x)


class _MaskDecoderMLP(nn.Module):
    def __init__(self, is_conv: bool = False) -> None:
        super().__init__()
        if is_conv:
            self.layers = nn.ModuleList([
                nn.Conv2d(768, 256, 1),
                nn.Conv2d(256, 256, 1),
                nn.Conv2d(256, 256, 1),
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Linear(768, 256),
                nn.Linear(256, 256),
                nn.Linear(256, 256),
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class _SideMaskDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query_mlp = _MaskDecoderMLP(is_conv=False)
        self.pix_mlp = _MaskDecoderMLP(is_conv=True)
        self.refiner = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            _ChannelLayerNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            _ChannelLayerNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        )


class _SideAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit_model = _SideViTModel()
        self.text_projection = nn.Linear(768, 768)
        self.fusion_layers = nn.ModuleDict({
            "layer_0": _FusionInputProj(),
            "layer_3": _FusionInputProj(),
            "layer_6": _FusionInputProj(),
            "layer_9": _FusionInputProj(),
        })
        self.mask_decoder = _SideMaskDecoder()


class _ClsAggregatorLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(1024, 16)
        self.layer_norm = nn.LayerNorm(1024)


class _ClsAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ClsAggregatorLayer()])
        self.projector = nn.Sequential(nn.Linear(1024, 768, bias=False))


class _PromptLearner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ctx_pos = nn.Parameter(torch.zeros(1, 10, 768))
        self.ctx_neg = nn.Parameter(torch.zeros(1, 10, 768))


class OfficialMaskCLIPNet(nn.Module):
    """Official MaskCLIP / SideCLIP architecture from weryw/MaskCLIP (CVPR 2025)."""

    def __init__(self) -> None:
        super().__init__()
        self.clip = _CLIPBackbone()
        self.aggregator = _SideAggregator()
        self.cls_aggregator = _ClsAggregator()
        self.prompt_learner = _PromptLearner()

    def _encode_prompts(self, batch_size: int) -> torch.Tensor:
        # Construct positive and negative learned prompt sequences (length 77)
        device = self.prompt_learner.ctx_pos.device
        dtype = self.prompt_learner.ctx_pos.dtype
        sos = self.clip.token_embedding(torch.tensor([[49406]], device=device))  # (1, 1, 768)
        eos = self.clip.token_embedding(torch.zeros((1, 66), dtype=torch.long, device=device))  # (1, 66, 768)
        pos_seq = torch.cat([sos, self.prompt_learner.ctx_pos, eos], dim=1) + self.clip.positional_embedding.unsqueeze(0)
        neg_seq = torch.cat([sos, self.prompt_learner.ctx_neg, eos], dim=1) + self.clip.positional_embedding.unsqueeze(0)
        seqs = torch.cat([neg_seq, pos_seq], dim=0).permute(1, 0, 2)  # (77, 2, 768)
        for blk in self.clip.transformer.resblocks:
            seqs = blk(seqs)
        seqs = self.clip.ln_final(seqs.permute(1, 0, 2))  # (2, 77, 768)
        txt_feat = seqs[:, 11, :] @ self.clip.text_projection  # (2, 768)
        txt_feat = F.normalize(txt_feat, dim=-1)
        return txt_feat.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # 1. CLIP visual stream (224x224 -> 16x16 patches)
        x_clip = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        v = self.clip.visual.conv1(x_clip).flatten(2).transpose(1, 2)  # (B, 256, 1024)
        cls_tok = self.clip.visual.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        v = torch.cat([cls_tok, v], dim=1) + self.clip.visual.positional_embedding.unsqueeze(0)
        v = self.clip.visual.ln_pre(v).permute(1, 0, 2)  # (257, B, 1024)

        clip_feats: Dict[int, torch.Tensor] = {}
        tap_map = {5: 0, 11: 3, 17: 6, 23: 9}
        for i, blk in enumerate(self.clip.visual.transformer.resblocks):
            v = blk(v)
            if i in tap_map:
                clip_feats[tap_map[i]] = v[1:, :, :].permute(1, 0, 2).contiguous()

        # 2. Side ViT stream (512x512 -> 32x32 patches for memory efficiency)
        side_size = 512
        x_side = F.interpolate(x, size=(side_size, side_size), mode="bilinear", align_corners=False)
        s = self.aggregator.vit_model.patch_embed["proj"](x_side)  # (B, 768, 32, 32)
        sh, sw = s.shape[2], s.shape[3]
        pos = self.aggregator.vit_model.pos_embed.reshape(1, 64, 64, 768).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(sh, sw), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
        s = s.flatten(2).transpose(1, 2) + pos

        for i, blk in enumerate(self.aggregator.vit_model.blocks):
            if f"layer_{i}" in self.aggregator.fusion_layers and i in clip_feats:
                fused = self.aggregator.fusion_layers[f"layer_{i}"](clip_feats[i])
                fused = F.interpolate(fused, size=(sh, sw), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
                s = s + fused
            s = blk(s)

        s_2d = s.transpose(1, 2).reshape(B, 768, sh, sw)
        pix_emb = self.aggregator.mask_decoder.refiner(self.aggregator.mask_decoder.pix_mlp(s_2d))

        txt_feat = self._encode_prompts(B)
        queries = self.aggregator.mask_decoder.query_mlp(self.aggregator.text_projection(txt_feat))  # (B, 2, 256)
        logits = torch.einsum("bqc,bchw->bqhw", queries, pix_emb)
        probs = torch.softmax(logits, dim=1)[:, 1:2, :, :]
        return F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)

