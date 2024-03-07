from einops import rearrange

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from .resnet import InflatedConv3d

def inflated_interpolate_3d(x, size=None, scale_factor=None, mode='nearest'):
    video_length = x.shape[2]
    x = rearrange(x, "b c f h w -> (b f) c h w")
    x = F.interpolate(x, size=size, scale_factor=scale_factor, mode=mode)
    x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)
    return x

def get_mask_decoder(
    module_type,
    in_channels,
):
    if module_type == "MaskDecoder":
        return MaskDecoder(
            in_channels=in_channels,
        )
    else:
        return None

class MaskDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        reduced_channel = 512,
    ):
        super().__init__()

        self.in_channels = in_channels

        self.projects = nn.ModuleList()
        for i in range(len(in_channels)):
            if i == 0:
                channel = in_channels[i]
            else:
                channel = reduced_channel + in_channels[i]
            self.projects.append(MultiScaleProj(channel, reduced_channel))

        # last projection map
        self.final_conv = InflatedConv3d(reduced_channel, 1, 1)
    
    def forward(self, features):
        x = features[0]
        for i in range(len(self.in_channels)):
            x = self.projects[i](x)
            if i < len(self.in_channels) - 2:
                x = inflated_interpolate_3d(x, scale_factor=2, mode='nearest')
            if i < len(self.in_channels) - 1:
                x = torch.cat([x, features[i+1]], dim=1)

        x = self.final_conv(x)
        x = F.sigmoid(x)
        return x

class MultiScaleProj(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv1 = InflatedConv3d(C_in, C_out, 3, 1, 1, bias=False)
        self.conv2 = InflatedConv3d(C_out, C_out, 3, 1, 1, bias=False)
        self.conv_temp = nn.Conv1d(C_out, C_out, 3, 1, 1, bias=False)
        self.norm1 = nn.GroupNorm(32, C_out)
        self.norm2 = nn.GroupNorm(32, C_out)
        self.norm_temp = nn.GroupNorm(32, C_out)

    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        # temporal convolution
        h, w = x.shape[-2:]
        x = rearrange(x, "b c f h w -> (b h w) c f")
        x = self.conv_temp(x)
        x = self.norm_temp(x)
        x = F.relu(x)
        x = rearrange(x, "(b h w) c f -> b c f h w", h=h, w=w)
        
        return x