"""
@FileName：utils.py
@Description：
@Author：zhangyt\n
@Time：2025/2/10 20:51
"""
import torch
import torch.nn.functional as F


def circular_padding_chw(batch, padding_width, padding_height):
    """
    对输入的批次图像进行循环填充。

    参数:
    - batch: 输入的批次图像，形状为 [batch_size, channels, height, width]
    - padding_width: 宽度方向的填充大小
    - padding_height: 高度方向的填充大小

    返回:
    - padded_batch: 填充后的批次图像
    """
    # 使用 F.pad 进行循环填充
    padded_batch = F.pad(batch, (padding_width, padding_width, padding_height, padding_height), mode='circular')
    return padded_batch
