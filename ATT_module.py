import math
from typing import List, Optional
from torch import nn
import torch
from torch.nn import functional as F

class ATTLayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(ATTLayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel * 3, bias=False),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x) # (1500,32,1,1)
        y = y.view(b, c) # (1500,32)
        y = self.fc(y) # (1500,32)
        y = y.view(b, c*3, 1, 1) # (1500,32,1,1)
        y1, y2, y3 = torch.split(y, [self.channel, self.channel, self.channel], dim=1)
        return y1,y2,y3





class RelationAwareAttentionLayer(nn.Module):
    """
    专为SelectE模型设计的关系感知注意力机制
    """
    
    def __init__(self, num_branches=3, feature_channels=20, relation_dim=300, 
                  num_heads=8, dropout=0.1):
        super(RelationAwareAttentionLayer, self).__init__()
        
        self.num_branches = num_branches  # 分支数量 (filter1, filter3, filter5)
        self.feature_channels = feature_channels  # 特征通道数
        self.relation_dim = relation_dim  # 关系嵌入维度
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        
        assert feature_channels % num_heads == 0, "feature_channels必须能被num_heads整除"
        
        # 关系到查询的映射
        self.relation_to_query = nn.Sequential(
            nn.Linear(relation_dim, feature_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels, feature_channels),
            nn.Dropout(dropout)
        )
        
        
        # 多头注意力的Key和Value投影
        self.key_proj = nn.Linear(feature_channels, feature_channels)
        self.value_proj = nn.Linear(feature_channels, feature_channels)
        self.output_proj = nn.Linear(feature_channels, feature_channels)
        
        
        
        # 门控机制
        # self.gate = nn.Sequential(
        #     nn.Linear(relation_dim + feature_channels, feature_channels),
        #     nn.Sigmoid()
        # )
        
        # Dropout和归一化
        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(feature_channels)
        
    # def spatial_pooling(self, x):
    #     """空间池化获取全局特征"""
    #     # x: [B, C, H, W]
    #     avg_pool = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]
    #     max_pool = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]
    #     return (avg_pool + max_pool) / 2
    
    def relation_guided_attention(self, features, relation_emb):
        """
        关系引导的注意力计算
        
        Args:
            features: 特征图 [B, C, H, W]
            relation_emb: 关系嵌入 [B, relation_dim]
            
        Returns:
            attended_features: 注意力增强的特征 [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # 生成关系查询
        relation_query = self.relation_to_query(relation_emb)  # [B, C]
        
        # 将特征图转为序列格式进行注意力计算
        features_seq = features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 多头注意力
        # Query来自关系信息
        q = relation_query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        
        # Key和Value来自特征图
        k = self.key_proj(features_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        v = self.value_proj(features_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, 1, H*W]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, C)
        attn_output = self.output_proj(attn_output).squeeze(1)  # [B, C]
    
        # 门控机制
        # global_feature = self.spatial_pooling(features)  # [B, C]
        # gate_input = torch.cat([relation_emb, global_feature], dim=1)
        # gate_weight = self.gate(gate_input)  # [B, C]
        
        # 应用门控
        enhanced_feature = attn_output  # [B, C]
        
        # 广播回特征图形状
        enhanced_feature = enhanced_feature.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        enhanced_features =  features + enhanced_feature.expand_as(features)
        
        return enhanced_features
    
    def forward(self, x1, x3, x5, relation_emb):
       
        
        # 对每个分支应用关系引导注意力
        x1_enhanced = self.relation_guided_attention(x1, relation_emb)
        x3_enhanced = self.relation_guided_attention(x3, relation_emb)
        x5_enhanced = self.relation_guided_attention(x5, relation_emb)
       
        
        # 综合注意力权重
        y1 = x1_enhanced + x1  # [B, C, H, W] 或 [B, 1, H, W]
        y3 = x3_enhanced + x3
        y5 = x5_enhanced + x5
        
        return y1, y3, y5



class RelationAwareAttentionLayer1(nn.Module):
    """
    专为SelectE模型设计的关系感知注意力机制
    """
    
    def __init__(self,  feature_channels=20, relation_dim=300, num_heads=8, dropout=0.1):
        super(RelationAwareAttentionLayer1, self).__init__()
        
        
        self.feature_channels = feature_channels  # 特征通道数
        self.relation_dim = relation_dim  # 关系嵌入维度
        self.num_heads = num_heads
        self.head_dim = feature_channels // num_heads
        
        assert feature_channels % num_heads == 0, "feature_channels必须能被num_heads整除"
        
        # 关系到查询的映射
        self.relation_to_query = nn.Sequential(
            nn.Linear(relation_dim, feature_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels, feature_channels),
            nn.Dropout(dropout)
        )
        
        
        # 多头注意力的Key和Value投影
        self.key_proj = nn.Linear(feature_channels, feature_channels)
        self.value_proj = nn.Linear(feature_channels, feature_channels)
        self.output_proj = nn.Linear(feature_channels, feature_channels)
        
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(relation_dim + feature_channels, feature_channels),
            nn.Sigmoid()
        )
        
        # Dropout和归一化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_channels)
        
    def spatial_pooling(self, x):
        """空间池化获取全局特征"""
        # x: [B, C, H, W]
        avg_pool = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]
        max_pool = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)  # [B, C]
        return (avg_pool + max_pool) / 2
    
    def relation_guided_attention(self, features, relation_emb):
        """
        关系引导的注意力计算
        
        Args:
            features: 特征图 [B, C, H, W]
            relation_emb: 关系嵌入 [B, relation_dim]
            
        Returns:
            attended_features: 注意力增强的特征 [B, C, H, W]
        """
        B, C, H, W = features.shape
        
        # 生成关系查询
        relation_query = self.relation_to_query(relation_emb)  # [B, C]
        
        # 将特征图转为序列格式进行注意力计算
        features_seq = features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 多头注意力
        # Query来自关系信息
        q = relation_query.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, 1, head_dim]
        
        # Key和Value来自特征图
        k = self.key_proj(features_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        v = self.value_proj(features_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, H*W, head_dim]
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, num_heads, 1, H*W]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, 1, C)
        attn_output = self.output_proj(attn_output).squeeze(1)  # [B, C]
        attn_output = self.layer_norm(attn_output)
        # 门控机制
        global_feature = self.spatial_pooling(features)  # [B, C]
        gate_input = torch.cat([relation_emb, global_feature], dim=1)
        gate_weight = self.gate(gate_input)  # [B, C]
        
        # 应用门控
        enhanced_feature = attn_output  # [B, C]
        
        # 广播回特征图形状
        enhanced_feature = enhanced_feature.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        enhanced_features = features + enhanced_feature.expand_as(features)
        
        return enhanced_features
    
    def forward(self, x, relation_emb):
       
        
        # 对每个分支应用关系引导注意力
        x1_enhanced = self.relation_guided_attention(x, relation_emb)
       
       
        
        # 综合注意力权重
        y1 = x1_enhanced  # [B, C, H, W] 或 [B, 1, H, W]
        
        return y1






import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class AdaptiveKernelSelection1(nn.Module):
    """
    自适应核选择模块 (V3 - 带残差连接).

    该模块根据输入的特征图和可选的关系嵌入，为来自不同卷积分支的特征图动态生成注意力权重。
    它返回一个加权后的特征图列表。通过残差连接，原始特征可以与加权后的特征相加，
    使得模块能够学习对原始特征进行调制，而不是完全替换它们。

    流程:
    1.  融合、压缩、激励、选择: 与V2版本相同，计算出各分支的注意力权重。
    2.  加权: 将原始特征图与对应的权重相乘。
    3.  残差连接 (可选): 将加权后的特征图与原始输入特征图逐元素相加。
    """
    def __init__(self, num_branches: int, feature_channels: int, relation_dim: Optional[int] = None, 
                 reduction: int = 8, dropout: float = 0.1, use_residual: bool = True):
        """
        初始化函数.

        Args:
            num_branches (int): 卷积分支的数量。
            feature_channels (int): 每个分支输出特征图的通道数。
            relation_dim (Optional[int]): 关系嵌入的维度。如果为 None，则不使用关系信息。
            reduction (int): 第一个全连接层的降维因子。
            dropout (float): Dropout比率。
            use_residual (bool): 是否启用残差连接。默认为 True。
        """
        super(AdaptiveKernelSelection1, self).__init__()
        self.num_branches = num_branches
        self.feature_channels = feature_channels
        self.relation_dim = relation_dim
        self.use_residual = use_residual

        bottleneck_dim = max(feature_channels // reduction, 4)

        excitation_input_dim = feature_channels
        if relation_dim is not None:
            excitation_input_dim += relation_dim

        self.excitation_net = nn.Sequential(
            nn.Linear(excitation_input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, feature_channels * num_branches)
        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: List[torch.Tensor], rel_embedded: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if len(features) != self.num_branches:
            raise ValueError(f"输入特征图的数量 ({len(features)}) 与初始化的分支数 ({self.num_branches}) 不匹配。")
        if self.relation_dim is not None and rel_embedded is None:
            raise ValueError("模块初始化时需要关系嵌入，但前向传播时未提供。")

        batch_size = features[0].size(0)

        # 步骤 1: 融合与压缩
        fused_features = torch.stack(features, dim=0).sum(dim=0)
        squeezed_vector = self.global_avg_pool(fused_features).view(batch_size, self.feature_channels)

        # 步骤 2: 激励
        if self.relation_dim is not None and rel_embedded is not None:
            context_vector = torch.cat([squeezed_vector, rel_embedded], dim=1)
        else:
            context_vector = squeezed_vector
        
        z = self.excitation_net(context_vector)
        
        # 步骤 3: 选择 (生成权重)
        attention_scores = z.reshape(batch_size, self.num_branches, self.feature_channels)
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)

        # 步骤 4: 加权
        stacked_features = torch.stack(features, dim=1)
        weighted_features_stacked = stacked_features * attention_weights
        weighted_features_list = list(torch.unbind(weighted_features_stacked, dim=1))

        # 步骤 5: 残差连接 (可选)
        if self.use_residual:
            output_features = []
            for i in range(self.num_branches):
                # 将原始输入特征与加权后的特征相加
                output_features.append(features[i] + weighted_features_list[i])
        else:
            output_features = weighted_features_list

        return output_features





# 自动选择核产生的机制
class AdaptiveKernelSelection(torch.nn.Module):
    def __init__(self, num_kernels, feature_dim, relation_dim=None, hidden_dim=64,
                 temperature=1.0, use_relation_aware=True):
        """
        自适应核选择模块
        Args:
            num_kernels: 卷积核的数量
            feature_dim: 特征维度
            relation_dim: 关系嵌入维度（可选）
            hidden_dim: 隐藏层维度
            temperature: softmax温度参数
            use_relation_aware: 是否使用关系感知机制
        """
        super(AdaptiveKernelSelection, self).__init__()
        self.num_kernels = num_kernels
        self.temperature = temperature
        self.use_relation_aware = use_relation_aware

        # 全局平均池化和最大池化
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = torch.nn.AdaptiveMaxPool2d(1)

        # 计算输入维度
        input_dim = feature_dim * 2  # avg + max pooling
        if use_relation_aware and relation_dim is not None:
            input_dim += relation_dim

        # 特征融合网络
        self.feature_fusion = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU()
        )

        # 核权重生成网络
        self.weight_generator = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, num_kernels),
            torch.nn.Softmax(dim=1)
        )

        # 可选的门控机制
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim // 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, feature_maps, relation_embedding=None):
        """
        Args:
            feature_maps: list of tensor, 不同卷积核产生的特征图
            relation_embedding: tensor, 关系嵌入（可选）
        Returns:
            weighted_features: 加权融合后的特征图
            attention_weights: 注意力权重
        """
        batch_size = feature_maps[0].size(0)

        # 1. 对每个特征图进行全局池化
        pooled_features = []
        for feat_map in feature_maps:
            avg_pooled = self.global_avg_pool(feat_map).view(batch_size, -1)
            max_pooled = self.global_max_pool(feat_map).view(batch_size, -1)
            combined_pooled = torch.cat([avg_pooled, max_pooled], dim=1)
            pooled_features.append(combined_pooled)

        # 2. 计算特征统计信息
        stacked_features = torch.stack(pooled_features, dim=1)  # (B, num_kernels, 2*C)
        mean_features = torch.mean(stacked_features, dim=1)  # (B, 2*C)

        # 3. 构建输入特征
        input_features = [mean_features]
        if self.use_relation_aware and relation_embedding is not None:
            input_features.append(relation_embedding)

        combined_input = torch.cat(input_features, dim=1)

        # 4. 生成权重
        fused_features = self.feature_fusion(combined_input)
        attention_weights = self.weight_generator(fused_features / self.temperature)

        # 5. 可选的门控机制
        gate_value = self.gate(fused_features)
        uniform_weights = torch.ones_like(attention_weights) / self.num_kernels
        final_weights = gate_value * attention_weights + (1 - gate_value) * uniform_weights

        # 6. 应用权重并融合特征
        weighted_features = []
        for i, feat_map in enumerate(feature_maps):
            weight = final_weights[:, i:i + 1, None, None]
            weighted_feat = feat_map * weight
            weighted_features.append(weighted_feat)

        return weighted_features, final_weights