import argparse
import ast
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import functional as F

from SE_module import SELayer
from ATT_module import RelationAwareAttentionLayer1
from ATT_module import RelationAwareAttentionLayer
from ATT_module import AdaptiveKernelSelection1

class RelationSpecificConv(torch.nn.Module):
    def __init__(self, num_emb, in_channel, output_channel, filter_size, reshape_H, reshape_W, init_fn):
        super(RelationSpecificConv, self).__init__()
        self.num_emb = num_emb
        self.in_channel = in_channel
        self.output_channel = output_channel
        self.h = filter_size[0]
        self.w = filter_size[1]
        self.dilate_height_rate = 1
        self.dilate_width_rate = 1
        if len(filter_size) == 3:
            self.dilate_height_rate = filter_size[2]
            self.dilate_width_rate = filter_size[2]
        if len(filter_size) == 4:
            self.dilate_height_rate = filter_size[2]
            self.dilate_width_rate = filter_size[3]
        filter_dim = self.in_channel * self.output_channel * self.h * self.w  # 1*8*1*5=40
        self.filter = torch.nn.Embedding(num_emb, filter_dim, padding_idx=0)
        self.reshape_H, self.reshape_W = reshape_H, reshape_W
        self.init_fn = init_fn
        self.bn = torch.nn.BatchNorm2d(self.output_channel)
        self.se = SELayer(self.output_channel, reduction=int(0.5 * output_channel))

    def init_weights(self):
        self.init_fn(self.filter.weight)

    def forward(self, e1_embedded, x, rel):
        f1 = self.filter(rel)  # (1500,160)
        f1 = f1.reshape(e1_embedded.size(0) * self.in_channel * self.output_channel, 1, self.h, self.w)  # (48000,4,1,5)
        if  self.dilate_height_rate==1 and  self.dilate_height_rate==1:
            x = F.conv2d(x, f1, groups=e1_embedded.size(0),
                         padding=(int((self.h - 1) // 2), int((self.w - 1) // 2)))  # (4,48000,20,20)
        else:
            x = F.conv2d(x, f1, groups=e1_embedded.size(0),
                         padding=(int((self.h - 1) * self.dilate_height_rate // 2),
                                  int((self.w - 1) * self.dilate_width_rate // 2)),
                         dilation=(self.dilate_height_rate, self.dilate_width_rate))  # (4,48000,20,20)
        x = x.reshape(e1_embedded.size(0), self.output_channel, self.reshape_H, self.reshape_W)  # (128,128,20,20)
        x = self.bn(x)
        x = self.se(x)
        return x


class SelectE(torch.nn.Module):
    def __init__(self, logger, num_emb=100, embedding_dim=300, input_drop=0.4, hidden_drop=0.3, feature_map_drop=0.3,
                 k_w=10, k_h=20, output_channel=20,
                 filter_size_list=[(1, 5), (3, 3), (1, 9)],
                 active_fn='relu', init_fn='xavier_normal'):
        super(SelectE, self).__init__()

        current_file_name = os.path.basename(__file__)
        logger.info("[Model Name]: " + str(current_file_name))

        # 定义模型
        self.emb = torch.nn.Embedding(num_emb, embedding_dim)
        self.logger = logger
        self.embedding_dim = embedding_dim
        self.perm = 1
        #
        self.k_w = k_w
        self.k_h = k_h

        self.loss = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda')
        self.active_fn = self.get_active_fn(active_fn)
        self.init_fn = self.get_init_fn(init_fn)
        # 定义尺寸
        self.chequer_perm = self.get_chequer_perm()
        self.reshape_H = 20
        self.reshape_W = 20
        self.in_channel = 1  # 输入通道数
        self.num_filters = len(filter_size_list)
        self.filter_size_list = filter_size_list
        if isinstance(output_channel, int):
            self.output_channels_list = [output_channel] * self.num_filters
        else:
            self.output_channels_list = output_channel

        if len(self.output_channels_list) != self.num_filters:
            raise ValueError("output_channels 长度必须与 filter_sizes 匹配")
            # 创建多个RelationSpecificConv实例
        self.conv_layers = torch.nn.ModuleList()
        for i, (out_ch, filter_size) in enumerate(zip(self.output_channels_list, filter_size_list)):
            conv = RelationSpecificConv(
                num_emb=num_emb,
                in_channel=self.in_channel,
                output_channel=out_ch,
                filter_size=filter_size,
                reshape_H=self.reshape_H,
                reshape_W=self.reshape_W,
                init_fn=self.init_fn
            )
            self.conv_layers.append(conv)
        total_channel = sum(self.output_channels_list)
        # 定义dropout和batchnorm
        self.input_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feature_map_drop)
        # self, num_branches=3, feature_channels=20, relation_dim=300, 
                #  reduction=10, num_heads=8, dropout=0.1
        self.auto_select = AdaptiveKernelSelection1(num_branches=self.num_filters,feature_channels=output_channel,relation_dim=None)
        self.att = RelationAwareAttentionLayer(num_branches=3,feature_channels=output_channel,
                                               relation_dim=embedding_dim,num_heads=2, dropout=0.2)
        self.bn0 = torch.nn.BatchNorm2d(self.in_channel)
        self.bn1 = torch.nn.BatchNorm2d(total_channel)

        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)

        fc_length = self.reshape_H * self.reshape_W * total_channel
        self.fc = torch.nn.Linear(fc_length, embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_emb)))

    def to_var(self, x, use_gpu=True):
        if use_gpu:
            return Variable(torch.from_numpy(x).long().cuda())

    def get_active_fn(self, active_fn_name):
        if active_fn_name == 'relu':
            return F.relu
        elif active_fn_name == 'leaky_relu':
            return F.leaky_relu
        elif active_fn_name == 'tanh':
            return F.tanh
        elif active_fn_name == 'sigmoid':
            return F.sigmoid
        elif active_fn_name == 'silu':
            return F.silu
        elif active_fn_name == 'softplus':
            return F.softplus
        elif active_fn_name == 'gelu':
            return F.gelu
        elif active_fn_name == 'elu':
            return F.elu
        elif active_fn_name == 'selu':
            return F.selu
        else:
            raise ValueError("Unsupported activation function: {}".format(active_fn_name))

    def get_init_fn(self, init_fn_name):
        if init_fn_name == 'xavier_normal':
            return torch.nn.init.xavier_normal_
        elif init_fn_name == 'xavier_uniform':
            return torch.nn.init.xavier_uniform_
        elif init_fn_name == 'kaiming_normal':
            return torch.nn.init.kaiming_normal_
        elif init_fn_name == 'kaiming_uniform':
            return torch.nn.init.kaiming_uniform_
        return torch.nn.init.xavier_normal_

    def init(self):
        init_fn = self.init_fn
        init_fn(self.emb.weight.data)
        for conv_layer in self.conv_layers:
            conv_layer.init_weights()

    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])  # 返回一个随机排列
        rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        comb_idx = []
        for k in range(self.perm):
            temp = []
            ent_idx, rel_idx = 0, 0

            for i in range(self.k_h):
                for j in range(self.k_w):
                    if k % 2 == 0:
                        if i % 2 == 0:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                        else:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                    else:
                        if i % 2 == 0:
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                        else:
                            temp.append(ent_perm[k, ent_idx])
                            ent_idx += 1
                            temp.append(rel_perm[k, rel_idx] + self.embedding_dim)
                            rel_idx += 1

            comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm


    def forward(self, e1, rel):
        e1 = self.to_var(e1)
        rel = self.to_var(rel)
        e1_embedded = self.emb(e1)
        rel_embedded = self.emb(rel)
        comb_emb = torch.cat([e1_embedded, rel_embedded], dim=1)
        chequer_perm = comb_emb[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.perm, 2 * self.k_w, self.k_h))
        x = self.bn0(stack_inp)
        x = self.input_drop(x)
        x = x.permute(1, 0, 2, 3)

        outputs = []
        for conv in self.conv_layers:
            output = conv(e1_embedded, x, rel)
            outputs.append(output)
        outputs=self.auto_select(outputs)
        
        x1,x2,x3 = self.att(outputs[0],outputs[1],outputs[2],rel_embedded)

        x = torch.cat([x1,x2,x3], dim=1)
        # x = self.att(x,rel_embedded)
        x = self.active_fn(x)
        x = self.feature_map_drop(x)

        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = self.active_fn(x)
        weight = self.emb.weight
        weight = weight.transpose(1, 0)
        x = torch.mm(x, weight)
        x += self.b.expand_as(x)
        pred = x
        return pred


import torch
import numpy as np
import logging


def test_selecte_forward():
    """SelectE模型前向传递简单测试函数"""

    # 设置日志
    logger = logging.getLogger('SelectE_Test')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    print("=" * 50)
    print("SelectE模型前向传递测试")
    print("=" * 50)

    try:
        # 创建模型
        model = SelectE(
            logger=logger,
            num_emb=1000,  # 实体/关系数量
            embedding_dim=200,  # 嵌入维度
            input_drop=0.2,
            hidden_drop=0.3,
            feature_map_drop=0.2,
            k_w=10,
            k_h=20,
            output_channel=32,
            filter_size_list=[(1, 5, 1, 2), (3, 3), (1, 9)],
            active_fn='relu',
            init_fn='xavier_normal'
        )

        # 移动到GPU（如果可用）
        device = torch.device('cuda' if torch.cuda.is_available() else 'cuda')
        model = model.to(device)
        print(f"✓ 模型创建成功，使用设备: {device}")

        # 初始化权重
        model.init()
        print("✓ 模型权重初始化完成")

        # 生成测试数据

        batch_size = 16
        e1 = torch.randint(0, 1000, (batch_size,), dtype=torch.long, device=device)

        rel = torch.randint(0, 1000, (batch_size,), dtype=torch.long, device=device)
        e2 = torch.randint(0, 1000, (batch_size,), dtype=torch.long, device=device)  # 目标实体

        print(f"✓ 测试数据生成完成 - 批次大小: {batch_size}")
        print(f"  e1形状: {e1.shape}, rel形状: {rel.shape}")
        print(f"e1: {e1.shape}")
        # 前向传递测试
        model.eval()
        with torch.no_grad():
            pred = model(e1, rel)

        print(f"✓ 前向传递成功!")
        print(f"  输出形状: {pred.shape}")
        print(f"  期望形状: ({batch_size}, 1000)")
        print(f"  输出范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
        print(f"  输出均值: {pred.mean().item():.4f}")

        # 检查输出有效性
        assert pred.shape == (batch_size, 1000), f"输出形状错误: {pred.shape}"
        assert not torch.isnan(pred).any(), "输出包含NaN值"
        assert not torch.isinf(pred).any(), "输出包含Inf值"
        print("✓ 输出数值检查通过")

        # 测试梯度计算
        model.train()
        pred = model(e1, rel)
        e2_tensor = torch.LongTensor(e2).to(device)
        loss = model.loss(pred, e2_tensor)
        loss.backward()

        print(f"✓ 梯度计算成功 - 损失值: {loss.item():.4f}")

        # 检查主要参数的梯度
        grad_check_params = ['emb.weight', 'fc.weight']
        for name, param in model.named_parameters():
            if any(key in name for key in grad_check_params):
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  {name}: 梯度范数 = {grad_norm:.6f}")
                else:
                    print(f"  {name}: 无梯度")

        print("\n" + "=" * 50)
        print("✓ 所有测试通过! SelectE模型工作正常")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_selecte_forward()
    parser = ArgumentParser("YYModel", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument("--data_path", default="./data/", help="Data sources.")
    parser.add_argument("--run_folder", default="./", help="Data sources.")
    parser.add_argument("--data_name", default="FB15k-237", help="Name of the dataset.")
    parser.add_argument("--embedding_dim", default=300, type=int, help="Entity/Relation dimension")
    parser.add_argument("--min_lr", default=5e-5, type=float, help='L2 regularization')
    parser.add_argument("--batch_size", default=1000, type=int, help='Batch Size')

    # 前两个位置是卷积核的高和宽，后两个位置是膨胀率
    parser.add_argument("--filter_size_list",default=[(1, 5, 1, 2), (3, 3), (1, 9)],help='filter size')
    args = parser.parse_args()
    print(ast.literal_eval(args.data_path))


def parse_list_argument(value):
    try:
        # 使用ast.literal_eval安全地解析字符串为Python数据结构
        return ast.literal_eval(value)
    except (ValueError, SyntaxError) as e:
        raise argparse.ArgumentTypeError(f"Invalid list format: {value}") from e
