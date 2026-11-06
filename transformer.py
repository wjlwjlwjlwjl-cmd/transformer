import math

import torch
import torch.nn as nn

# 得到词嵌入矩阵 E
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x): # x 为 token 索引张量，shape 为 [batch_size, seq_len]
        # 原论文中，这里乘了 根号d_model，目的是增强语义信息相对位置编码对输入嵌入矩阵的影响
        # 但是后来发现后续步骤可以抵消这部分影响，所以很多时候忽略这一步
        # 这里输出的是 [batch_size, seq_len, d_model]，句子数、每个句子token数、向量维度数
        return self.embedding(x) * math.sqrt(self.d_model)

# 得到词嵌入矩阵E和位置编码矩阵结合后的输入嵌入矩阵 X
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        # 在 __init__ 中，先算出来位置编码矩阵，后面找输入嵌入矩阵时直接切片使用即可
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, 1).unsqueeze(1) # 加上一维，基于广播机制，使得能够和 div_term 运算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 每一行中的元素，基数取 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 每一行中的元素，基数取 cos

        pe.unsqueeze_(0) # 为了能够和词嵌入矩阵E进行加法运算
        self.register_buffer('pe', pe) # 这里的位置编码矩阵是不需要被训练的

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :] # batch_size、d_model 继承，但是在 __init__ 时构造的 pe 中的 seq_len 可能比每个句子的实际切分token要大
        return self.dropout(x) # 将我们最后得到的输入嵌入矩阵随机失活，防止过拟合

# Layer Norm 层归一化
class LayerNorm(nn.Module):
    def __init__(self, d_model:int, eps: float=10**-6):
        super().__init__()
        self.eps = eps # 极小数字，用来防止方差为零导致分母为0的情况
        self.alpha = nn.Parameter(torch.ones(d_model)) # LN 训练参数之一
        self.bias = nn.Parameter(torch.zeros(d_model)) # LN 训练参数之一

    def forward(self, x):
        aval = torch.mean(x, dim=-1, keepdim=True) # 求平均值矩阵，keepdim=True 表示保留维度，dim=-1表示取最后一个维度，即 dim
        std = torch.std(x, dim=-1, keepdim=True) # 求标准差矩阵
        return self.alpha * (x - aval) / (std + self.eps) + self.bias # 计算层归一化后的矩阵

# Feed-forward Neural Network，FNN，前馈神经网络
class FeedforwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float=0.1): # d_ff 为中间层维度，升维后有利于捕捉到更多的非线性特征
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # 升维操作，权重和偏置由 pytorch 自己保存并作为训练参数（默认偏置也训练）
        self.linear_2 = nn.Linear(d_ff, d_model) # 降维操作
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 对于线性变换，我们做的任何变换最后都还是线性的，但是 relu 函数（其实就是 max）可以实现“弯折”直线
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x)))) 

# MultiheadAttention，多头注意力机制，将一个模型维度分成若干份，每一份自己去进行 softmax，每一份都可能形成自己关注的方向，而整体的工作量近乎不变
class MultiheadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float=0.1):
        super().__init__()
        assert d_model % h == 0, "模型维度应当被头数量整除"

        self.d_model = d_model

        self.dropout = nn.Dropout(dropout)
        self.h = h
        self.d_h = d_model // h # 没个头处理的维度

        # 要训练的四个权重矩阵，按照论文中的方式，偏重不需要背训练，后续归一化反而显得训练偏重多余
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
    # 用来进行注意力计算
    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout, d_h, mask):
        # [batch_size, seq_ret, h, d_h] -> [batch_size, h, seq_ret, d_h]
        attention_scores = query @ (key.transpose(-2, -1)) / math.sqrt(d_h) # 将最后两个维度交换就等价于转置了
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9) # 防止模型预先看到训练的答案
        attention = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention = dropout(attention)
        return attention @ value, attention

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor):
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        # 计算出整个模型维度的qkv
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # 从逻辑上，我们更喜欢 batch_size 个句子，分为 h 个头，总的 token 数量为 seq_ret，每个 token 的维度为 d_h
        q = q.view(q.shape[0], q.shape[1], self.h, self.d_h).transpose(1, 2)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_h).transpose(1, 2)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_h).transpose(1, 2)

        x, self.attention_w = MultiheadAttentionBlock.attention(q, k, v, self.dropout, self.d_h, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_h * self.h)
        return self.w_o(x)

# 残差连接，这里将残差链接和 FNN 或者 多头注意力结合在一起
class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)

    def forward(self, x, sublayer):
        # 这里采取的是预先归一化的方式，相比于使用论文最初的后置归一化，具有以下优点
        # 梯度流动更高，训练效果更明显
        # 不过会导致方差不断积累（因为每次的结果都是直接加了进来），因此最后返回结果前建议 finalnorm 进行归一化
        return x + self.dropout(sublayer(self.norm(x))) 

# 编码器
class EncoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block: MultiheadAttentionBlock, feed_forward_block: FeedforwardNetwork, dropout: float = 0.1):
        super().__init__()
        self.self_attention_block = self_attention_block # 多头注意力
        self.feed_forward_block = feed_forward_block # 前馈神经网络
        self.dropout = nn.Dropout(dropout) # 随机失活

        self.residual_connections = nn.ModuleList([ResidualConnection(d_model) for i in range(2)]) # Encoder 需要的两次残差链接
    
    def forward(self, x, src_mask: torch.Tensor):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # 进行多头注意力训练，并残差链接
        x = self.residual_connections[1](x, lambda x: self.feed_forward_block(x)) # 进行前馈神经网络，并残差链接
        return x

class Encoder(nn.Module):
    def __init__(self, d_model: int, layers: nn.ModuleList): # layers 就是我们要跑 n 层的 Encoder
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)

    def forward(self, x, src_mask: torch.Tensor):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x) # 前面我们使用的是预归一化，好处是更好的梯度流动，但是代价是方差会不断积累，因此最后需要加上归一化约束

# 译码器
class DecoderBlock(nn.Module):
    def __init__(self, d_model, self_attention_block: MultiheadAttentionBlock,
                cross_attention_block: MultiheadAttentionBlock, feed_forward: FeedforwardNetwork):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward = feed_forward

        self.residual_connections = nn.ModuleList([ResidualConnection(d_model) for i in range(3)]) # 自注意力、交叉注意力、前馈神经网络需要的三层残差链接

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward(x))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
# 输出投影层：将输出映射到词表
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.project = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.project(x)  # 只输出 logits，训练交给 CrossEntropyLoss，推理用 argmax 等价

# 封装出 整个 Transformer 模块
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, src_pos: PositionalEncoding, 
                 tgt_embed: InputEmbeddings, tgt_pos: PositionalEncoding, proj: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.proj_layer = proj

    def encode(self, src, src_mask):
        src = src.long()
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, src_mask, tgt_mask, encoder_output):
        tgt = tgt.long()
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.proj_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, 
                      N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048):
    """_summary_ 构建 Transformer 类

    Args:
        src_vocab_size (int): _description_ 源词库大小
        tgt_vocab_size (int): _description_ 目标词库大小
        src_seq_len (int): _description_ 源词库中最大token数量
        tgt_seq_len (int): _description_ 目标词库中最大token数量
        d_model (int, optional): _description_. Defaults to 512. 嵌入维度
        N (int, optional): _description_. Defaults to 6. Decoder、Encoder 堆叠层数
        h (int, optional): _description_. Defaults to 8. 多头注意力头数
        dropout (float, optional): _description_. Defaults to 0.6. 随机失活概率
        d_ff (int, optional): _description_. Defaults to 2048. FNN 升维维度，默认是嵌入维度四倍
    """
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    src_pos = PositionalEncoding(src_seq_len, d_model, dropout)

    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    tgt_pos = PositionalEncoding(tgt_seq_len, d_model, dropout)

    encoder_blocks = []
    for _ in range(N):
        multihead_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedforwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, multihead_attention_block, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        self_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        cross_attention_block = MultiheadAttentionBlock(d_model, h, dropout)
        feed_forward = FeedforwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, self_attention_block, cross_attention_block, feed_forward)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    proj_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, src_pos, tgt_embed, tgt_pos, proj_layer)

    for p in transformer.parameters():
        if p.dim() > 1: # 权重矩阵进行初始化，偏置不训练
            nn.init.kaiming_uniform_(p, nonlinearity="relu")

    return transformer