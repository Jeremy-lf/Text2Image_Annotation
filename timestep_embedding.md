### 时间步嵌入（Timestep Embedding）
在扩散模型中，时间步 t 通常是一个离散或连续的标量（如 0 到 1000）。嵌入将其转换为向量，使模型能理解时间信息。与Transformer的位置编码不同，这里的时间步可能是非整数（如 t=3.5），因此嵌入需要支持连续值。

1.频率衰减设计
使用指数衰减的频率（max_period^(-i/half)）确保嵌入能捕捉从全局到局部的时间特征：
- 低频（大周期）捕捉长期趋势。
- 高频（小周期）捕捉细微变化。

2.正弦/余弦组合
通过 cos 和 sin 的组合，嵌入可以表示任意相位偏移，类似傅里叶变换的基函数。这种设计允许模型通过线性组合学习不同时间尺度的模式。


### 代码解读
```python
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        将输入的时间步 t（可以是标量或分数）映射到一个高维向量空间（维度为 dim），生成一个与位置无关的连续嵌入表示。
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        # math.log==自然对数（以 e 为底的对数），即 ln(10)。
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )  # 生成half个频率值，范围从 max_period^(-0) 到 max_period^(-1)，呈指数衰减。
        args = t[:, None].float() * freqs[None] # [N, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1) # (N,2*half)
        if dim % 2:
            # 如果 dim 是奇数，补一个零列
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            # 返回形状为 (N, dim) 的嵌入张量，每个时间步 t_i 被映射为一个 dim 维向量。
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

```

### 具体示例
1.假设输入 t = [0, 1, 2]，dim=4，max_period=1000：

2.计算 freqs（half=2）：
freqs = [1000^(-0/2), 1000^(-1/2)] = [1.0, 0.0316]。

3.计算 args：
args = [[0*1.0, 0*0.0316], [1*1.0, 1*0.0316], [2*1.0, 2*0.0316]]。

4.生成嵌入：
cos(args) 和 sin(args) 分别计算后拼接，得到 3x4 的矩阵。
