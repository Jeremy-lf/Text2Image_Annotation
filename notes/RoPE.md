
要理解**1D-RoPE**和**2D-RoPE**的实现，我们需要从**核心逻辑**（旋转矩阵+相对位置编码）出发，用PyTorch代码逐步实现，并结合具体场景测试。


## 一、RoPE的核心逻辑回顾
RoPE（旋转位置编码）的本质是**用旋转矩阵将位置信息注入`query`/`key`向量**，让Attention计算直接感知**相对位置**（而非绝对位置）。其核心公式为：  
对于位置`m`，旋转矩阵`R(m)`由正弦/余弦函数构造，满足：  
$$ R(m)^T R(n) = R(n-m) $$  
因此，`query`（位置`m`）与`key`（位置`n`）的点积会包含**相对位置`n-m`**的信息，模型由此感知“token A在token B左边3位”这类相对关系。


## 二、1D-RoPE实现（处理序列位置）
1D-RoPE用于**1D序列**（如文本、展平后的图像patch序列），每个token只有一个位置坐标`m`（序列索引）。


### 1. 代码实现
```python
import torch
import torch.nn as nn
import math

class RoPE1D(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        """
        dim: 模型隐藏维度（需为偶数）
        max_seq_len: 最大序列长度（预计算缓存用）
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # 1. 计算频率向量：θ_i = 10000^(-2i/dim)，i∈[0, dim//2-1]
        self.freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # shape: (dim//2,)
        
        # 2. 预计算所有位置的cos/sin缓存（避免重复计算）
        self.register_buffer("cos_cached", torch.zeros(max_seq_len, dim//2))
        self.register_buffer("sin_cached", torch.zeros(max_seq_len, dim//2))
        for pos in range(max_seq_len):
            self.cos_cached[pos] = torch.cos(pos * self.freq)
            self.sin_cached[pos] = torch.sin(pos * self.freq)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: 输入张量，shape: (batch_size, seq_len, dim) 或 (seq_len, dim)
        positions: 每个token的位置索引，shape: (seq_len,) 或 (batch_size, seq_len)
        返回：旋转后的张量（shape与x一致）
        """
        positions = positions.long()  # 确保位置是整数
        # 获取当前位置的cos/sin（从缓存中取）
        cos = self.cos_cached[positions]  # shape: (seq_len, dim//2) 或 (batch, seq_len, dim//2)
        sin = self.sin_cached[positions]  # 同上
        
        # 将x分成前半和后半（各dim//2维）
        x1 = x[..., :self.dim//2]  # 前半部分
        x2 = x[..., self.dim//2:]  # 后半部分
        
        # 旋转操作：x_rot = [x1*cos - x2*sin, x1*sin + x2*cos]
        x_rot = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        return x_rot
```


### 2. 测试1D-RoPE
```python
# 初始化1D-RoPE（隐藏维度64，最大序列长度10）
rope1d = RoPE1D(dim=64, max_seq_len=10)

# 输入：batch=1，序列长度5，隐藏维度64
x = torch.randn(1, 5, 64)  
positions = torch.arange(5)  # 位置索引：0,1,2,3,4

# 旋转后的输出
x_rot = rope1d(x, positions)
print("1D-RoPE输入shape:", x.shape)    # torch.Size([1,5,64])
print("1D-RoPE输出shape:", x_rot.shape)# torch.Size([1,5,64])
```


## 三、2D-RoPE实现（处理空间位置）
2D-RoPE用于**2D空间结构**（如图像），每个token有两个位置坐标`(h, w)`（行、列索引）。核心是**同时对行和列方向应用RoPE**，让模型感知空间相对位置（如“patch A在patch B的右上方2行3列”）。


### 1. 代码实现
```python
class RoPE2D(nn.Module):
    def __init__(self, dim: int, max_h: int = 256, max_w: int = 256):
        """
        dim: 模型隐藏维度（需为偶数）
        max_h: 最大行数（图像高度上限）
        max_w: 最大列数（图像宽度上限）
        """
        super().__init__()
        self.dim = dim
        self.max_h = max_h
        self.max_w = max_w
        
        # 将隐藏维度分成两部分：行方向（dim_h）和列方向（dim_w）
        self.dim_h = dim // 2
        self.dim_w = dim - self.dim_h  # 确保总和为dim
        
        # 2. 行方向的频率向量（处理高度h）
        self.freq_h = 1.0 / (10000 ** (torch.arange(0, self.dim_h, 2).float() / self.dim_h))
        # 3. 列方向的频率向量（处理宽度w）
        self.freq_w = 1.0 / (10000 ** (torch.arange(0, self.dim_w, 2).float() / self.dim_w))
        
        # 4. 预计算行方向的cos/sin缓存（shape: (max_h, dim_h//2)）
        self.register_buffer("cos_h_cached", torch.zeros(max_h, self.dim_h//2))
        self.register_buffer("sin_h_cached", torch.zeros(max_h, self.dim_h//2))
        for h in range(max_h):
            self.cos_h_cached[h] = torch.cos(h * self.freq_h)
            self.sin_h_cached[h] = torch.sin(h * self.freq_h)
        
        # 5. 预计算列方向的cos/sin缓存（shape: (max_w, dim_w//2)）
        self.register_buffer("cos_w_cached", torch.zeros(max_w, self.dim_w//2))
        self.register_buffer("sin_w_cached", torch.zeros(max_w, self.dim_w//2))
        for w in range(max_w):
            self.cos_w_cached[w] = torch.cos(w * self.freq_w)
            self.sin_w_cached[w] = torch.sin(w * self.freq_w)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        x: 输入张量，shape: (batch_size, seq_len, dim) 或 (seq_len, dim)
        positions: 2D位置索引，shape: (seq_len, 2) 或 (batch_size, seq_len, 2)，最后一维是(h, w)
        返回：旋转后的张量（shape与x一致）
        """
        positions = positions.long()
        h = positions[..., 0]  # 行索引（高度）
        w = positions[..., 1]  # 列索引（宽度）
        
        # 获取行方向的cos/sin（shape: (seq_len, dim_h//2) 或 (batch, seq_len, dim_h//2)）
        cos_h = self.cos_h_cached[h]
        sin_h = self.sin_h_cached[h]
        # 获取列方向的cos/sin（shape: (seq_len, dim_w//2) 或 (batch, seq_len, dim_w//2)）
        cos_w = self.cos_w_cached[w]
        sin_w = self.sin_w_cached[w]
        
        # 将x分成行部分（前dim_h维）和列部分（后dim_w维）
        x_h = x[..., :self.dim_h]  # 行方向特征
        x_w = x[..., self.dim_h:]  # 列方向特征
        
        # 6. 对行部分应用RoPE
        x_h1 = x_h[..., :self.dim_h//2]  # 行部分前半
        x_h2 = x_h[..., self.dim_h//2:]  # 行部分后半
        x_h_rot = torch.cat([x_h1*cos_h - x_h2*sin_h, x_h1*sin_h + x_h2*cos_h], dim=-1)
        
        # 7. 对列部分应用RoPE
        x_w1 = x_w[..., :self.dim_w//2]  # 列部分前半
        x_w2 = x_w[..., self.dim_w//2:]  # 列部分后半
        x_w_rot = torch.cat([x_w1*cos_w - x_w2*sin_w, x_w1*sin_w + x_w2*cos_w], dim=-1)
        
        # 8. 合并行和列的旋转结果
        x_rot = torch.cat([x_h_rot, x_w_rot], dim=-1)
        return x_rot
```


### 2. 测试2D-RoPE
```python
# 初始化2D-RoPE（隐藏维度64，最大高度16，最大宽度16）
rope2d = RoPE2D(dim=64, max_h=16, max_w=16)

# 输入：batch=1，序列长度4（2x2图像展平为4个patch），隐藏维度64
x = torch.randn(1, 4, 64)  
# 2D位置：(行h, 列w)，对应2x2图像的4个patch
positions = torch.tensor([[0,0], [0,1], [1,0], [1,1]])  # shape: (4,2)

# 旋转后的输出
x_rot = rope2d(x, positions)
print("2D-RoPE输入shape:", x.shape)    # torch.Size([1,4,64])
print("2D-RoPE输出shape:", x_rot.shape)# torch.Size([1,4,64])
```


## 四、关键差异与应用场景
| **维度** | **1D-RoPE**                | **2D-RoPE**                |
|----------|-----------------------------|-----------------------------|
| 位置类型 | 1D序列索引（如文本位置）    | 2D空间坐标（如图像h,w）     |
| 应用场景 | 文本、语音等1D序列          | 图像、视频等2D/3D空间数据   |
| 旋转逻辑 | 仅处理一个方向的位置        | 同时处理行（h）和列（w）方向 |


## 五、在Attention中的实际应用
RoPE通常**仅应用于`query`和`key`**（`value`不需要位置编码，因为它是内容特征）。以下是带RoPE的Attention计算示例：

```python
def attention_with_rope(q, k, v, positions, rope):
    """
    q: query，shape: (batch, seq_len, dim)
    k: key，shape: (batch, seq_len, dim)
    v: value，shape: (batch, seq_len, dim)
    positions: 位置索引，shape: (seq_len,) 或 (batch, seq_len, 2)（2D时）
    rope: RoPE实例（1D或2D）
    """
    # 对query和key应用RoPE
    q_rot = rope(q, positions)
    k_rot = rope(k, positions)
    
    # 计算Attention分数（带相对位置）
    d_k = q.size(-1)
    scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(d_k)
    attn_weights = torch.softmax(scores, dim=-1)
    
    # 加权求和value
    output = torch.matmul(attn_weights, v)
    return output
```


## 六、总结
- **1D-RoPE**：处理1D序列的位置编码，通过旋转矩阵将序列索引注入`query/key`，让Attention感知相对顺序。  
- **2D-RoPE**：扩展到2D空间，同时处理行和列的位置，让模型理解图像的空间结构（如物体的位置、形状、相对关系）。  
- **核心优势**：相对位置编码泛化能力强（训练短序列，测试长序列无需外推）、无额外参数、计算高效。

通过上述代码，你可以清晰看到RoPE如何将位置信息“注入”模型，解决1D/2D数据的位置丢失问题。在实际多模态模型（如Qwen2-VL、GPT-4V）中，2D-RoPE是理解图像空间结构的关键组件。
