class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # rmsnorm = x/sqrt(mean(x**2, dim=-1, keepdim=True) + eps) * weight
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms
        return normed * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        # x-mean / sqrt(var + eps) * weight + bias
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        # (x-mean) / sqrt(var + eps) * weight + bias
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        return normed * self.weight + self.bias


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.qkv = nn.Linear(dim, 3*dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, mask=None):
        q,k,v = self.qkv(x).chunk(3,dim=-1)
        attn = (q @ k.transpose(-2, -1)) / self.scale
        if mask:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return self.proj(out)


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_dims = dim // n_heads
        self.scale = self.n_dims ** 0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        q,k,v = self.qkv(x).chunk(3, dim=-1) # BXNxD
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_dims).permute(0, 2, 1, 3)
        k = k.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_dims).permute(0, 2, 1, 3)
        v = v.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_dims).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) / self.scale # BXNxN
        attn = attn.softmax(dim=-1) # BXNxN
        out = attn @ v # B x n_heads x N x n_dims
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.proj(out)


class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert self.hidden_dim % nums_head == 0
        assert nums_head % nums_key_value_head == 0

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(hidden_dim, nums_head*self.head_dim)
        self.k_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)
        self.v_proj = nn.Linear(hidden_dim, nums_key_value_head * self.head_dim)

        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, X, attention_mask=None):
        batch_size, seq, _ = X.size()

        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

        # attention_weight 目标shape 是 (batch, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim).transpose(1, 2) # (b, nums_head, seq, head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim).transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim).transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        # q = q.transpose(1, 2) # (b, nums_head, seq, head_dim)
        # k = k.transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        # v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)
        
        # k v repeat； （广播操作）
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / self.scale
        attention_weight = torch.softmax(attention_score, dim=-1)
        output = attention_weight @ v  # (b, nums_head, seq, head_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output




# 测试自定义softmax函数和torch.nn.functional.softmax函数的输出是否相同
def custom_softmax(x):
    # 数值稳定性：减去最大值避免指数运算溢出
    exp_x = torch.exp(x - x.max(dim=1, keepdim=True)[0])  # 数值稳定性优化
    return exp_x / exp_x.sum(dim=1, keepdim=True)

custom_prob = custom_softmax(logits)
print(torch.allclose(prob, custom_prob))  # 输出True


from torch.nn import functional as F
class LoRALinear(nn.Module):
    def __init__(self, in_feature, out_feature, rank=8, alpha=1.0):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.rank = rank
        self.alpha = alpha
        self.linear = nn.Linear(in_feature, out_feature)
        # 冻结原始权重
        self.linear.weight.requires_grad_(False)

        # self.weight = nn.Parameter(torch.randn(out_feature, in_feature))
        # 定义A和B矩阵,分别是rank x in_feature和out_feature x rank的参数矩阵
        self.A = nn.Parameter(torch.randn(in_feature, rank))
        self.B = nn.Parameter(torch.randn(rank, out_feasture))

        # 初始化A和B矩阵,使得它们的乘积在训练初期接近于0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(1/rank))
        nn.init.zeros_(self.B)
        self.scaling = alpha / rank
    
    def forward(self, x):
        base = self.linear(x)
        low_rank_update = x @ self.A @ self.B * self.scaling
        return base + low_rank_update



def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim, max_len=2048):
        super().__init__()
        # 1/10000^(2i/dim)的频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # m*freq
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, seq_len):
        cos = self.cos[:seq_len].unsqueeze(0).to(x.device) # [seq_len, 1, dim]
        sin = self.sin[:seq_len].unsqueeze(0).to(x.device)
        return (x * cos) + (rotate_half(x) * sin) # [seq_len, b, dim]


class PositionalEncoding(nn.Module):
    """
    正弦 / 余弦绝对位置编码,接口风格跟 PyTorch Transformer 一致：
    输入输出形状都是 [seq_len, batch_size, d_model]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        ) 

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(0)
        x = x + self.pe[:seq_len] 
        return self.dropout(x)
