## 典型应用场景
* 多查询注意力（MQA）：nums_head=N, nums_key_value_head=1（单键值头）
* 分组查询注意力（GQA）：nums_head=N, nums_key_value_head=M（M为分组数，N是M的倍数）
* 标准多头注意力：nums_head = nums_key_value_head（此时广播操作相当于复制1次，无实际变化）

```python
class SelfAttention(nn.Module):
    def __init__(self, weights_dim, n_heads):
        super().__init__()
        self.qkv_matrices = nn.Linear(weights_dim, 3 * weights_dim)
        self.linear_layer = nn.Linear(weights_dim, weights_dim)

    def forward(self, x, mask=None):
        q, k, v = self.qkv_matrices(x).chunk(3, dim=-1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1) @ v
        return self.linear_layer(attention)

```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int = 8, d_model: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.n_dims = d_model // n_heads
        self.scale = d_model ** 0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        q,k,v = self.qkv(x).chunk(3, dim=-1) # BxNxD
        q = q.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_dims).permute(0, 2, 1, 3)
        k = k.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_dims).permute(0, 2, 1, 3)
        v = v.reshape(x.shape[0], x.shape[1], self.n_heads, self.n_dims).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1,-2))/self.scale
        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
        return self.proj(out)
```


```python
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head, nums_key_value_head):
        super().__init__()
        assert self.hidden_dim % nums_head == 0
        assert nums_head % nums_key_value_head == 0

        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head

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
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # 关注: nums_head 和 nums_key_value_head 的关系
        q = q.transpose(1, 2) # (b, nums_head, seq, head_dim)
        k = k.transpose(1, 2) # (b, nums_key_value_head, seq, head_dim)
        v = v.transpose(1, 2)  # (b, nums_key_value_head, seq, head_dim)

        # k v repeat； （广播操作）
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim=1)

        attention_score = (q @ k.transpose(2, 3)) / math.sqrt(self.head_dim)
        attention_weight = torch.softmax(attention_score, dim=-1)
        output = attention_weight @ v  # (b, nums_head, seq, head_dim)
        output = output.transpose(1, 2).contiguous()
        final_output = self.o_proj(output.view(batch_size, seq, -1))

        return final_output

# 测试
x = torch.rand(3, 2, 128)
net = GroupQueryAttention(128, 8, 4)
net(x).shape
```
