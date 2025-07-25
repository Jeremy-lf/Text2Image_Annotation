## Transformer Mask

机器学习领域中，掩码（Mask）本质是一个跟需要掩盖的目标张量大小一致的（大多数是0-1二值）张量，其思想最早起源于 word2vec 的CBOW的训练机制：通过上下文来预测中心词。掩码就相当于把中心词给遮掩住。不同的任务和应用场景可能需要不同类型的mask操作。在自注意力模型中，常见的mask操作有两种：Padding mask和Sequence mask。

- Padding mask（填充掩码）：在处理变长序列时，为了保持序列的长度一致，通常会在序列的末尾添加一些特殊的填充符号（如）。Padding mask的作用是将这些填充符号对应位置的注意力分数设为一个很小的值（如负无穷），从而使模型在计算注意力分数时忽略这些填充符号，避免填充符号对计算产生干扰。
- Sequence mask（序列掩码）：在某些任务中，为了避免模型在生成序列时看到未来的信息，需要对注意力分数进行掩码操作。Sequence mask的作用是通过构建下三角（或者上三角）的注意力分数矩阵，将当前位置之后位置的注意力分数设为一个很小的值，从而使模型只关注当前 token 与之前 token 的注意力关系，不理会它与后续 token 的关系。这样可以保证模型在生成序列时只依赖于已经生成的部分，不会受到未来信息的影响，即只”看”当前及前面的 tokens。也有把Sequence mask叫做Casual Mask的。

总结，Padding Mask的作用是避免填充符号带来的偏差。Sequence mask的作用是屏蔽未来信息，防止偷看，保证每个位置只能看到前面的tokens。

![image](https://github.com/user-attachments/assets/022528ed-cbeb-4519-b640-71b9f9fae20f)
src_mask和tgt_mask。Encoder只会看src_mask。Decoder会看src_mask和tgt_task。src_mask就是Padding Mask，而tgt_mask是包含了padding mask和sequence mask的融合mask。

对于解码器，实际操作会将两种掩码合并，每个位置取最小值，相当于两个掩码只要有任意一种情况需要被遮蔽，则就应该被遮蔽。具体可以参见下图。
<img width="1796" height="385" alt="image" src="https://github.com/user-attachments/assets/8c647af5-33fe-40b2-a426-c3322eec2ebb" />


#### 代码实现
```python
class Batch:
    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src # 源语言句子列表，形状是[batch_size,Length]
        # 创建源语言的掩码，这样可以忽略填充部分，unsqueeze()的作用是增加一维度，因为后续要和注意力分数进行掩码计算，而注意力分数是三个维度，所以这里要保持一致。
        # (src != pad)返回一个等大的布尔张量，src元素等于pad的位置为False,否则为True
        # unsqueeze(1)作用是增加了一个维度，变成pad_attn_mask: [batch_size,1,seq_len]
        # 最终得到返回一个[batch_size, 1, seq_len]大小的布尔张量，False是需要mask掉的位置
        self.src_mask = (src != pad).unsqueeze(-2) 


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    
    # 先计算注意力分数
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 在query和key的转置相乘得出（len_q,len_k）这个注意力分数矩阵以后，使用mask来掩盖相乘结果矩阵，此处把创建掩码矩阵和应用掩码矩阵合二为一
    if mask is not None:
        # 如果发现mask是0，就用-1e9来替换它
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 然后才开始实施softmax操作    
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

作者：罗西的思考
链接：https://juejin.cn/post/7478967140377198626

```


```
## 题目
假设我对一段自然语言分chunk，每个chunk是由连续的字符串组成且长度不等，现在我给他们依次打上标签，如
"[0,0,1,1,1,1,2,2,2,3,4,4,4,5,6,6,6,6,6,6,7,7]",
每个单词都有一个标签，同样的数字代表这些单词属于同一个chunk。现在让你实现一个attention mask，
要求每个单词可以关注同chunk和之前chunk内的所有单词。优先尝试用tensor计算完成。
 
## 例子
```python
Input: List = [0,0,1,1,1,1,2,2,2,3,4,4]
 
Output: 
tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
```



```
# 实现
import torch
labels = [0,0,1,1,1,1,2,2,2,3,4,4]
labels = torch.tensor(labels)
n = len(labels)
 
attn_mask = labels[:,None] >= labels[None,:]
print(attn_mask.int())
```
