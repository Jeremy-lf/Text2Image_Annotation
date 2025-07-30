
## Patchify
在DiT（Diffusion Transformer）架构中，Patchify操作不涉及下采样，而是通过图像分块和线性嵌入将空间图像数据转换为Transformer可处理的序列化标记（token），同时保留原始空间信息。

### **1. Patchify的核心作用**
- **图像分块**：将输入图像（如尺寸为 $\(256 \times 256 \times 3\)$ 的RGB图像）划分为多个不重叠的小块（patches）。例如，若分块大小 $\(p=4\)$，则图像被划分为 $\((256/4)^2 = 4096\)$ 个 $\(4 \times 4 \times 3\)$ 的小块。
- **线性嵌入**：对每个小块进行线性变换（全连接层），将其映射为一个固定维度的向量（标记）。例如，若嵌入维度 $\(d=1152\)$，则每个小块转换为 $\(1 \times 1152\)$ 的向量。
- **序列化**：将所有标记拼接成一个序列（长度为 $\(T = (H \times W) / p^2\)$，维度为 $\(d\)）$，作为Transformer的输入。

### **2. 与下采样的区别**
- **下采样**：通过池化或卷积操作减少空间分辨率（如 $\(256 \times 256 \to 128 \times 128\)$），同时可能增加通道数，但会丢失部分空间细节。
- **Patchify**：
  - **不改变分辨率**：分块后的小块仍保留原始像素信息，仅通过线性嵌入转换维度。
  - **保留全局信息**：所有标记共同构成序列，Transformer通过自注意力机制捕捉全局依赖关系。
  - **计算效率**：标记数量 $\(T\)$ 由分块大小 $\(p\)$ 决定$(\(p\)$ 越小，$\(T\)$ 越大），直接影响计算量（Gflops）。例如，$\(p=2\)$ 时 $\(T\)$ 是 $\(p=4\)$ 的4倍。

### **3. DiT中的关键实现细节**
- **位置嵌入**：对Patchify生成的标记序列应用正弦-余弦位置编码，使模型感知标记的空间位置。
- **条件输入融合**：
  - **时间步嵌入**：将扩散模型的时间步 $\(t\)$ 编码为向量，与标记序列拼接或通过自适应层归一化（adaLN）融入模型。
  - **类别/文本嵌入**：将类别标签或文本条件编码为向量，与时间步嵌入结合后输入模型。
- **块设计**：DiT采用多种Transformer块变体（如adaLN-Zero），通过条件信息动态调整层归一化参数，提升生成质量。

### **4. 示例流程**
以输入图像 $\(x \in \mathbb{R}^{256 \times 256 \times 3}\)$ 为例：
1. **Patchify**：
   - 分块大小 $\(p=4\)$，生成 $\(4096\)$ 个 $\(4 \times 4 \times 3\)$ 的小块。
   - 线性嵌入为 $\(4096\)$ 个 $\(1152\)$ 维标记，序列长度 $\(T=4096\)$。
2. **位置嵌入**：为每个标记添加位置编码，保留空间顺序。
3. **条件融合**：将时间步 $\(t\)$ 和类别标签 $\(y\)$ 编码为向量，与标记序列结合。
4. **Transformer处理**：通过多层自注意力机制和前馈网络生成输出标记。
5. **解码**：将输出标记映射回图像空间，生成去噪后的图像。

### **5. 为什么不用下采样？**
- **信息保留**：下采样会丢失高频细节，而Patchify通过分块保留原始像素信息，适合高分辨率生成任务。
- **扩展性**：Transformer的并行计算能力使其能高效处理长序列（如大量标记），避免下采样后的分辨率限制。
- **灵活性**：分块大小 $\(p\)$ 可调整，平衡计算量与生成质量（如 $\(p=2\)$ 适合更高分辨率，但计算量更大）。

### Python版本
```python
import numpy as np

def patchify(image, patch_size, step=1):
    """
    将图像分割成小块
    
    参数:
        image: 输入图像 (2D 或 3D numpy 数组)
        patch_size: 小块大小 (h, w) 或 (h, w, c)
        step: 滑动步长 (默认为1)
    
    返回:
        patches: 4D numpy 数组 (num_patches_h, num_patches_w, patch_h, patch_w) 或
                 5D numpy 数组 (num_patches_h, num_patches_w, patch_h, patch_w, c)
    """
    # 获取图像和小块尺寸
    if len(image.shape) == 2:
        h, w = image.shape
        c = 1
        image = image.reshape(h, w, c)
    else:
        h, w, c = image.shape
    
    patch_h, patch_w = patch_size[:2]
    
    # 计算小块数量
    num_patches_h = (h - patch_h) // step + 1
    num_patches_w = (w - patch_w) // step + 1
    
    # 创建输出数组
    if len(patch_size) == 2:
        patches = np.zeros((num_patches_h, num_patches_w, patch_h, patch_w))
    else:
        patches = np.zeros((num_patches_h, num_patches_w, patch_h, patch_w, c))
    
    # 提取小块
    for i in range(0, num_patches_h):
        for j in range(0, num_patches_w):
            h_start = i * step
            w_start = j * step
            
            if len(patch_size) == 2:
                patches[i, j] = image[h_start:h_start+patch_h, w_start:w_start+patch_w]
            else:
                patches[i, j] = image[h_start:h_start+patch_h, w_start:w_start+patch_w, :]
    
    # 如果原始图像是2D的，去除多余的维度
    if c == 1:
        patches = patches.squeeze(axis=-1)
    
    return patches

def unpatchify(patches, image_size, step=1):
    """
    将小块重新组合成完整图像
    
    参数:
        patches: patchify函数生成的输出
        image_size: 原始图像尺寸 (h, w) 或 (h, w, c)
        step: 滑动步长 (应与patchify时相同)
    
    返回:
        reconstructed: 重建的图像
    """
    if len(image_size) == 2:
        h, w = image_size
        c = 1
    else:
        h, w, c = image_size
    
    # 获取小块尺寸
    if len(patches.shape) == 4:
        patch_h, patch_w = patches.shape[2:]
        num_c = 1
    else:
        patch_h, patch_w, num_c = patches.shape[2:]
    
    # 计算小块数量
    num_patches_h = (h - patch_h) // step + 1
    num_patches_w = (w - patch_w) // step + 1
    
    # 验证输入
    assert patches.shape[0] == num_patches_h, "小块数量不匹配"
    assert patches.shape[1] == num_patches_w, "小块数量不匹配"
    assert num_c == c, "通道数不匹配"
    
    # 创建输出图像
    if c == 1:
        reconstructed = np.zeros((h, w))
        count_matrix = np.zeros((h, w))
    else:
        reconstructed = np.zeros((h, w, c))
        count_matrix = np.zeros((h, w, c))
    
    # 重建图像
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * step
            w_start = j * step
            
            if c == 1:
                reconstructed[h_start:h_start+patch_h, w_start:w_start+patch_w] += patches[i, j]
                count_matrix[h_start:h_start+patch_h, w_start:w_start+patch_w] += 1
            else:
                reconstructed[h_start:h_start+patch_h, w_start:w_start+patch_w, :] += patches[i, j]
                count_matrix[h_start:h_start+patch_h, w_start:w_start+patch_w, :] += 1
    
    # 避免除以零
    count_matrix[count_matrix == 0] = 1
    reconstructed = reconstructed / count_matrix
    
    # 如果原始图像是2D的，去除多余的维度
    if c == 1:
        reconstructed = reconstructed.squeeze(axis=-1)
    
    return reconstructed
```


### Pytorch版本
```python
import torch
import torch.nn.functional as F

def patchify_torch(image, patch_size, step=1):
    """
    PyTorch 版本的 patchify
    
    参数:
        image: 输入图像张量 (C, H, W) 或 (B, C, H, W)
        patch_size: 小块大小 (h, w)
        step: 滑动步长 (默认为1)
    
    返回:
        patches: 张量 (B, C, num_patches_h, num_patches_w, patch_h, patch_w) 或
                 (C, num_patches_h, num_patches_w, patch_h, patch_w)
    """
    if len(image.shape) == 3:
        # 单张图像 (C, H, W)
        image = image.unsqueeze(0)
        batch_dim = False
    else:
        batch_dim = True
    
    B, C, H, W = image.shape
    patch_h, patch_w = patch_size
    
    # 使用 unfold 操作提取小块
    patches = image.unfold(2, patch_h, step).unfold(3, patch_w, step)
    patches = patches.contiguous()
    
    # 重塑张量
    num_patches_h = patches.shape[2]
    num_patches_w = patches.shape[3]
    patches = patches.view(B, C, num_patches_h, num_patches_w, patch_h, patch_w)
    
    if not batch_dim:
        patches = patches.squeeze(0)
    
    return patches

def unpatchify_torch(patches, image_size, step=1):
    """
    PyTorch 版本的 unpatchify
    
    参数:
        patches: patchify_torch 函数生成的输出
        image_size: 原始图像尺寸 (H, W)
        step: 滑动步长 (应与 patchify 时相同)
    
    返回:
        reconstructed: 重建的图像张量
    """
    if len(patches.shape) == 4:
        # 单张图像的情况 (C, num_patches_h, num_patches_w, patch_h, patch_w)
        patches = patches.unsqueeze(0)
        batch_dim = False
    else:
        batch_dim = True
    
    B, C, num_patches_h, num_patches_w, patch_h, patch_w = patches.shape
    H, W = image_size
    
    # 计算输出尺寸
    out_h = (num_patches_h - 1) * step + patch_h
    out_w = (num_patches_w - 1) * step + patch_w
    
    # 初始化输出和计数矩阵
    reconstructed = torch.zeros(B, C, out_h, out_w, device=patches.device)
    count_matrix = torch.zeros(B, C, out_h, out_w, device=patches.device)
    
    # 填充输出
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * step
            w_start = j * step
            
            reconstructed[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] += patches[:, :, i, j]
            count_matrix[:, :, h_start:h_start+patch_h, w_start:w_start+patch_w] += 1
    
    # 避免除以零
    count_matrix[count_matrix == 0] = 1
    reconstructed = reconstructed / count_matrix
    
    if not batch_dim:
        reconstructed = reconstructed.squeeze(0)
    
    return reconstructed
```

### 可视化分析
```python
import numpy as np
import matplotlib.pyplot as plt

# 创建一个示例图像
image = np.random.rand(256, 256, 3)  # 256x256 RGB图像

# 使用 patchify
patches = patchify(image, patch_size=(64, 64), step=32)
print("Patches shape:", patches.shape)  # 应该是 (7, 7, 64, 64, 3) 对于256x256图像

# 显示一些小块
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        axes[i, j].imshow(patches[i, j])
        axes[i, j].axis('off')
plt.show()

# 重建图像
reconstructed = unpatchify(patches, image_size=(256, 256, 3), step=32)

# 检查重建误差
print("最大重建误差:", np.max(np.abs(image - reconstructed)))
```



## Unpatchify
unpatchify 方法一般用于将分块（patchified）的图像张量重新组合成完整的图像。假设输入是一个分块后的张量 x，形状为 (N, T, patch_size**2 * C)，其中：
- N 是批量大小（batch size）。
- T 是序列长度（可能包含条件、caption等分块的总数）。
- patch_size**2 * C 是每个分块展平后的维度（patch_size×patch_size 的图像块，通道数为 C）。

输出是重新组合后的图像列表（或张量），形状为 (N, H, W, C)
```python

def unpatchify(
        self, x: torch.Tensor, img_size: List[Tuple[int, int]], cond_size: List[int], cap_size: List[int], return_tensor=False
    ) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size # 分块的高度和宽度（假设正方形分块）
        imgs = [] # 存储重建后的图像
        for i in range(x.size(0)):
            H, W = img_size[i]  # 当前样本的目标图像高度和宽度
            begin = cap_size[i] + cond_size[i]  # 跳过caption和条件分块，定位到图像分块的起始索引
            end = begin + (H // pH) * (W // pW)  # 计算图像分块的结束索引
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )  

        if return_tensor:
            imgs = torch.stack(imgs, dim=0) # 合并为形状 (N, C, H, W)
        return imgs
```
