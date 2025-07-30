## Matplotlib库函数学习

```python
import matplotlib.pyplot as plt
import torch

# 模拟一个 PyTorch 图像张量 (C, H, W)
image = torch.rand(3, 256, 256)  # 随机RGB图像

plt.figure(figsize=(10, 5)) # 创建一个新的图形窗口（Figure 对象）,设置图形的尺寸为宽度 10 英寸、高度 5 英寸。
plt.subplot(1, 2, 1) # 在图形窗口中创建一个子图（Axes 对象），并指定其位置。
# 1, 2, 1：表示将图形区域划分为 1行×2列的网格，并激活第1个子图（索引从1开始）。

plt.imshow(image.permute(1, 2, 0))  # 调整为 (H, W, C) 并显示
plt.title("Original Image")
plt.axis('off')  # 关闭坐标轴

plt.subplot(1, 2, 2)
plt.imshow((image * 0.5).permute(1, 2, 0))  # 显示调整亮度后的图像
plt.title("Darker Image")
plt.show()
```

#### 1.PyTorch 张量与 NumPy 数组
- 如果 image 是 PyTorch 张量，permute 是合法的。如果是 NumPy 数组，应使用 np.transpose。
- 可能需要先通过 .numpy() 转换（如果 image 在 GPU 上，还需先移到 CPU：.cpu()）。

```
plt.imshow(image.permute(1, 2, 0).numpy())  # 如果 image 是 PyTorch 张量
```

#### 2.归一化问题
- 如果 image 是归一化后的张量（如值在 [-1, 1]），需反归一化：
```
image = (image.permute(1, 2, 0) + 1) / 2  # 将 [-1,1] 映射到 [0,1]
plt.imshow(image)
```
