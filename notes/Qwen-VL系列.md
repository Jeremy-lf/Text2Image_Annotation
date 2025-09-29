## Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
具备能力：图像描述Image Caption、视觉问答VQA、视觉定位Visual Grounding、文本阅读Text Reading

模型结构：
（1）LLM：QWen-7B初始化；
（2）visual encoder：使用ViT结构，用Openclip的ViT-bigG初始化，输入图片resize到固定大小448x448，patch=14；
（3）Position-aware Vision-Language Adapter：为了解决long image feature sentence带来的效率问题，Qwen-VL用了一个adapter层对图像特征进行压缩，adaper是一个随机初始化的单层cross-attention，使用一组可训练的query vector作为query，图像特征作为key，position embedding使用2维绝对位置编码，最终把图像特征的长度压缩到256，送入llm中。

训练流程：训练过程分为3个阶段，2个预训练，1个是指令微调

1.图文对预训练：用大规模弱标注的互联网爬虫图文pair数据进行预训练， 只训练adapter与LLM；

2.多任务预训练阶段：用一些高质量精细标注的数据进行训练，包括7种不同的任务（caption\VQA\Grounding\Ref\OCR），训练visual encoder、adapter与LLM；

3.SFT:通过指令微调来对Qwen-VL预训练模型进行微调，以增强其指令遵循和对话能力。多模态指令调整数据主要来自通过LLM生成的图片描述或对话数据，这些数据通常只涉及单图像对话和推理，并且仅限于图像内容理解。因此这里通过手动标注、模型生成和策略串联来构建一些额外的对话数据，以将定位和多图像理解能力纳入Qwen-VL模型。冻结ViT，训练adapter与LLM；
![Pretraining](https://github.com/user-attachments/assets/df01dea1-92fd-42c3-9f03-06a84f681c1d)
<img width="1080" height="865" alt="image" src="https://github.com/user-attachments/assets/3a1b2f70-724c-4817-ae5c-208fb78c92b1" />

---

## Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution
#### 1.Naive Dynamic Resolution 
传统LVLM需将所有输入图像缩放/填充至固定分辨率（如224×224），导致高分辨率图像细节丢失（如文档小字、复杂图表）或低分辨率图像冗余计算。Qwen2-VL则让模型像人类视觉一样“按需分配注意力”——不同分辨率图像生成不同数量的视觉令牌，无需强制统一尺寸。
* 移除绝对位置嵌入：替换传统ViT的固定位置嵌入，引入2D-RoPE（二维旋转位置嵌入） ，精准捕获图像的空间坐标信息（高度、宽度），适配任意宽高比。
* 动态令牌压缩：通过简单MLP层将相邻2×2的视觉令牌压缩为1个，搭配<|vision_start|>/<|vision_end|>特殊令牌标记边界，在保证精度的同时控制序列长度。
* 分辨率边界控制：仅设定min_pixels（100×28×28）与max_pixels（16384×28×28），令牌数量完全由图像原生分辨率决定，避免过度缩放。

#### 2.多模态旋转位置嵌入（M-RoPE）
传统LVLM用1D-RoPE处理文本，用独立机制处理图像/视频位置，导致多模态位置信息割裂——无法有效关联“文本描述的空间位置”与“图像中的实际坐标”，也难以建模视频的“时空动态”（如动作顺序）。
* 三维组件拆解：将旋转嵌入拆分为时间、高度、宽度三个独立组件，分别适配不同模态的位置特性：（2）文本：三组件共享同一位置ID，等效传统1D-RoPE，保持语言理解能力；（2）图像：时间ID固定（静态），高度/宽度ID随像素坐标变化（捕捉空间位置）；（3）视频：时间ID随帧序号递增（捕捉时序），高度/宽度ID同图像（捕捉帧内空间）。
* 长序列外推能力：通过降低图像/视频的位置ID数值，模型可在训练时仅支持16K令牌的情况下，推理时处理80K令牌（对应超20分钟长视频），突破序列长度限制。

#### 3.图像与视频的一体化处理范式
传统模型需为图像和视频设计两套编码器（如2D CNN处理图像、3D CNN处理视频），导致架构冗余且跨模态迁移能力弱（如无法用图像知识辅助视频理解）。
* 统一输入格式：将图像视为“2帧完全相同的视频”，视频按2帧/秒采样，确保图像与视频的输入结构一致；
* 3D卷积处理动态信息：引入深度为2的3D卷积，将视频帧转换为“3D管状物”（而非孤立2D补丁），捕捉帧间的时序关联，同时避免序列长度爆炸（单视频令牌限制16384）。

#### 训练方案:
* 三阶段训练：仅训练视觉编码器 → 全参数训练 → LLM指令微调
* 训练数据：1.4万亿tokens数据集（含图像-文本对、OCR数据等） 
<img width="1434" height="1002" alt="image" src="https://github.com/user-attachments/assets/29fe5bac-991c-40bf-a288-e0aad900f414" />
<img width="1080" height="507" alt="image" src="https://github.com/user-attachments/assets/a1772c02-5c31-4473-88ac-fad58be41e0d" />



## Qwen2.5-VL
#### 1. 融合窗口注意力的动态分辨率视觉编码器

核心创新：重新设计视觉Transformer（ViT），在多数层采用 窗口注意力（Window Attention），仅4层保留全注意力。窗口尺寸固定为112×112（对应8×8 patch），小于该尺寸的区域无需填充，直接以原生分辨率处理。

解决痛点：传统ViT的全注意力计算复杂度随图像尺寸呈二次增长，难以适配大尺寸图像或高分辨率文档；该设计将复杂度降至线性，同时避免强制分辨率归一化导致的空间信息失真（如小物体丢失、比例失衡）。

技术细节：采用14×14 patch尺寸，输入图像仅调整为28的整数倍（而非固定尺寸），配合2D-RoPE（旋转位置编码）精准捕捉空间关系，且ViT与LLM架构对齐（统一使用RMSNorm归一化、SwiGLU激活函数），提升跨模态兼容性。

#### 2.轻量化视觉-语言融合器
核心创新：通过“分组压缩+MLP投影”实现视觉特征与文本特征的高效对齐——将空间相邻的4个ViT patch特征分组拼接，再通过两层MLP投影至LLM的文本嵌入维度。

解决痛点：直接输入原始ViT patch序列会产生大量tokens，导致LLM计算负担过重；该设计动态压缩特征长度，在降低开销的同时保留关键空间信息。


#### 3. 动态FPS采样与绝对时间编码
核心创新：将“动态分辨率”从空间维度拓展至时序维度，通过 动态FPS采样 适配不同帧率的视频（如15fps、30fps），同时将 多模态旋转位置编码（MRoPE）的时序分量与绝对时间戳对齐。

解决痛点：传统LVLM处理视频时依赖固定帧率采样或文本时间标记，难以捕捉事件节奏（如“快速移动”“长时间静止”），且无法精准定位秒级事件；该设计通过时间ID间隔直接学习时序动态，无需额外计算开销。

技术效果：支持处理长达数小时的视频，在Charades-STA（事件时序定位）中实现50.9的mIoU，显著超越GPT-4o（35.7）。

<img width="1080" height="505" alt="image" src="https://github.com/user-attachments/assets/587170bd-f7b0-4c46-8f37-d966734c4b33" />

## Qwen3-VL
### Interleaved-MRoPE：重塑多模态位置编码​​
Qwen3-VL 对多模态旋转位置编码（M-RoPE）的布局进行了重构，这是其提升时空联合建模能力的关键
* ​​问题背景​​：Qwen2.5-VL 的M-RoPE采用分块布局（[TTT...HHH...WWW]），即将时间（T）、高度（H）、宽度（W）维度的信息分别集中编码，这种布局可能导致各维度信息关联性不强。

* ​​解决方案：Qwen3-VL 将其改为​​交错式布局​​（[THTHWHTHW...]），将T、H、W三个维度的频率信息交织在一起。
* ​​核心价值​​：原始MRoPE按时间(t)、高度(h)、宽度(w)顺序分块，导致时间信息全部分布在高频维度上。MRoPE-Interleave通过交错排列的方式改进了原始MRoPE对特征维度的分块方式，优化了维度分配。

### DeepStack：多层视觉特征融合​​
Qwen3-VL 引入了 ​​DeepStack 机制​​，彻底改变了视觉特征与语言模型的融合方式，该机制基于论文《DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs》（https://arxiv.org/pdf/2406.04334）
* 问题背景​​：传统多模态模型通常只将视觉编码器（ViT）最后一层的输出特征注入语言模型，这可能损失图像的低级细节（如纹理、边缘）信息。
* ​​解决方案：DeepStack 机制将传统多模态大模型(LMM)的单层视觉tokens输入改为多层注入，即从ViT的​​不同中间层（如第8, 16, 24层）提取多层次特征​​。这些特征不会一次性全部输入语言模型，而是在语言模型解码的​​特定层级​​，通过残差连接的方式，​​叠加到对应位置的文本隐藏状态上​​。
    * 将不同的视觉token输入到LLMs的不同层中，从而显著减轻视觉token引入的效率开销
    * 将token堆叠成网格，从下至上注入到第一层和中间的Transformer层
    * 堆栈的每一层通过简单的残差连接与LLMs中的一层相连

​​核心价值​​：类似于计算机视觉中的特征金字塔网络（FPN），让语言模型在推理的不同阶段（浅层处理细节，深层处理语义）都能接触到最相关的视觉信息。这显著提升了对图像​​细节的捕捉能力​​，例如在识别模糊文字、复杂图表结构时表现更优。

### 文本时间戳对齐：优化视频时序理解​​
为了更精准地理解视频中的时序信息，Qwen3-VL 对视频的时序建模机制做了重要简化
* ​​问题背景​​：Qwen2.5-VL 使用 T-RoPE，需要依赖“绝对时间id”和“动态fps”等复杂参数来计算每一帧的位置。
* ​​解决方案：Qwen3-VL 将其简化为​​文本时间戳对齐​​。具体来说，模型将输入的视频帧按顺序排列，每一帧的​​行号（即其在列表中的位置）就作为一个天然的、线性的时间戳​​。这个顺序信息被直接用于和文本Token进行对齐，无需额外的绝对时间编码。
* ​​核心价值​​：这种改动降低了时序建模的复杂度，使模型能够更稳定、精确地进行​​帧级事件定位​​（例如，准确回答“视频中第10秒发生了什么”）。模型原生支持输出“秒数”和“时:分:秒”两种格式，进一步提升了实用性。

<img width="1920" height="1109" alt="image" src="https://github.com/user-attachments/assets/f3efbfe7-7106-4c97-ba47-9ad69bf3b8c6" />

![image](https://github.com/user-attachments/assets/9e3b99e2-1fcf-42d7-a354-ce750b0f7368)


参考：https://mp.weixin.qq.com/s/Y3k86yJRWHdZahizC3ygHg
