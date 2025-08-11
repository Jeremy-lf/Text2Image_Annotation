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

---

## Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution
