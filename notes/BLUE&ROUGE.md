## ROUGE分数&BLEU分数

ROUGE分数与BLEU分数是自然语言处理中用于评估生成文本质量的两种核心指标，ROUGE侧重召回率，通过重叠单元评估信息覆盖度；BLEU侧重精确率，通过n-gram匹配和长度惩罚评估翻译质量。

### ROUGE分数原理与实现
原理：ROUGE（Recall-Oriented Understudy for Gisting Evaluation）由微软研究院提出，最初用于自动文本摘要评估，通过比较生成文本与参考文本之间的重叠单元（如n-gram、最长公共子序列）来衡量信息覆盖度。其核心思想是：好的生成文本应包含更多与参考文本相同的词语或短语。ROUGE包含多个变体，如ROUGE-N（基于n-gram重叠）、ROUGE-L（基于最长公共子序列）等。

#### 核心原理
‌*召回率导向‌：侧重参考文本内容覆盖度，而非生成文本的精确性。*

常见类型‌：
- ROUGE-N：基于n-gram（如ROUGE-1为unigram，ROUGE-2为bigram）的召回率。
- ROUGE-L：基于最长公共子序列（LCS）的F1值，衡量语义连贯性。 ‌

#### 计算示例
- ROUGE-1：候选文本“the cat sits”与参考文本“the cat is on the mat”的unigram重叠度为2/6≈0.333。 ‌
- ROUGE-L：LCS长度为2（“the cat”），F1值为0.444。 ‌

#### 应用场景
- 评估摘要生成或翻译模型的输出质量。
- 与BLEU指标互补，BLEU侧重精确度，ROUGE侧重召回率。

---

### BLEU分数原理与实现
原理：BLEU（Bilingual Evaluation Understudy）由IBM提出，最初用于机器翻译质量评估，通过比较生成文本与参考文本之间的n-gram匹配程度来量化翻译质量。其核心思想是：好的翻译结果应包含更多与参考译文相同的连续词组。BLEU分数范围为0-100分，分数越高表示翻译质量越好。‌
