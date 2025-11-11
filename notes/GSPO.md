## GSPO
动机：GRPO在训练巨大的语言模型时表现出严重的稳定性问题，通常会导致灾难性和不可逆的模型崩溃。根本原因在于其算法设计中对重要性权重的使用不合理（token-Level的重要性权重，无法起到预期的矫正分布作用），这引入了高方差的训练噪声，并在响应序列长度增加和截断机制的作用下进一步累积和放大，最终导致模型崩溃。

核心原则：优化目标的单位应与奖励单位相匹配，the unit of optimization objective should match the unit of reward.

GSPO与之前采用token-level重要性比的算法不同，GSPO基于序列似然性定义重要性比，并执行序列级剪切clip、奖励和优化。此外，GSPO将归一化奖励计算为对查询的多个响应的优势，确保序列级奖励和优化之间的对齐。请注意，我们采用si（θ）的长度归一化来减小方差，并将si（θ）控制在统一的数值范围内。
因此，GSPO对整个response而不是单个token应用剪切，以排除过度梯度估计中的“非策略off-policy”样本，与序列级奖励和优化相匹配。
否则，几个token的可能性变化可能会导致序列级重要性比的剧烈波动，不同长度的响应的重要性比将需要不同的剪切范围。

![image](https://github.com/user-attachments/assets/64e00c8b-6446-453c-b270-2663ae1f47f5)
![image](https://github.com/user-attachments/assets/096818df-a0dc-4de4-a62f-7a7c61dc7c98)
![image](https://github.com/user-attachments/assets/dd637c8c-7296-42ef-ab09-dd10ea76bec7)
![image](https://github.com/user-attachments/assets/3e729379-cbc3-4ca9-89f9-4df9065e8e2d)


