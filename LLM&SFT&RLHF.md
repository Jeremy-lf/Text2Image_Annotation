## LLM


### 1.训练

训练流程：预训练、SFT微调、奖励模型、强化学习

```
# SFT数据
{ 
    "Instruction":"",
    "Input":"",  //Input字段为可选字段，有时Instruction部分会包含Input的内容
    "Output":""
}

// 例子:  
{
    "Instruction":"请帮我翻译一句话",
    "Input":"hello",            
    "Output":"你好"
},
{
    "Instruction":"请帮我翻译一句话:hello",
    "Input":"",  
    "Output":"你好"
}
```

数据编码

`[Instruction] + [分隔符] +（[Input]）+ [分隔符] + [Output]`

损失函数：交叉熵损失

---

### 2.奖励模型（Reward Model）
奖励模型（Reward Model, RM）是强化学习与人类反馈（RLHF）过程中至关重要的一环，它的作用是评估大语言模型输出的文本质量，给出一个分数，指导模型在后续生成过程中更好地符合人类偏好和需求。

#### 2.1为什么需要奖励模型？
在指令微调（SFT）阶段，虽然模型已经被训练并具备一定的语言生成能力，但其输出结果仍然可能不符合人类的偏好，可能存在「幻觉」问题（模型生成的内容不真实或不准确）或者「有害性」问题（输出有害、不合适或令人不安的内容）。

这是因为，SFT 仅通过有限的人工标注数据来微调预训练模型，可能并未完全纠正预训练阶段中潜在的错误知识或不合适的输出。为了进一步提高模型的生成质量，解决这些问题，必须引入奖励模型，利用强化学习进行进一步优化。

#### 2.2 强化学习与奖励模型
强化学习的核心思想是通过奖惩机制来引导模型的学习。 在 RLHF（强化学习与人类反馈）中，奖励模型负责为模型生成的每个响应提供一个奖励分数，帮助模型学习哪些输出符合人类的期望，哪些输出不符合。

奖励模型的训练数据通常来自人工标注的排序数据，标注员会对多个生成的回答进行排名，奖励模型基于这些排名进行训练。

与传统的有监督学习不同，奖励模型并不要求直接对每个输出给出明确的分数，而是通过相对排序的方式对多个输出进行比较，告诉模型哪些输出更好，哪些输出更差。这种相对排序方式能有效减少人工评分时的主观差异，提高标注的一致性和模型的学习效率。
```
//基于比较的数据格式
{
    "input": "用户输入的文本",
    "choices": [
        {"text": "候选输出 1", "rank": 1},
        {"text": "候选输出 2", "rank": 2}
]
}

//基于评分的数据格式
{
    "input": "用户输入的文本",
    "output": "生成模型的输出文本",
    "score": 4.5
}
```

奖励模型的输入包括：
- 输入文本：用户给定的提示或问题，作为上下文。
- 输出文本：生成模型的候选答案，用于评估质量。
- 上下文和候选文本拼接：奖励模型通常会将 input（上下文）和每个 choice（候选文本）进行拼接，然后输入到模型中。这样，模型就能够理解生成文本与上下文之间的关系，并基于该关系来评估生成文本的质量。

奖励模型的设计和训练过程中存在一定的挑战，主要体现在以下几个方面：

(1) 人类偏好的多样性：不同的标注员可能对同一生成结果有不同的看法，这就要求奖励模型能够容忍一定的主观性，并通过排序学习来减少偏差。

(2) 模型不稳定：由于奖励模型通常较小，训练过程中可能会出现不稳定的情况。为了提高训练的稳定性，奖励模型通常会采取合适的正则化技术和优化方法。

(3) 数据质量与多样性：为了确保奖励模型的有效性，训练数据需要足够多样化，涵盖不同类型的问题和答案。如果数据质量不高或过于单一，模型可能无法学到有效的评分规则。

---

### 3.评价模型（Critic Model）
Critic Model用于预测期望总收益 ，和Actor模型一样，它需要做参数更新。实践中，Critic Model的设计和初始化方式也有很多种，例如和Actor共享部分参数、从RW阶段的Reward Model初始化而来等等。

在RLHF中，我们不仅要训练模型生成符合人类喜好的内容的能力（Actor），也要提升模型对人类喜好量化判断的能力（Critic）。这就是Critic模型存在的意义。

---

### 4.强化学习（Reinforcement Learning）PPO
强化学习的目标就是模型可以自我迭代，其损失函数包括两部分构成：
- Actor loss：用于评估Actor是否产生了符合人类喜好的结果;
- Critic loss：用于评估Critic是否正确预测了人类的喜好;

想达到以上要求，所以设计出了如下一系列的训练方法，一共有四个主要模型，分别是：
- Actor Model：演员模型，这就是我们想要训练的目标语言模型
- Critic Model：评价模型，它的作用是预期收益
- Reward Model：奖励模型，它的作用是计算实际收益
- Reference Model：参考模型，它的作用是给语言模型增加一些“约束”，防止语言模型训歪，使模型的回答结果最好与之前SFT模型的回答分布相近

Actor与Reference的初始化模型就是SFT模型，Reward与Critic的初始化模型就是Reward模型，其中Actor与Critic在后续训练中需要更新参数，而Reward与Reference Model是参数冻结的。

<img width="610" height="900" alt="image" src="https://github.com/user-attachments/assets/18b27c18-ee2b-41b3-91d8-20b821a8d229" center />




Ref模型的作用是衡量Actor模型生成的完整响应（response）的质量。如果仅输入"prompt"，Ref模型会生成自己的响应，这与Actor的输出无关，无法直接比较两者分布的差异。输入"prompt + response"是为了让Ref模型在相同的上下文和输出序列下提供评价，确保 ref_log_probs 和 log_probs 可直接比较，从而准确衡量Actor模型生成与参考分布的相似度。这是对齐评估（Alignment Evaluation）中的常见做法。


[为什么Ref模型的输入是Actor模型的输出response+prompt](https://yiyan.baidu.com/share/SiCfdxvp5H)

[强化学习详细解读](https://zhuanlan.zhihu.com/p/677607581)

[RLHF解读参考](https://xfyun.csdn.net/68536b31eabc61314fc359f2.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6NjI3NTU4LCJleHAiOjE3NTQ2Mzc3MjYsImlhdCI6MTc1NDAzMjkyNiwidXNlcm5hbWUiOiJKZXJlbXlfbGYifQ.SEV6pQC7zFNSMQc7I1XQZazy82sAgavWAMtfoMQWSKY&spm=1001.2101.3001.6650.12&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-144420491-blog-148930417.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7Eactivity-12-144420491-blog-148930417.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=20)


---

### 推理



[HuggingFace关于Toknizer与Model的解释](https://mp.weixin.qq.com/s?__biz=MzU5MzcwODE3OQ==&mid=2247485590&idx=1&sn=5a79f0e7719c95fafb2a06b374d92b25&chksm=fe0d1d6ac97a947c4e88b8a8d5c8dee52da707a5a7252e5b6717c22512326508469ead61b3e0&cur_album_id=1343689864403042305&scene=189#wechat_redirect)

### 代码
```python
class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(  
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()  # 在序列预测任务中，我们通常用前n-1个token来预测第n个token。
            shift_labels = labels[..., 1:].contiguous() # 取labels从第二个元素开始的所有元素。这样shift_labels中的每个元素就对应于shift_logits所预测的token。
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

```
