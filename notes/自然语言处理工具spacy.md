
## Spacy

Spacy 是一个高效、易用的开源自然语言处理（NLP）库，专为生产环境设计。它支持多种语言（如英语、中文、德语等），提供词性标注、命名实体识别（NER）、依存句法分析、词向量嵌入等核心功能，且处理速度极快（比NLTK等传统库快数十倍）。

- 预训练模型直接加载，无需额外训练（如en_core_web_sm、zh_core_web_sm）。
- 支持管道（Pipeline）自定义，可灵活添加/移除功能（如禁用NER以加速处理）。



#### 1.安装
```
pip install spacy
```

#### 2.下载预训练模型(英语为例，如果使用中文，en切换为zh)
- 小型模型（en_core_web_sm）：速度快，适合基础任务。
- 中型模型（en_core_web_md）：包含词向量，适合语义分析。
- 大型模型（en_core_web_lg）：更高精度，词向量维度更大。

```
python -m spacy download en_core_web_sm  # 下载英语小型模型

https://github.com/explosion/spacy-models/releases  # release地址
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

#### 3.加载模型
```
import spacy
nlp = spacy.load("en_core_web_sm")  # 加载英语模型
```

#### 4. 应用
4.1 词性标注（POS Tagging）：识别句子中每个词的词性（名词、动词、形容词等）。
```
doc = nlp("The quick brown fox jumps over the lazy dog.")
for token in doc:
    print(f"{token.text:<10} {token.pos_:<10} {token.tag_}")  # tag_为细粒度标签

The        DET        DT
quick      ADJ        JJ
brown      ADJ        JJ
fox        NOUN       NN
jumps      VERB       VBZ
over       ADP        IN
the        DET        DT
lazy       ADJ        JJ
dog        NOUN       NN
.          PUNCT      .
```
- pos_：粗粒度词性（如NOUN、VERB）。
- tag_：细粒度标签（如NN表示单数名词，VBZ表示第三人称单数动词）。

```
# 提取名词与专有名词
def extract_nouns(text):
    doc = nlp(text)
    nouns = [token.text for token in doc if token.pos_ == "NOUN" or token.pos_ == "PROPN"]  # 普通名词+专有名词
    return nouns
```

4.2 命名实体识别（NER）：识别文本中的人名、地名、组织名等实体。
```
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
for ent in doc.ents:
    print(f"{ent.text:<15} {ent.label_}")

Apple           ORG
U.K.            GPE
$1 billion      MONEY
```
- label_：实体类型（如ORG=组织，GPE=国家/地区，MONEY=货币）。

4.3 词形还原（Lemmatization）：将词还原为基本形式（如“running”→“run”）。
```
doc = nlp("The striped bats are hanging on their feet for best.")
for token in doc:
    print(f"{token.text:<10} -> {token.lemma_}")

The        -> the
striped    -> stripe
bats       -> bat
are        -> be
hanging    -> hang
on         -> on
their      -> their
feet       -> foot
for        -> for
best       -> best
.          -> .
```
4.4 语义搜索：计算句子相似度（需词向量支持）
```
doc1 = nlp_md("I like cats.")
doc2 = nlp_md("I love dogs.")
similarity = doc1.similarity(doc2)  # 输出: 0.8（示例值）
```

4.5 信息提取：从新闻中提取公司名、日期、金额等实体。
```
doc = nlp("Google announced Q3 earnings of $15 billion on October 25.")
for ent in doc.ents:
    if ent.label_ in ["ORG", "DATE", "MONEY"]:
        print(f"{ent.label_}: {ent.text}")
```
