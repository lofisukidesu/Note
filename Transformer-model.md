# 基本结构：

![[Transformer_struct.png]]

## 输入层（Input）

词嵌入（Embedding）:将输入的词转换为词向量

位置编码：Transformer 由于依赖自注意力机制进行处理（无序的），故而对输入的顺序没有感知能力、位置编码提供单词在序列中的位置（绝对/相对）->确保生成的输出遵循正确的顺序
## 编码器（Encoder）

### 多头自注意力机制 - Muti-Head [Self-Attention](自注意力机制.md)

*捕捉多种语义关系*（每个头 学习到不同的语义关系）

*并行计算*（多个角度关注不同部分的信息，不仅仅局限于单一部分 -> 复杂语义的建模能力）

*增强表示能力*（提供了更丰富的上下文信息 -> 增强模型对序列中复杂依赖关系的理解能力）

```

eg. 假设有此句：I love NLP


Head1 关注 "I" 与 "love" 间的词汇语法关系

Head2 关注 "I" 与 "love" 间的情感语义关系

Head3 关注 "love" 与 "NLP" 间的词汇语法关系

  
本质上是让模型多角度理解输入的句子，增强模型语境感知

```

### 前馈全连接网络 - Feed Forward Neural Network

非线性变换

第一层线性变换

*相当于将输入的维度（ d_model ）向量扩展到更高维度（ d_ff ）得到一个更复杂的表示 即输出一个维度为 d_ff 的向量*

$$

Output_{1}= \text{ReLU}(XW_{1}+b_{1})

$$

第二层线性变换

*相当于将 Output1输出的 d_ff 的高维度向量映射回低维度 d_model 即 Output2 是前馈全连接网络处理后的结果*

$$

Output_{2} = Output_{1}W_{1} + b_{2}

$$
信息处理

### 层归一化 - Layer Normalization

层归一化是对每个特征进行标准化

$$

Output = \gamma \cdot \frac{Input - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta

$$

其中 $μ$ 为输入的均值、$σ^2$ 为输入的方差、 $ε$ 为常数（很小 防止分母为0）、$γ$ 为缩放因子、$β$ 为偏置因子

*计算每一层的均值和方差*： 对于每个样本，计算该样本在当前层所有特征的均值和方差

*归一化* ：计算出来的均值和方差对每个特征进行标准化 -> 正态分布<font color = red>（说是归一化，实质用到了标准化的公式）</font>

标准化：$\frac{\text{Input} - \mu}{\sqrt{\sigma^2 + \epsilon}}$ -> 使其具有0均值和单位方差

*缩放和平移* ：引入可学习的缩放因子和偏置因子 -> 恢复输入数据的分布特征，避免标准化过程中过度改变数据

### 残差连接 - Residual Connection

$$

Output = Layer(Input) + Input

$$


存在的意义：

*避免梯度消失或梯度爆炸*

*加速模型收敛* -> 减少训练时间、节省训练资源；避免过度拟合 <font color = red>提高训练效率</font>

*保证信息不会在深层网络中丢失*


## 解码器（Decoder）


基本编码器结构相同 是一个[自回归](自回归.md)模型

### 掩蔽多头自注意力机制 - Masked Multi-Head [Self-Attention](自注意力机制.md)

将输入矩阵进行 掩蔽（masking） 操作获取 Mask 矩阵 其目的是使解码器不能利用后续的词来预测当前的词

$$
\begin{matrix}
i & love & NLP\\
i & love & NLP\\
i & love & NLP
\end{matrix} \tag{undo masked}
$$
$$
\begin{matrix}
i & * & *\\
i & love & *\\
i & love & NLP
\end{matrix} \tag{masked}
$$
其中 * 代表了被遮挡的部分

掩蔽（masking）可以给定也可以随机，取决于具体的模型/任务

给定掩蔽（自回归模型 eg.GPT）：掩蔽的结构是确定的，以确保模型看不到未来的信息
随机掩蔽（自编码模型 eg.BERT）：用于<font color = red>预训练阶段</font> 随机选择一定比例的词汇 将其替换为MASK标记，模型任务为根据上下文信息预测被masked的词汇（提升模型的泛化能力）

Masked Multi-Head Self-Attention 与 Multi-Head Self-Attention 的区别仅在于计算注意力权重时 引入了掩蔽矩阵（定义了哪些位置不允许访问 设置为 $-∞$ ）
