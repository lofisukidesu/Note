## 基本结构：

#输入层（Input）
词嵌入（Embedding）将输入的词转换为词向量

位置编码 Transformer 没有序列性 位置编码提供单词在序列中的位置

 #编码器（Encoder）

 #解码器（Decoder）

## 构成（由多个层（Layer）堆叠而成，每层包含如下）：

### 多头自注意力机制 Muti-Head [Self-Attention](Self-Attention.md)

*捕捉多种语义关系*（每个头 学习到不同的语义关系）

*并行计算*（多个角度关注不同部分的信息，不仅仅局限于单一部分 -> 复杂语义的建模能力）

*增强表示能力*（提供了更丰富的上下文信息 -> 增强模型对序列中复杂依赖关系的理解能力）

```
eg. 假设有此句：I love NLP

Head1 关注 "I" 与 "love" 
Head2 关注 "love" 与 "NLP"

本质上是让模型多角度理解输入的句子，增强模型语境感知
```
### 前馈全连接网络 Feed Forward Neural Network

非线性变换

第一层线性变换
*相当于将输入的维度（ d_model ）向量扩展到更高维度（ d_ff ）得到一个更复杂的表示 即输出一个维度为 d_ff 的向量*
$$
Output_{1}= \text{ReLU}(XW_{1}+b_{1})
$$
第二层线性变换
*相当于将 Output1输出的 d_ff 的高维度向量映射回低维度 d_model 即 Output2 是前馈全连接网络处理后的结果*
$$
Output_{1} = Output_{1}W_{1} + b_{2}
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

