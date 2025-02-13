
### 核心：计算输入向量间的相似度 动态分配不同的注意力权重（<font color = red >输入元素在处理是可以关注到输入序列中于其他元素的相关性</font> ）
### 输入的每个词向量，生成三个不同的向量（通常通过输入词向量线性变换得到）

#查询（Query） 词向量 “想要获取”的信息

#键（Key） 词向量“拥有”的信息

#值（Value） 词向量包含的实际信息

### 计算 #查询（Query）与 #键（Key）之间的相似度 得到注意力权重

#### 点积计算

$$
\text{score}(Q_i, K_j) = Q_i \cdot K_j = \sum_{d=1}^D Q_i^{(d)} \cdot K_j^{(d)}
$$

#### 缩放
点积的值随着 #查询（Query）和 #键（Key）的增大而增大 ->数值过大-> 影响梯度的稳定性

$$
\text{scaled score}(Q_i, K_j) = \frac{Q_i \cdot K_j}{\sqrt{d_k}}
$$

#### 计算注意力权重
每个 #查询（Query）与所有 #键（Key）的点击结果 通过 #softmax函数 计算出每个键对应的注意力权重

- 掩蔽矩阵不存在
$$
\text{Attention Weights}(Q_i) = \text{softmax}\left( \frac{Q_i \cdot K_1}{\sqrt{d_k}}, \frac{Q_i \cdot K_2}{\sqrt{d_k}}, \dots, \frac{Q_i \cdot K_n}{\sqrt{d_k}} \right)
$$
- 掩蔽矩阵存在
$$
\text{Attention Weights}(Q_i) = \text{softmax}\left( \frac{Q_i \cdot K_1}{\sqrt{d_k}}, \frac{Q_i \cdot K_2}{\sqrt{d_k}}, \dots, \frac{Q_i \cdot K_n}{\sqrt{d_k}} + \text{Mask} \right)
$$

#### 加权求和
$$
\text{Attention Output} = \sum_{j=1}^n \text{Attention Weights}(Q_i, K_j) \cdot V_j
$$


> softmax 函数 将一组实数转换为一个概率分部 使得输出值的和为1
> 
> $$
> p_i ={\frac{e^{z_i}}{\sum_{j=1}^ne^{z_j}}} 
> $$
> 大白话 将一组数求和取平均值后将平均值赋值给组中全部的数