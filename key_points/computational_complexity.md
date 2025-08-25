# computational complexity in mutual information estimation

## 对于现实数据集，直接计算任务相关互信息$I(X;Y)$具有相当的困难

互信息的原始数学定义为：

$$ I(X;Y) = \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} dx dy $$

直接计算这一积分，需要三个概率密度函数：

- $p(x,y)$：联合概率密度函数
- $p(x)$：边缘概率密度函数
- $p(y)$：边缘概率密度函数

对于现代机器学习中涉及的数据，直接计算这些概率密度函数是非常困难的。

1. 对于样本x，其维度通常相当巨大
2. 在巨大的样本空间中，相对较少的样本数量使得分布极其稀疏
3. 在高维空间中，任何一个点的“邻域”内几乎都没有其它样本，使得传统密度估计方法失效

在实际计算中，通常需要使用近似方法，并在原始信息编码出的低维代理表示上进行优化。

## 使用InfoNCE与InfoNCE-CLUB的互信息上下界估计

### InfoNCE（下界估计）

InfoNCE的理论基础是，如果一个模型能够很好的区分**正样本对**（来自联合分布$p(x,y)$）和**负样本对**（来自边缘分布的乘积$p(x)p(y)$），那么它就间接的捕获了X与Y之间的互信息。

$$
\mathcal{L}_{\text{InfoNCE}}(\theta)
= \mathbb{E}\left[
-\log \frac{\exp\big(s_\theta(x,y)\big)}
{\exp\big(s_\theta(x,y)\big) + \sum_{j=1}^{K-1} \exp\big(s_\theta(x,\tilde y_j)\big)}
\right].
$$

$K$为负样本数量，$s_\theta$为一个相似度函数。

可证明其与互信息满足下界关系：

$$
I(X;Y) \;\ge\; \log K \;-\; \mathcal{L}_{\text{InfoNCE}}(\theta).
$$

- 若 $s_\theta$ 能让正样本分数显著高于所有负样本，则分类难度低、交叉熵小，下界更紧。
- $\log K$ 是下界的天花板：当 $K$ 较小、真实互信息很大时，下界会“饱和”（低估）。

### CLUB（上界估计）

CLUB（Contrastive Log-ratio Upper Bound）从条件对数似然出发，构造互信息的上界。对真实条件分布 $p(y|x)$，有

$$
I(X;Y)
\;\le\;
\underbrace{\mathbb{E}_{p(x,y)}\!\left[\log p(y|x)\right]
-\mathbb{E}_{p(x)}\mathbb{E}_{p(y)}\!\left[\log p(y|x)\right]}_{\text{CLUB}(p)}.
$$

证明思路（要点）：

- 利用 $p(y)=\mathbb{E}_{p(x)}[p(y|x)]$ 与 Jensen 不等式，对每个 $y$ 有
  $E_{p(x)}[\log p(y|x)] \le \log E_{p(x)}[p(y|x)] = \log p(y)$。
- 取对 $p(y)$ 的期望可得 $\text{CLUB}(p) - I(X;Y) \ge 0$，因此 CLUB 确为上界。

实际不可直接用 $p(y|x)$，于是以参数化条件密度 $q_\phi(y|x)$ 近似 $p(y|x)$，得到可计算的上界估计量：

$$
\widehat{I}_{\text{CLUB}} =
\frac{1}{N}\sum_{i=1}^N \log q_\phi(y_i|x_i)
\;-\;
\frac{1}{N}\sum_{i=1}^N \frac{1}{M}\sum_{j \in \mathcal{N}_i} \log q_\phi(y_j|x_i),
$$

其中第二项用不配对的 $(x_i,y_j)$ 近似 $\mathbb{E}_{p(x)}\mathbb{E}_{p(y)}[\log q_\phi(y|x)]$，$\mathcal{N}_i$ 是对每个 $x_i$ 选取的 $M$ 个负样本索引（常用同一 batch 内的 $j\neq i$）。

训练与复杂度要点：

- 先以最大似然（仅正对）训练 $q_\phi$ 近似 $p(y|x)$，例如对连续 $y$ 用高斯条件模型 $q_\phi(y|x)=\mathcal{N}(y;\mu_\phi(x),\Sigma_\phi(x))$，其 $\log q_\phi$ 有闭式。
- 估计阶段采用 in-batch 负样本，全部配对是 $O(B^2)$；也可子采样 $M$ 个负样本，降为 $O(BM)$。
- 性质：上界、偏大。若 $q_\phi$ 逼近真实 $p(y|x)$，上界更紧；模型失配会导致上界偏松。

### 夹逼估计：InfoNCE-CLUB

把 InfoNCE 下界与 CLUB 上界联合使用，可得到对互信息的夹逼区间：

$$
\underbrace{\log K \;-\; \mathcal{L}_{\text{InfoNCE}}(\theta)}_{\text{Lower}}
\;\;\le\;\;
I(X;Y)
\;\;\le\;\;
\underbrace{\widehat{I}_{\text{CLUB}}(q_\phi)}_{\text{Upper}}.
$$

实践策略：

- 作为评估：若上下界间隙较小，则可较为可靠地“读数”。若间隙很大，需增大 $K$、改进打分函数 $s_\theta$ 或提升条件模型 $q_\phi$ 的表达能力。
- 作为训练约束：最大化下界（提升依赖）同时最小化上界（抑制不必要依赖），常见于多模态表征/解耦因果/隐私抑制等任务。
