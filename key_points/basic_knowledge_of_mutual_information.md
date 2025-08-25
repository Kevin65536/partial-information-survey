# 互信息计算

## 定义

- 熵与条件熵
  - $H(X)$: 随机变量 $X$ 的熵，表示 $X$ 的不确定性。
  - $H(X|Y)$: 在已知随机变量 $Y$ 的情况下，$X$ 的条件熵，表示在给定 $Y$ 的条件下 $X$ 的不确定性。
  - 性质：
    1. $H(X) \geq H(X|Y)$: 已知更多信息后不确定性减小。
    2. $H(X,Y) = H(X) + H(Y|X)$: 联合熵的链式法则。
- 互信息与条件互信息
  - 两变量互信息：
    $$ \begin{aligned}
    I(X;Y) &= H(X) - H(X|Y) \\
    &= H(Y) - H(Y|X) \\
    &= H(X) + H(Y) - H(X,Y)\\
    &=\mathbb{E}_{p(x,y)}\left[\log \frac{p(x,y)}{p(x)p(y)}\right]
    \end{aligned}
    $$
  - 条件互信息：
    $$ \begin{aligned}
    I(X;Y|Z) &= H(X|Z) - H(X|Y,Z) \\
    &= H(Y|Z) - H(Y|X,Z) \\
    &= \mathbb{E}_{p(x,y,z)}\left[\log \frac{p(x,y|z)}{p(x|z)p(y|z)}\right]
    \end{aligned}
    $$

## 三变量情况的讨论

- 三变量交互信息的定义
    $$ \begin{aligned}
    I(X_1;X_2;Y) &= I(X_1;X_2) - I(X_1;X_2|Y) \\
    &= H(X_1) + H(X_2) - H(X_1,X_2) + H(X_1|Y) \\
    &\quad + H(X_2|Y) - H(X_1,X_2|Y) \\
    &= H(X_1) + H(X_2) + H(Y) \\
    &\quad - H(X_1,X_2) - H(X_1,Y) - H(X_2,Y) \\
    &\quad + H(X_1,X_2,Y)
    \end{aligned}
    $$
  - 若$I(X_1,X_2,Y)>0$，即“知道$Y$后，$X_1$和$X_2$之间的互信息减少了”，$Y$解释了$X_1$和$X_2$之间共享的一部分信息，$X_1$和$X_2$之间冗余更多。
  - 若$I(X_1,X_2,Y)<0$，即“知道$Y$后，$X_1$和$X_2$之间的互信息增加了”，$Y$解释了$X_1$和$X_2$之间的协同信息，$X_1$和$X_2$之间协同更多。
