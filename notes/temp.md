# temp file for problems to solve

## calculate $I(X_1,X_2;Y)$ and $I(X_1;X_2;Y)$

$$\begin{aligned}
I(X_1;X_2;Y) &= I(X_1;Y) - I(X_1;Y|X_2) \\
&= I(X_1;X_2) - I(X_1;X_2|Y)
\end{aligned}
$$

$I(X_1;X_2;Y)$的含义是，知道了$Y$之后，$X_1$和$X_2$之间的互信息量发生了多少变化。

$$\begin{aligned}
I(X_1,X_2;Y) &= I(X_1;X_2;Y) + I(X_1;Y|X_2) + I(X_2;Y|X_1)\\
&= I(X_1;Y) - I(X_1;Y|X_2) + I(X_1;Y|X_2) + I(X_2;Y|X_1)\\
&= I(X_1;Y) + I(X_2;Y|X_1)\\
&= H(Y) - H(Y|X_1) + H(Y|X_1) - H(Y|X_1,X_2)\\
&= H(Y) - H(Y|X_1,X_2)\\
&= I(X_1,X_2;Y)
\end{aligned}
$$

### XOR example

$$\begin{align}
Y &= X_1 \oplus X_2\\
I(X_1,X_2;Y) &= 1 \text{ bit}\\
I(X_1;X_2;Y) &= I(X_1;X_2) - I(X_1;X_2|Y) = 0 \text{ bit} - 1 \text{ bit} = -1 \text{ bit}\\
Red(X_1,X_2;Y) &= 0 \text{ bit}\\
Unq(X_1;Y|X_2) &= I(X_1;Y) - Red = 0 \text{ bit}\\
Unq(X_2;Y|X_1) &= I(X_2;Y) - Red = 0 \text{ bit}\\
Syn(X_1;X_2;Y) &= I(X_1,X_2;Y) - Red -Unq(X_1) - Unq(X_2) = 1 \text{ bit}
\end{align}$$

引起双变量到三变量扩展问题的一个可能原因是，在互信息中，$;$算符表达的是两个变量之间的交互。对于两个变量，$I(X_1;X_2) = H(X_1) - H(X_1|X_2)$，而对于三个变量，$I(X_1;X_2;Y)$不能写作$H(X_1;X_2) - H(X_1;X_2|Y)$，$H(X_1;X_2)$没有定义。$I(X_1;X_2;Y)$只能表示为$I(X_1;X_2) - I(X_1;X_2|Y)$。

$$ I(X_1;X_2;Y) = Red(X_1,X_2;Y) - Syn(X_1,X_2;Y) $$
$$ I(X_1;Y|X_2) = Unq(X_1;Y|X_2) + Syn(X_1,X_2;Y) $$
$$ I(X_1;Y) - Red = Unq(X_1;Y|X_2) = I(X_1;Y|X_2) - Syn(X_1,X_2;Y) $$
