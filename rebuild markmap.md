# PID 方法综述与研究想法

## 0. 导航

## 1.Factorized Contrastive Learning (FactorCL)

- 概述：通过对比学习框架，因子化地捕捉多模态数据中的冗余和独特信息。
- 增强表示：$I(X;X')$=$I(X;Y)$。$X'$保留了全部与任务相关的信息，而排除了任务无关信息。

## 2.What to align in multimodal learning (CoMM)

- 概述：通过先融合再对比的方式，捕捉多模态数据中的协同信息。
- 增强表示：

## 3.切入点

## 4.部分信息分解方法固有的局限性

- 计算复杂性

  - ```text
    直接计算任务相关互信息$I(X;Y)$是NP-hard的。
    ```

- 其他限制