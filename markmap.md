# PID 方法综述与研究想法（Markmap）

## 0. 导航
- 关联笔记
  - `usable pid method.md`（实现细节观察）
  - `PID_to_do.md`（FactorCL vs CoMM 对比、增强哲学）
  - `direct pid method survey.md`（dit / pidpy 工具与实操）
  - `pid with biological meaning.md`（生理意义对齐与PGID框架）
  - `a full copy of the coversation session.md`（FactorCL 原文摘录）
- 目标
  - 形成“概念—方法—工具—生理对齐—仿真—增强—评估—TODO”的统一视图
  - 面向 EEG + fNIRS 的落地可操作清单

## 1. 基础概念与分解视角
- PID（部分信息分解）
  - 目标变量 $Y$，源变量 $X_1,X_2$
  - 总互信息分解：$I(X_1,X_2;Y)=R+U_1+U_2+S$
    - $R$: 冗余信息
    - $U_i$: 源 $X_i$ 的独特信息
    - $S$: 协同信息
- 另一种基于条件互信息的任务相关分解（CMI 视角）
  - $I(X_1,X_2;Y)=I(X_1;X_2;Y)+I(X_1;Y|X_2)+I(X_2;Y|X_1)$
  - 对应：$S_{\text{cmi}}=I(X_1;X_2;Y)$，$U_{1,\text{cmi}}=I(X_1;Y|X_2)$，$U_{2,\text{cmi}}=I(X_2;Y|X_1)$
- 实践核心挑战
  - 互信息上/下界估计稳定性与偏差
  - “最小标签保持增强”难以完美构造：$I(X,X')\approx I(X,Y)$
  - 表征层分解 vs 原始信号逐维分解：现实多为“子空间级”可辨识

## 2. 表征学习路线：FactorCL 与 CoMM
- Factorized Contrastive Learning（FACTORCL）
  - 目标（任务相关）：$S=I(X_1;X_2;Y)$，$U_i=I(X_i;Y|X_{-i})$
  - 估计器组合
    - 下界：$I_{\text{NCE}}(\cdot)$；条件下界：$I_{\text{NCE}}(\cdot|\cdot)$
    - 上界：$I_{\text{NCE-CLUB}}(\cdot)$；条件上界：$I_{\text{NCE-CLUB}}(\cdot|\cdot)$
    - 关键不等式（Theorem 6）
      - $S \ge I_{\text{NCE}}(X_1;X_2) - I_{\text{NCE-CLUB}}(X_1;X_2|X_1',X_2')$
      - $U_i \ge I_{\text{NCE}}(X_i;X_i') - I_{\text{NCE-CLUB}}(X_1;X_2) + I_{\text{NCE}}(X_1;X_2|X_1',X_2')$
  - 增强策略：独有信息增强（Unique Aug）与条件增强，避免破坏跨模态共享
  - 实践要点（摘自“usable pid method.md”）
    - 输出为 $Z_{S1},Z_{S2},Z_{U1},Z_{U2}$ 等表征子空间（非逐维信息矩阵）
    - 强依赖增强设计是否契合任务相关性
- CoMM（“先融合，再对比”）
  - 主目标：最大化融合表示 $Z',Z''$ 的互信息以捕获 $R+U_1+U_2+S$
    - 假设“最小标签保持多模态增强”：$I(X,X')\approx I(X,Y)$
  - 辅助目标：掩码单模态得到 $Z_i$，与 $Z',Z''$ 对比以捕获 $R+U_i$
  - 协同 $S$ 的“涌现”：由必须同时处理多模态的主目标驱动
  - 哲学差异（与 FactorCL）
    - CoMM：强力、通用、无条件增强；实现简洁、鲁棒
    - FactorCL：精细、有条件、需避免破坏共享；实现难度更高
- 小结
  - 二者都依赖 $I(X,X')\approx I(X,Y)$ 的理念，但增强策略与优化目标粒度不同
  - 数据与任务决定哪种更合适；可先用 CoMM 打基线，再尝试 FactorCL 精细分解

## 3. 直接 PID 计算工具与管线
- dit（离散）
  - 离散分布为前提：从样本构建 $p(Y,X_1,X_2)$
  - 多种 PID 定义：PID_MMI、BROJA、dep 等
  - 优势：理论严谨、生态完善；劣势：离散化敏感
  - 适用：低维、可稳健离散化的特征
- pidpy（连续高斯）
  - 面向高斯变量的 PID 计算，更适合神经科学特征
  - 优势：免离散化、API 简洁；注意高斯假设
- 输入准备与降维（关键）
  - 不直接用高维原始波形；需“trial 级”低维特征
  - EEG（运动想象示例）：CSP + log-variance（4–6 维）
  - fNIRS：HbO/HbR 峰值、GLM beta、时延/坡度等
  - 统一窗口、对齐、标准化；离散化或高斯化以匹配工具假设

## 4. 生理意义对齐：EEG + fNIRS
- 预期对应
  - 冗余 R：局部激活事件（ERP↑；HbO↑/HbR↓（延迟））
  - 独特 U（EEG）：频段模式（Alpha/Beta/Gamma…）
  - 独特 U（fNIRS）：代谢成本幅度（HbO/HbR 变化）
  - 协同 S：神经血管耦合延迟；神经效率（强度 vs 成本）
- 障碍
  - 原始信号是“混合物”；同一载体可承载多类信息（语义重叠 vs 数学互斥）
  - 非线性识别的可辨识性不足，往往只能到子空间层面
- 两条路线
  - Decompose then Interpret（先分解，后语义验证）
    - 线性探测：从 $z_R,z_{U\_eeg},z_{U\_fnirs},z_S$ 预测手工生理特征（P300、HbO峰等）
    - 显著图/归因：检查 $z_{U\_eeg}$ 是否关注特定频段
    - 表示相关：$z$ 维度与生理特征的相关矩阵
  - PGID（Physiologically-Guided Information Decomposition）
    - 总损失：$L=L_{\text{task}}+\lambda_{\text{ortho}}L_{\text{ortho}}+\lambda_{\text{recon}}L_{\text{recon}}+\gamma_{\text{guide}}L_{\text{guide}}$
    - 引导特征：feat_ERP_amp / feat_alpha_power / feat_hbo_peak
    - 典型正则：独立性/解耦（TC/MINE）、正交化、对抗防泄漏

## 5. 仿真数据设计（可控 R/U/S）
- 变量与独立性
  - 令 $w_1,w_2,w_s$ 两两独立，噪声独立；通道隔离：$x_1=f_1(w_1,w_s,\epsilon_1)$，$x_2=f_2(w_2,w_s,\epsilon_2)$
- 标签生成
  - $t=\rho_s\langle u_s,w_s\rangle+\rho_1\langle u_1,w_1\rangle+\rho_2\langle u_2,w_2\rangle+\xi$；$y\sim \mathrm{Bernoulli}(\sigma(g(t)))$
  - 协同项：需要时显式加入交互（如乘积/异或）并记录强度
- 自检
  - HSIC/距离相关≈0（独立性）、$I(x_1;w_2|w_s)\approx0$ 等
  - 调 $\rho_s,\rho_1,\rho_2$ 的可解释趋势检验

## 6. 增强策略清单与哲学
- 通用增强实例
  - 图像：RandomResizedCrop / ColorJitter / Flip / Blur
  - 文本：Masking / 轻微删除打乱 / 回译
  - 音频/时间序列：噪声、Time/Freq Mask、Pitch/Time Stretch
- 多模态增强哲学
  - CoMM：强力、独立、通用增强；构造困难对比任务
  - FactorCL：条件/依赖增强；避免破坏共享以近似 $Y$
- EEG/fNIRS 场景建议
  - EEG：带通抖动（轻微变动频段界）、通道 Dropout、随机时移（小幅）、幅度归一化扰动
  - fNIRS：轻微重采样/时移（近耦合时延）、低频趋势扰动、通道掩码
  - 跨模态共同增强：对齐级别的微扰，控制不破坏“共享语义”

## 7. 评估与度量
- 互信息近似与注意事项
  - 下界（InfoNCE/MINE）易偏低；上界（CLUB）易偏高；联合使用更稳健
- 分解正确性指标（表征级）
  - 共享性：$I(Z_s^{(1)};Z_s^{(2)}) \uparrow$
  - 私有性：$I(Z_{p1};X_2|Z_s)\downarrow,\ I(Z_{p2};X_1|Z_s)\downarrow$
  - 任务侧：
    - 唯一性：$I(Y;Z_{p1}|Z_s,Z_{p2}),\ I(Y;Z_{p2}|Z_s,Z_{p1})$
    - 协同性：$I(Y;[Z_{p1},Z_{p2}]|Z_s)-I(Y;Z_{p1}|Z_s)-I(Y;Z_{p2}|Z_s)>0$
- 任务端消融
  - 仅EEG、仅fNIRS、融合；融合显著优于各自之和→协同证据

## 8. 面向 EEG+fNIRS 的落地路线图（TODO）
- 基线（线性）
  - [ ] 预处理/对齐；CCA/pCCA 获取共享 $\hat{Z}_s$，残差为私有
  - [ ] 验证共享/私有性与对 $Y$ 的贡献
- 非线性与对比
  - [ ] Deep CCA / 对比双塔（学 $Z_s$）+ 正交层得 $Z_{p1},Z_{p2}$
  - [ ] PGID：加入引导损失与对抗器，做表征解耦
- 直接 PID 估计
  - [ ] 提取 trial 级特征（CSP/HbO 等），尝试 pidpy（高斯）与 dit（离散化）
  - [ ] 横向比较不同 PID 定义的一致性与稳健性
- 表征解释
  - [ ] 线性探测与表示相关；显著图验证频段/时延关注
- 工程集成
  - [ ] 以 CoMM 建立对比预训练基线
  - [ ] 在关键数据集（`data/A simultaneous EEG-fNIRS ...`）上复现与对比 FactorCL

## 9. 已知局限与开放问题
- 最优增强不可验证，仅能启发式近似
- PID 在非线性情形下的可辨识性与度量选择（MMI/BROJA/dep…）对结论的影响
- 原始空间逐维“信息分配矩阵”难以获得，现实多在表征子空间层面
