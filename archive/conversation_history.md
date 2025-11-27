接下来我会给出两篇文章，两篇文章都涉及到了部分信息分解相关的知识，并应用于多模态的研究中。他们都使用了对单个模态做增强以使得互信息项具有意义的方法，但是我希望你详细的比较他们的异同。首先，我将给出第一篇文章，阅读并进行思考和总结。

---

好的，请您提供第一篇文章。

我会仔细阅读并进行深入的思考和总结，重点关注您提到的几个关键点：信息分解的理论框架、在多模态研究中的具体应用、以及通过对单个模态进行增强来使互信息项有意义的技术细节和动机。

在我完成对第一篇文章的分析和总结后，您便可以提供第二篇文章，届时我将为您详细地比较二者在方法、目标和实现上的异同点。

请分享第一篇文章。

---

FACTORIZED CONTRASTIVE LEARNING:
Going Beyond Multi-view Redundancy
Paul Pu Liang1∗
, Zihao Deng2∗
, Martin Q. Ma1∗
James Zou3
, Louis-Philippe Morency1
, Ruslan Salakhutdinov1
1Carnegie Mellon University, 2University of Pennsylvania, 3Stanford University
pliang@cs.cmu.edu,zihaoden@cs.cmu.edu,qianlim@cs.cmu.edu
Abstract
In a wide range of multimodal tasks, contrastive learning has become a particularly
appealing approach since it can successfully learn representations from abundant
unlabeled data with only pairing information (e.g., image-caption or video-audio
pairs). Underpinning these approaches is the assumption of multi-view redundancy
- that shared information between modalities is necessary and sufficient for down￾stream tasks. However, in many real-world settings, task-relevant information is
also contained in modality-unique regions: information that is only present in one
modality but still relevant to the task. How can we learn self-supervised multimodal
representations to capture both shared and unique information relevant to down￾stream tasks? This paper proposes FACTORCL, a new multimodal representation
learning method to go beyond multi-view redundancy. FACTORCL is built from
three new contributions: (1) factorizing task-relevant information into shared and
unique representations, (2) capturing task-relevant information via maximizing MI
lower bounds and removing task-irrelevant information via minimizing MI upper
bounds, and (3) multimodal data augmentations to approximate task relevance with￾out labels. On large-scale real-world datasets, FACTORCL captures both shared
and unique information and achieves state-of-the-art results on six benchmarks.
1 Introduction
Learning representations from different modalities is a central paradigm in machine learning [48].
Today, a popular learning method is to first pre-train general representations on unlabeled multimodal
data before fine-tuning on task-specific labels [10, 40, 47, 48, 50]. These current multimodal pre￾training approaches have largely been inherited from prior work in multi-view learning [13, 58]
that exploit a critical assumption of multi-view redundancy: the property that shared information
between modalities is almost exactly what is relevant for downstream tasks [70, 73, 76]. When this
assumption holds, approaches based on contrastive pre-training to capture shared information [13, 41,
61, 72], followed by fine-tuning to keep task-relevant shared information [76], have seen successful
applications in learning from images and captions [61], video and audio [3], speech and transcribed
text [58], and instructions and actions [21]. However, our paper studies two fundamental limitations
in the application of contrastive learning (CL) to broader real-world multimodal settings (see Figure 1
for a visual depiction and experimental results showing the performance drop of CL):
1. Low shared information relevant to tasks: There exists a wide range of multimodal tasks
involving small amounts of shared information, such as between cartoon images and figurative
captions (i.e., not literal but metaphoric or idiomatic descriptions of the images [52, 88]). In these
situations, standard multimodal CL will only receive a small percentage of information from the
learned representations and struggle to learn the desired task-relevant information.
2. High unique information relevant to tasks: Many real-world modalities can provide unique
information not present in other modalities. Examples include healthcare with medical sensors or
robotics with force sensors [45, 49]. Standard CL will discard task-relevant unique information,
leading to poor downstream performance.
∗
First three authors contributed equally.
37th Conference on Neural Information Processing Systems (NeurIPS 2023).
Figure 1: Left: We define S = I(X1; X2; Y ) as task-relevant shared information and U1 = I(X1; Y ∣X2),
U2 = I(X2; Y ∣X1) as task-relevant unique information. Right: On controllable datasets with varying ratios
of S, U1, and U2, standard CL captures S but struggles when there is more U1 and U2. Our FACTORCL
approach maintains best performance, whereas SimCLR [13] and SupCon [41] see performance drops as unique
information increases, and Cross+Self [33, 36, 44, 89] recovers in fully unique settings but suffers at other ratios.
In light of these limitations, how can we design suitable multimodal learning objectives that work
beyond multi-view redundancy? In this paper, starting from the first principles in information
theory, we provide formal definitions of shared and unique information via conditional mutual
information and propose an approach, FACTORIZED CONTRASTIVE LEARNING (FACTORCL for
short), to learn these multimodal representations beyond multi-view redundancy using three key
ideas. The first idea is to explicitly factorize shared and unique representations. The second idea is to
capture task-relevant information via maximizing lower bounds on MI and remove task-irrelevant
information via minimizing upper bounds on MI, resulting in representations with sufficient and
necessary information content. Finally, a notion of task relevance without explicit labels in the self￾supervised setting is achieved by leveraging multimodal augmentations. Experimentally, we evaluate
the effectiveness of FACTORCL on a suite of synthetic datasets and large-scale real-world multimodal
benchmarks involving images and figurative language [88], prediction of human sentiment [91],
emotions [93], humor [27], and sarcasm [12], as well as patient disease and mortality prediction from
health indicators and sensor readings [38], achieving new state-of-the-art performance on six datasets.
Overall, we summarize our key technical contributions here:
1. A new analysis of contrastive learning performance showing that standard multimodal CL fails to
capture task-relevant unique information under low shared or high unique information cases.
2. A new contrastive learning algorithm called FACTORCL:
(a) FACTORCL factorizes task-relevant information into shared and unique information, expand￾ing contrastive learning to better handle low shared or high unique information.
(b) FACTORCL optimizes shared and unique information separately, by removing task-irrelevant
information via MI upper bounds and capturing task-relevant information via lower bounds,
yielding optimal task-relevant representations.
(c) FACTORCL leverages multimodal augmentations to approximate task-relevant information,
enabling self-supervised learning from our proposed FACTORCL.
2 Analysis of Multi-view Contrastive Learning
We begin by formalizing definitions of four types of information: shared, unique, task-relevant, and
task-irrelevant information in multimodal data. To formalize the learning setting, we assume there
exist two modalities expressed as random variables X1 and X2 with outcomes x1 and x2, and a task
with the random variable Y and outcome y. We denote X−i as the other modality where appropriate.
Shared and unique information: We formalize shared and unique information by decomposing the
total multimodal information I(X1, X2; Y ) into three conditional mutual information (MI) terms:
I(X1, X2; Y ) = I
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶(X1; X2; Y )
S = shared
+ I(X1; Y ∣X2)
U1
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶ = uniqueness in X1
+ I(X2; Y ∣X1)
U2
´¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¸¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¹¶ = uniqueness in X2
, (1)
where I(X1, X2; Y ) = ∫ p(x1, x2, y) log p(x1,x2,y)
p(x1,x2)p(y)
dx1dx2dy is the total MI between the
joint random variable X1, X2 and the task Y , S = I(X1; X2; Y ) = I(X1; X2) −
I(X1; X2∣Y ) = ∫ p(x1, x2) log p(x1,x2)
p(x1)p(x2)
dx1dx2 − I(X1; X2∣Y ) is the task-relevant shared in-
2
formation, I(X1; X2∣Y ) = ∫ p(x1, x2, y) log p(x1,x2∣y)
p(x1∣y)p(x2∣y)
dx1dx2dy is the task-irrelevant shared
information, and U1 = I(X1; Y ∣X2), U2 = I(X2; Y ∣X1) denote unique task-relevant information.
Limitations of CL: Current approaches for CL maximize mutual information I(X1; X2) (and
subsequently task-relevant shared information I(X1; X2; Y ) during supervised fine-tuning), without
modeling unique information. These methods generally learn a pair of representations [73, 76],
Z1 = arg max
Z1∶=fθ(X1)
I(Z1; X2), Z2 = arg max
Z2∶=fθ(X2)
I(X1;Z2). (2)
For example, Z1 could encode images X1 and Z2 encodes text X2 via maximizing a lower bound on
I(X1; X2) using the NCE objective [58]. The NCE objective falls into a broader class of contrastive
learning methods [13, 15, 28, 41, 61] that model the ratio between joint densities p(x1, x2) and
product of marginal densities p(x1)p(x2) using positive and negative samples [57, 59, 60, 79, 84] or
probabilistic classifiers [55, 77]. It has been shown that contrastive learning works well under the
assumption of multi-view redundancy [4, 31, 70, 71, 76]:
Definition 1. (Multi-view redundancy) ∃ϵ > 0 such that I(X1; Y ∣X2) ≤ ϵ and I(X2; Y ∣X1) ≤ ϵ.
In other words, the task-relevant information in data is mostly shared across both views and the
unique information is at most a small ϵ. From a representation perspective, Tian et al. [72] further
introduces the assumption that the optimal representation is minimal and sufficient, where all learned
task-relevant information is shared information: I(Z1; Y ∣X2) = I(Z2; Y ∣X1) = 0. While the multi￾view redundancy is certainly true for particular types of multimodal distributions, it crucially ignores
settings that display multi-view non-redundancy and unique information can be important, such as
when health indicators, medical sensors, and robotic visual or force sensors each provide unique
information not present in other modalities [45, 49].
Definition 2. (Multi-view non-redundancy) ∃ϵ > 0 such that I(X1; Y ∣X2) > ϵ or I(X2; Y ∣X1) > ϵ.
Under multi-view non-redundancy, we show that standard CL only receives a weak training signal
since it can only maximize a lower bound on shared information I(X1; X2), and struggles to learn
task-relevant unique information. We formalize this intuition with the following statement:
Theorem 1. (Suboptimality of standard CL) When there is multi-view non-redundancy as in Definition
2, given optimal representations {Z1, Z2} that satisfy Eq.(2 and I(Z1; Y ∣X2) = I(Z2; Y ∣X1) =
0 [72], we have that
I(Z1, Z2; Y ) = I(X1, X2; Y ) − I(X1; Y ∣X2) − I(X2; Y ∣X1) = I(X1; X2) − I(X1; X2∣Y ) < I(X1, X2; Y ).
(3)
Correspondingly, the Bayes error rate Pe(Z1, Z2) ∶= 1 − Ep(z1,z2) [maxy∈Y P (Yˆ = y ∣ z1, z2)] of
contrastive representations {Z1, Z2} for a downstream task Y is given by:
Pe ≤ 1 − exp [I(X1, X2; Y ) − I(X1; Y ∣X2) − I(X2; Y ∣X1) − H(Y )] (4)
= 1 − exp [I(X1; X2; Y ) − H(Y )] (5)
We include proofs and a detailed discussion of the assumptions in Appendix B. Based on
Eq.(3), I(Z1, Z2; Y ) decreases with higher task-relevant unique information I(X1; Y ∣X2) and
I(X2; Y ∣X1); we call this the difference I(X1, X2; Y ) − I(Z1, Z2; Y ) the uniqueness gap. The
uniqueness gap measures the loss in task-relevant information between the input and encoded rep￾resentation: as task-relevant unique information grows, the uniqueness gap increases. In addition,
I(Z1, Z2; Y ) also drops with lower I(X1; X2) (i.e., two modalities sharing little information to
begin with), or with higher I(X1; X2∣Y ) (i.e., when the shared information is mostly task-irrelevant).
Similarly, in Eq.(5), the Bayes error rate of using {Z1, Z2} for prediction is directly related to the
task-relevant information in {Z1, Z2}: error on the downstream task increases with higher unique
information and lower shared information.
3 FACTORIZED CONTRASTIVE LEARNING
We now present a suite of new CL objectives that alleviate the challenges above and work at all ranges
of shared and unique information. At a high level, we aim to learn a set of factorized representations
ZS1
, ZS2
, ZU1
, ZU2
representing task-relevant information in X1 shared with X2, in X2 shared with
X1, unique to X1, and unique to X2 respectively. As common in practice [61, 72], we define
3
Figure 2: FACTORCL: We propose a self-supervised CL method to learn factorized representations ZS1
, ZS2
,
ZU1
, and ZU2
to capture task-relevant information shared in both X1 and X2, unique to X1, and unique to X2.
By starting with information-theoretic first principles of shared and unique information, we design contrastive
estimators to both capture task-relevant and remove task-irrelevant information, where a notion of task-relevance
without explicit labels is afforded by a new definition of multimodal augmentations X1
′
, X2
′
. Lower bounds are
in green and upper bounds are in red.
neural networks fθ with trainable parameters θ to extract representations from inputs X1 and X2.
Learning these parameters requires optimizing differentiable and scalable training objectives to
capture task-relevant shared and unique information (see overview in Figure 2):
ZS1 = arg max
Z1=fθ(X1)
I(Z1; X2; Y ), ZS2 = arg max
Z2=fθ(X2)
I(Z2; X1; Y ), (6)
ZU1 = arg max
Z1=fθ(X1)
I(Z1; Y ∣X2), ZU2 = arg max
Z2=fθ(X2)
I(Z2; Y ∣X1). (7)
where I(Z1; X2; Y ) = I(Z1; X2) − I(Z1; X2∣Y ) is the shared information and I(Z2; X1; Y ) =
I(Z2; X2) − I(Z2; X1∣Y ) is the unique information. One important characteristic of our framework
is that when unique information is zero: I(X1; Y ∣X2) = 0 and I(X2; Y ∣X1) = 0, or all shared
information is task-relevant: I(X1; X2; Y ) = I(X1; X2), our framework recovers standard CL as in
Eq.(2). However, as we have previously indicated and will show empirically, these assumptions can
easily be violated, and our framework enlarges Eq.(2) to cases where unique information is present.
The learned Zs can then be used as input to a linear classifier and fine-tuned to predict the label for
multimodal classification or retrieval tasks. However, the shared and unique MI terms above are
often intractable in practice. In the next section, we will build up our method step by step, eventually
showing that each term in Eqs.(6- 7) can be approximated as follows:
S = I(X1; X2; Y ) ≥ INCE(X1; X2) − INCE-CLUB(X1; X2∣X1
′
, X2
′
) (8)
Ui = I(Xi
; Y ∣X−i) ≥ INCE(Xi
; Xi
′
) − INCE-CLUB(X1; X2) + INCE(X1; X2∣X1
′
, X2
′
) (9)
where INCE and INCE-CLUB are scalable contrastive estimators (Section 3.1) and X1
′
, X2
′
are suitable
data augmentations (Section 3.2) on each modality. Overall, these equations can be interpreted as
both positive and negative signals to learn representations for S and U. For shared information
S, the estimator maximizes task-relevant shared information via INCE(X1; X2) while removing
task-irrelevant shared information via a novel upper bound −INCE-CLUB(X1; X2∣X1
′
, X2
′
). For unique
information Ui
, we capture task-relevant uniqueness via +INCE(Xi
; Xi
′
) while non-unique informa￾tion is removed via −(INCE-CLUB(X1; X2) − INCE(X1; X2∣X1
′
, X2
′
)). In the following sections, we
derive this final objective step-by-step: (1) approximating the MI objectives in S and U with CL
estimators, (2) relaxing the dependence on labels Y with self-supervised data augmentations, finally
(3) discussing overall training and implementation details of end-to-end self-supervised learning.
3.1 Supervised FACTORCL with shared and unique information
To capture shared and unique information via an objective function, we will need to maximize lower
bounds for all terms with a positive sign in Eq.(8) and (9) (I (X1; X2) , I (Xi
; Y ) , I (X1; X2∣Y ))
and minimize upper bounds for all terms with a negative sign (I (X1; X2) , I (X1; X2∣Y )). Our first
theorem derives general lower and upper bounds for MI terms as variants of contrastive estimation:
4
0
2
4
6
8
10
12
14
16
Gaussian, dim=20
True MI
NCE
CLUB
NCE CLUB
Gaussian, dim=50 Gaussian, dim=100 Gaussian, dim=200
Figure 3: Estimated INCE lower bound [58] and our proposed upper bound INCE-CLUB on sample distributions
with changing mutual information: our upper bound is tighter, more accurate, and more stable than ICLUB upper
bound [16], and also comes for ‘free’ via jointly estimating both lower and upper bounds simultaneously. We
find that as dimension increases, the ICLUB estimator collapses to zero and no longer tracks true MI.
Theorem 2. (Contrastive estimators for I(X1; X2)) Defining the NCE and NCE-CLUB estimators,
INCE(X1; X2) = Ex1,x
+
2∼p(x1,x2)
x
−
2∼p(x2)
[log exp f(x1, x+
2
)
∑k
exp f(x1, x−
2
)
] (10)
INCE-CLUB(X1; X2) = Ex1,x+
2
∼p(x1,x2) [f
∗
(x1, x+
2
)] − Ex1∼p(x1)
x
−
2∼p(x2)
[f
∗
(x1, x−
2
)] (11)
where f
∗
(x1, x2) is the optimal critic from INCE plugged into the ICLUB objective [16]. We call the
proposed plug-in objective Eq.(11) INCE-CLUB, and obtain lower and upper bounds on I(X1; X2):
INCE(X1; X2) ≤ I(X1; X2) ≤ INCE-CLUB(X1; X2). (12)
Proof. The lower bound INCE(X1; X2) ≤ I(X1; X2) follows from Oord et al. [58]: optimizing the
objective leads to an optimal critic [60] f
∗ = log p(x1∣x2) + c(x1), with a deterministic function
c(⋅). Plugging optimal critic f
∗
into INCE-CLUB(X1; X2) cancels out the c(x1) term and yields
INCE-CLUB(X1; X2) and I(X1; X2) ≤ INCE-CLUB. We include a detailed proof in Appendix C.1.
INCE-CLUB(X1; X2) gives a desired upper bound of I(X1; X2) “for free” while avoiding separately
optimizing lower bound and upper bounds. In Figure 3, we show these two bounds in practice across
two Gaussian distributions X1 and X2 with varying amounts of MI I(X1; X2). We use the second
formulation of ICLUB [16], which assumes p(x1∣x2) to be unknown. Our upper bound is empirically
tighter (see Figure 3) and comes for “free” via jointly maximizing the lower bound INCE. These
lower and upper bounds can be seen as new contrastive objectives over positive and negative (x1, x2)
pairs, enabling a close integration with existing pre-training paradigms. Finally, we can similarly
obtain bounds for the conditional MI INCE(X1; X2∣Y ) ≤ I(X1; X2∣Y ) ≤ INCE-CLUB(X1; X2∣Y ):
INCE(X1; X2∣Y ) = Ep(y)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x
+
2
∼p(x1,x2∣y)
x
−
2
∼p(x2∣y)
[log exp f(x1, x
+
2 , y)
∑k
exp f(x1, x−
2
, y)
]
⎤
⎥
⎥
⎥
⎥
⎦
(13)
INCE-CLUB(X1; X2∣Y ) = Ep(y)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x+
2
∼p(x1,x2∣y)
[f
∗
(x1, x
+
2 , y)] − Ex1∼p(x1∣y)
x
−
2
∼p(x2∣y)
[f
∗
(x1, x
−
2 , y)]
⎤
⎥
⎥
⎥
⎥
⎦
(14)
These two bounds result in conditional CL objectives [51, 74, 78] - they differ critically from standard
CL methods since they capture task-irrelevant shared information that remains between X1 and
X2 after observing Y . This task-irrelevant shared information is removed by minimizing its upper
bound. Note that f(x1, x2, y) here denotes a different function from f(x1, x2) in Eq.(10), as the
general forms are different (taking in x1, x2 versus x1, x2, y). f(x1, x2, y) can be implemented in
different ways, e.g., g([x1, y])T h(x2) where g(), h() are trainable encoders and [x1, y] denotes
concatenation [69].
3.2 Self-supervised FACTORCL via multimodal augmentations
The derivations above bring about supervised CL objectives with access to Y [41]. For unsupervised
CL [58, 72], we derive similar objectives without access to Y by leveraging semantic augmentations
on each modality. Denote X′
as some augmentation of X (e.g., rotating, shifting, or cropping). Under
5
Mutual Information
the optimal augmentation assumption from Tian et al. [72] (restated below), replacing Y with X′
in
our formulations enables learning of task-relevant information without access to labels:
Definition 3. (Optimal unimodal augmentation) [72] X1
′
is an optimal unimodal augmentation for
X1 when I(X; X′
) = I(X; Y ), which implies that the only information shared between X and X′
is
task-relevant with no irrelevant noise.
This assumption is satisfied when all information shared between X and X′
is task-relevant, which
implies that the augmentation keeps task-relevant information constant while changing task-irrelevant
information. In the case of image classification, task-relevant information is the object in the
picture, while task-irrelevant information is the background. By performing two separate unimodal
augmentations giving X1
′
and X2
′
, we can substitute contrastive estimators in Eqs.(13) and (14), by
replacing I(Xi
; Y ) terms with I(Xi
; Xi
′
) and replacing I(X1; X2∣Y ) terms with I(X1; X2∣X1
′
, X2
′
):
INCE(X1; X2∣X1
′
, X2
′
) = Ep(x′
1
,x′
2
)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x
+
2
∼p(x1,x2∣x
′
1
,x
′
2
)
x
−
2
∼p(x2∣x
′
1
,x
′
2
)
[log exp f(x1, x
+
2 , x
′
1, x
′
2)
∑k
exp f(x1, x−
2
, x′
1
, x′
2
)
]
⎤
⎥
⎥
⎥
⎥
⎦
(15)
INCE-CLUB(X1; X2∣X1
′
, X2
′
) = Ep(x′
1
,x′
2
)[Ex1,x+
2
∼p(x1,x2∣x′
1
,x′
2
)
[f
∗
(x1, x
+
2 , x
′
1, x
′
2)]
− Ex1∼p(x1∣x
′
1
,x
′
2
)
x
−
2
∼p(x2∣x
′
1
,x
′
2
)
[f
∗
(x1, x
−
2 , x
′
1, x
′
2)]] (16)
The objectives can be seen as conditional contrastive learning on augmentations (X1
′
, X2
′
). Here
again f(x1, x2, x′
1
, x′
2
) is different from the critics in Eqs.(13 because of the different general forms.
We implement f() here as g([x1, x′
1
])T h([x2, x′
2
]) where g(), h() are trainable encoders specific
for each modality and [x1, x′
1
] denotes concatenation. This concatenation is justified by the CMI
estimators in Sordoni et al. [69], who show that concatenating the conditioning variable with the
input in the critic f(x1, x2, x′
1
, x′
2
) yields a Conditional InfoNCE estimator (Eq.(15)) that is a lower
bound for CMI. However, the exact Conditional InfoNCE estimator learns a different conditional
distribution p(x1, x2∣x
′
1
, x′
2
) for each augmented pair x
′
1
, x′
2
, which can be prohibitively expensive.
We could approximate this by creating multiple augmentations of a single paired x1, x2. Our code
uses one augmented pair x
′
1
, x′
2
for each x1, x2 but could be extended to multiple pairs, and we find
this simple approach yields consistent CMI lower and upper bounds that are empirically comparable
to existing CMI estimators [55, 69]. We include full comparisons and implementation details in
Appendix D.1, and in Appendix C.2 we discuss an alternative interpretation based on viewing CL as
kernel learning which permits using conditional kernel estimation for our objectives.
Although we find this method to work well in practice, a more careful analysis reveals that 2 separate
unimodal augmentations X1
′
and X2
′
each satisfying I(Xi
; Xi
′
) = I(Xi
; Y ) do not together satisfy
I(X1; X2∣Y ) = I(X1; X2∣X1
′
, X2
′
) needed for the substitution in Eqs.(15) and (16) to hold with
equality. To satisfy this property exactly, we define optimal multimodal augmentations:
Definition 4. (Optimal multimodal augmentation) X1
′
and X2
′
are optimal multimodal augmentation
for X1 and X2 when I(X1, X2; X1
′
, X2
′
) = I(X1, X2; Y ), which implies that the only information
shared between X1, X2 and X1
′
, X2
′
is task-relevant with no irrelevant noise.
We satisfy I(X1, X2; X1
′
, X2
′
) = I(X1, X2; Y ) using two steps:
Unimodal aug: X1
′
s.t. I(X1; X1
′
) = I(X1; Y ), (17)
Unique aug: X2
′
s.t. I(X2; X2
′
∣X1) = I(X2; Y ∣X1). (18)
We call the second step unique augmentation: after observing X1, we create augmented X2
′
from
X2 to keep task-relevant information not already in X1. To empirically satisfy optimal multimodal
augmentations, we avoid augmentations in one modality that will remove or strongly destroy infor￾mation shared with the other modality. For example, in image captioning, we should avoid image
augmentations such as cropping that destroy information from the caption (e.g., cropping object parts
referred to by the caption), and instead, only augment images via flipping or color jittering which
retains all caption information. Figure 4 shows an example of unique augmentation that satisfies these
conditions. In our experiments, we will show that our augmentations consistently perform better than
standard augmentations (Table 3), suggesting that approximately satisfying Eqs.(17) and (18) can be
empirically sufficient, which is simple and straightforward to implement on real-world datasets.
3.3 Overall method and implementation
6
Algorithm 1 Standard multimodal CL.
Require: Multimodal dataset {X1, X2}.
Initialize networks f(⋅).
while not converged do
for sampled batch {x1,x2} do
Estimate INCE(X1; X2) from Eq. 10
L = −INCE(X1; X2)
Update f(⋅) to minimize L
end for
end while
return f(⋅)
Algorithm 2 FACTORCL.
Require: Multimodal dataset {X1, X2}.
Initialize networks f(⋅).
while not converged do
for sampled batch {x1,x2} do
x
′
1 ← Augment(x1)
x
′
2 ← Unique-Augment(x2∣x1)
Plug x
′
1
and x
′
2
into Eq. 15 and 16
Estimate S, U1, U2 from Eq. 8 and 9
L = −(S + U1 + U2)
Update f(⋅) to minimize L
end for
end while
return f(⋅)
Figure 4: Standard vs. unique augmentations for
the figurative language [88] dataset. After aug￾menting text modality X1 independently (same for
both augmentation types), we illustrate their differ￾ences for image augmentation: unique augmenta￾tion on images should avoid removing information
referred to by X1 (the text). The text mentions that
the car is fast so unique augmentation for images
should not remove the highway pixels of the image
which can suggest the car is fast.
The final algorithm sketch is in Algorithm 2, which
we compare against standard CL in Algorithm 1. It
can be shown that FACTORCL learns all the task￾relevant information from both modalities:
Theorem 3. (Optimality of FACTORCL)
If ZS1
, ZS2
, ZU1
, ZU2 perfectly maximize
Eqs.(6-7) and the estimations in Eqs.(8)
and (9) are tight, we obtain I(X1, X2; Y ) =
I(ZS1
;ZS2
; Y ) + I(ZU1
; Y ∣ZS2
) + I(ZU2
; Y ∣ZS1
),
suggesting that FACTORCL learns both shared and
unique task-relevant information.
We include the full proof in Appendix C.3. In prac￾tice, while we do not expect perfect estimation of MI
quantities and maximization with respect to MI ob￾jectives, we include implementation details regarding
architectures and contrastive objectives that improve
empirical performance in Appendix D.1.
Complexity: Compared to heuristic combinations of
cross-modal and single-modality CL [33, 36, 44, 64,
81, 85, 89], our approach does not significantly in￾crease complexity: (1) upper bounds on MI can be es￾timated “for free” by directly plugging in the optimal
critic from INCE, (2) removal of task-irrelevant infor￾mation via I(X1; X2∣X1
′
, X2
′
) shares encoders with
INCE, and (3) separate unimodal augmentations per￾form empirically well. We describe some extensions
of other self-supervised methods in Appendix C.4.
4 Experiments
We run comprehensive experiments on a suite of synthetic and large-scale real-world datasets with
varying requirements of shared and unique task-relevant information, comparing our FACTORCL
method to key baselines:
1. SimCLR [13]: the straightforward method of cross-modal (X1, X2) contrastive learning.
2. Cross+Self [33, 36, 44, 64, 85, 89]: captures a range of methods combining cross-modal (X1, X2)
CL with additional unimodal (Xi
, Xi
′
) CL objectives. This category also includes other ways of
preserving unique information, such as through (variational) autoencoder reconstructions [81].
3. Cross+Self+Fact [86, 89]: A factorized extension of Cross+Self, which is approximately done in
prior work that adds separate (typically pre-trained) unimodal encoders for each modality.
4. SupCon [41], which learns I(X1; X2∣Y ) using CL conditioned on Y from labeled data.
We also carefully ablate each component of our method and investigate factors, including training
data size and choice of augmentations. The intermediate ablations that emerge include:
1. FACTORCL-SUP: The supervised CL version which uses labels Y in Eqs.(13) and (14).
7
Table 1: We probe whether contrastive representations learned by classic CL methods and FACTORCL contain
shared ws or unique w1, w2 information. FACTORCL captures the most unique information.
Model SimCLR Cross+self SupCon FACTORCL
Representations Z1 Z2 Z1 Z2 Z1 Z2 ZU1 ZU2 ZS1 ZS2
I(Z; w1) 4.45 0.16 4.39 0.14 5.17 0.19 7.83 0.03 6.25 0.04
I(Z; w2) 0.17 3.92 0.13 4.26 0.23 5.17 0.06 7.17 0.05 5.79
I(Z; ws) 12.61 12.06 11.30 11.47 7.48 7.17 9.47 9.89 10.13 9.40
2. FACTORCL-SSL: The fully self-supervised version of our approach replacing Y with multimodal
augmentations X1
′
and X2
′
to approximate the task.
3. OurCL-SUP: FACTORCL-SUP but removing the factorization so only two features Z1 is optimized
for both I(X1; X2; Y ) and I(X1; Y ∣X2), Z2 optimized for both I(X1; X2; Y ) and I(X2; Y ∣X1).
4. OurCL-SSL: FACTORCL-SSL but also removing the factorization in the self-supervised setting.
The formulation of each ablation and implementation can be found in Appendix D.1.
4.1 Controlled experiments on synthetic datasets
Synthetic data generation: We begin by generating data with controllable ratios of task-relevant
shared and unique information. Starting with a set of latent vectors w1, w2, ws ∼ N (0d, Σ
2
d
), d = 50
representing information unique to
[w1, ws] is transformed into high-dimensional
X1, X2 and common to both respectively, the concatenated vector
x1 using a fixed transformation T1 and likewise
[w2, ws] to x2 via T2. The label y is generated as a function (with nonlinearity and noise) of varying
ratios of ws, w1, and w2 to represent shared and unique task-relevant information.
Results: In Figure 1, we show our main result on synthetic data comparing FACTORCL with existing
CL baselines. FACTORCL consistently maintains the best performance, whereas SimCLR [13] and
SupCon [41] see performance drops as unique information increases. Cross+Self [33, 36, 44, 89]
recovers in fully unique settings (x-axis= 1.0) but suffers at other ratios.
Representation probing information: We run a probing experiment to compute how well different
contrastive representations capture shared and unique information. In Table 1, for the Zi’s learned by
each method, we approximately compute I(Zi
; w1), I(Zi
; w2), and I(Zi
; ws) with respect to ground
truth generative variables ws, w1, and w2. As expected, existing methods such as SimCLR capture
smaller amounts of unique information (roughly 4 bits in I(Zi
; w1) and I(Zi
; w2)), focusing instead
on learning I(Zi
; ws) (12 bits). Cross+self captures slightly larger I(Zi
; w2) = 4.26, and SupCon
with labeled data captures up to 5 bits of unique information. Our FACTORCL approach captures
7 bits of unique information and maintains 10 bits of shared information, with total information
captured higher than the other approaches. Furthermore, {ZS1
, ZS2
} capture more information about
ws, ZU1
about w1, and ZU2
about w2, indicating that factorization in our approach is successful.
4.2 Self-supervised multimodal learning with low redundancy and high uniqueness
Multimodal fusion datasets: We use a large collection of real-world datasets provided in Multi￾Bench [45], where we expect varying ratios of shared and unique information important for the task,
to compare FACTORCL with other CL baselines:
1. MIMIC [38]: mortality and disease prediction from 36, 212 medical records (tabular patient data
and medical time-series sensors from ICU).
2. MOSEI [93]: multimodal sentiment and emotion benchmark with 23, 000 monologue videos.
3. MOSI [91]: multimodal sentiment analysis from 2, 199 YouTube videos.
4. UR-FUNNY [27]: a dataset of humor detection from more than 16, 000 TED talk videos.
5. MUSTARD [12]: a corpus of 690 videos for research in sarcasm detection from TV shows.
6. IRFL [88]: 6, 697 matching images and figurative captions (rather than literal captions).
Together, these datasets cover seven different modalities from the healthcare, affective computing,
and multimedia research areas and total more than 84, 000 data points. For MIMIC with tabular and
medical sensor inputs, we train self-supervised CL models on top of raw modality inputs. For IRFL
with image and caption inputs, we start with a pretrained CLIP model [61] and perform continued
pre-training to update CLIP weights with our FACTORCL objectives, before linear classifier testing.
For the remaining four video datasets, we train self-supervised CL models starting from standard
pre-extracted text, video, and audio features [45]. Please refer to Appendix D.2 for experimental
details. We release our code and models at https://github.com/pliang279/FactorCL.
8
Table 2: Results on MultiBench [45] datasets with varying shared and unique information: FACTORCL achieves
strong results vs self-supervised (top 5 rows) and supervised (bottom 3 rows) baselines that do not have unique
representations, factorization, upper-bounds to remove irrelevant information, and multimodal augmentations.
Model (X1; X2) (Xi; Xi
′
) (X1; X2∣Y ) (X2
′′) Fact MIMIC MOSEI MOSI UR-FUNNY MUSTARD
SimCLR [13] ✓ ✗ ✗ ✗ ✗ 66.67% 71.03% 46.21% 50.09% 53.48%
Cross+Self [81] ✓ ✓ ✗ ✗ ✗ 65.20% 71.04% 46.92% 56.52% 53.91%
Cross+Self+Fact [89] ✓ ✓ ✗ ✗ ✓ 65.49% 71.07% 52.37% 59.91% 53.91%
OurCL-SSL ✓ ✓ ✓ ✓ ✗ 65.22% 71.16% 48.98% 58.79% 53.98%
FACTORCL-SSL ✓ ✓ ✓ ✓ ✓ 67.34% 74.88% 52.91% 60.50% 55.80%
SupCon [41] ✗ ✗ ✓ ✗ ✗ 67.37% 72.71% 47.23% 50.98% 52.75%
OurCL-SUP ✓ ✓ ✓ ✗ ✗ 68.16% 71.15% 65.32% 58.32% 65.05%
FACTORCL-SUP ✓ ✓ ✓ ✗ ✓ 76.79% 77.34% 70.69% 63.52% 69.86%
Multimodal fusion results: From Table 2, FACTORCL significantly outperforms the baselines that
do not capture both shared and unique information in both supervised and self-supervised settings,
particularly on MUSTARD (where unique information expresses sarcasm, such as sardonic facial
expressions or ironic tone of voice), and on MIMIC (with unique health indicators and sensor
readings). In Table 3, we also show that FACTORCL substantially improves the state-of-the-art in
classifying images and figurative captions which are not literally descriptive of the image on IRFL,
outperforming zero-shot and fine-tuned CLIP [61] as well as continued pre-training baselines on top
of CLIP.
Modeling ablations: In Table 2, we also carefully ablate each component in our method and indicate
either existing baselines or newly-run ablation models.
1. Factorized representations: In comparing FACTORCL-SSL with OurCL-SSL, and also FAC￾TORCL-SUP with OurCL-SUP, we find that factorization is critical: without it, performance drops
on average 6.1%, with performance drop as high as 8.6% for MIMIC.
2. Information removal via upper bound: By comparing FACTORCL with SimCLR, Cross+Self,
and Cross+Self+Fact, and SupCon that only seek to capture task-relevant information via con￾trastive lower bounds on MI, we find that separately modeling the task-relevant information
(to be captured) and task-irrelevant information (to be removed) is helpful. Without removing
task-irrelevant information via the upper-bound objective, performance drops on average 13.6%,
with performance drops as high as 23.5% for the MOSI dataset. We also found that training
was more difficult without this objective, which is expected due to overwhelming superfluous
information from the dataset [93].
3. Multimodal augmentations: Finally, we investigate the differences between separate uni￾modal augmentations (FACTORCL-IndAug in Table 3) versus a joint multimodal augmentation
(FACTORCL-SSL) on the IRFL dataset. We choose this dataset since its images and captions are
the easiest to visualize (see Figure 4 for augmentations from both strategies). In the self-supervised
setting, we find that multimodal augmentations achieve 95% performance, higher than the 92%
for separate unimodal augmentations, and both outperform baselines SimCLR and Cross+Self.
Table 3: Continued pre-training on CLIP
with our
sifying images and figurative language.
FACTORCL objectives on clas￾Task IRFL
Zero-shot CLIP [61] 89.15%
SimCLR [13] 91.57%
Cross+Self [81, 89] 95.18%
FACTORCL-IndAug 92.77%
FACTORCL-SSL 95.18%
Fine-tuned CLIP [61] 96.39%
SupCon [41] 89.16%
FACTORCL-SUP 98.80%
Ablations on S, U1 and U2: In Table 4, we also test FAC￾TOR
{ZS1
CL
, ZS
when training linear classifiers on top of only shared
2
} and unique ZU1
, ZU2
separately. We call these
models FACTORCL-S, FACTORCL-U1, and FACTORCL-U2.
Immediately, we observe that performance drops as compared
to the full FACTORCL model, indicating that both shared and
unique information are critical in real-world multimodal tasks.
As expected, the best-performing submodel is the one that
captures the region with the largest amount of task-relevant
information: MOSEI and MOSI are known to include a lot
of redundancy and unique information since language is very
important for detecting sentiment [93], so FACTORCL-S and
FACTORCL-U2 perform best. For sarcasm detection on MUS￾TARD, video information is most important with FACTORCL￾U1 performing best (59.4%), and ablation models are also the furthest away from full multimodal
performance (69.9%). This is aligned with intuition where sarcasm is expressed through tone of voice
and visual gestures (high U1), as well as from contradictions between language and video (higher
multimodal performance).
9
Table 4: We ablate using only shared representations {ZS1
, ZS2 }, unique representation ZU1
, and ZU2
separately
for prediction. Both shared and unique information are critical in real-world multimodal tasks.
Model MIMIC MOSEI MOSI UR-FUNNY MUSTARD
FACTORCL-S 63.77% 77.17% 70.12% 63.42% 57.25%
FACTORCL-U1 55.90% 77.06% 70.11% 62.00% 59.42%
FACTORCL-U2 69.08% 71.01% 52.33% 54.35% 53.62%
FACTORCL-SUP 76.79% 77.34% 70.69% 63.52% 69.86%
Additional results: In Appendix D.3, we also verify FACTORCL in settings with abundant shared
information, where we expect to recover the same performance as standard CL [13, 58, 72].
5 Related Work
Contrastive learning is a successful self-supervised learning paradigm for computer vision [11, 13,
14, 25, 28, 58], natural language [24, 54, 56], speech [5, 58, 63], and multimodal tasks [1, 37, 61]. Its
foundational underpinnings are inspired by work in multiview information theory [23, 41, 70, 72, 76]
studying the shared information between two views and whether they are necessary or sufficient in
predicting the label. Recently, Wang et al. [81] and Kahana and Hoshen [39] discuss the limitations
of assuming multiview redundancy and propose autoencoder reconstruction or unimodal contrastive
learning to retain unique information, which resembles the Cross+self baselines in our experiments.
We refer the reader to Shwartz-Ziv and LeCun [67] for a comprehensive review on multiview and
contrastive learning. Our work also relates to conditional contrastive learning [17, 51, 78, 87], where
positive or negative pairs are supposed to sample from conditional distributions.
Multimodal contrastive learning aims to align related data from different modalities, typically
provided as positive pairs. This could be done via optimizing a contrastive objective for inter￾modality pairs [1, 2, 37, 61], or both intra- and inter-modality data pairs [33, 36, 42, 44, 89]. Our
work also relates to factorized representation learning, which primarily studies how to capture
modality-specific information primarily in each modality and multimodal information redundant in
both modalities [32, 75]. Prior work has used disentangled latent variable models [8, 30, 32, 75],
mixture-of-experts [66], or product-of-experts [83] layer to explain factors in multimodal data.
Information theory [18, 65] has been used to study several phenomena in multimodal learning,
including co-learning [62, 92] and multi-view learning [34, 76]. Due to its theoretical importance,
several lower and upper bounds have been proposed for practical estimation [58–60, 84]. We build
on the CLUB upper bound [16] to create a more accurate and stable bound. Our characterizations
of shared and unique information are also related to partial information decomposition [82], co￾information [7, 80], interaction information [53], and cross-domain disentanglement [35] research.
6 Conclusion
This paper studied how standard CL methods suffer when task-relevant information lies in regions
unique to each modality, which is extremely common in real-world applications such as sensor
placement, medical testing, and multimodal interaction. In response, we proposed FACTORCL,
a new method expanding CL techniques through the use of factorized representations, removing
task-irrelevant information via upper bounds on MI, and multimodal data augmentations suitable for
approximating the unobserved task. Based on FACTORCL’s strong performance, there are several
exciting directions in extending these ideas for masked and non-contrastive pre-training; we further
discuss broader impacts and limitations of this line of work in Appendix A.
Acknowledgements
This material is based upon work partially supported by Meta, National Science Foundation awards
1722822 and 1750439, and National Institutes of Health awards R01MH125740, R01MH132225,
R01MH096951 and R21MH130767. PPL is supported in part by a Siebel Scholarship and a Waibel
Presidential Fellowship. RS is supported in part by ONR grant N000142312368 and DARPA
FA87502321015. One of the aims of this project is to understand the comfort zone of people for
better privacy and integrity. Any opinions, findings, conclusions, or recommendations expressed in
this material are those of the author(s) and do not necessarily reflect the views of the sponsors, and no
official endorsement should be inferred. Finally, we would also like to acknowledge feedback from
anonymous reviewers who significantly improved the paper and NVIDIA’s GPU support.
10
References
[1] Hassan Akbari, Liangzhe Yuan, Rui Qian, Wei-Hong Chuang, Shih-Fu Chang, Yin Cui, and Boqing Gong.
Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text. Advances in
Neural Information Processing Systems, 34:24206–24221, 2021.
[2] Jean-Baptiste Alayrac, Adria Recasens, Rosalia Schneider, Relja Arandjelovic, Jason Ramapuram, Jeffrey ´
De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Self-supervised multimodal versatile
networks. Advances in Neural Information Processing Systems, 33:25–37, 2020.
[3] Relja Arandjelovic and Andrew Zisserman. Look, listen and learn. In Proceedings of the IEEE international
conference on computer vision, pages 609–617, 2017.
[4] Philip Bachman, R Devon Hjelm, and William Buchwalter. Learning representations by maximizing
mutual information across views. Advances in neural information processing systems, 32, 2019.
[5] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for
self-supervised learning of speech representations. Advances in neural information processing systems, 33:
12449–12460, 2020.
[6] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for
self-supervised learning. In International Conference on Learning Representations, 2021.
[7] Anthony J Bell. The co-information lattice. In Proceedings of the fifth international workshop on
independent component analysis and blind signal separation: ICA, volume 2003, 2003.
[8] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new
perspectives. TPAMI, 35(8), August 2013.
[9] Tolga Bolukbasi, Kai-Wei Chang, James Y Zou, Venkatesh Saligrama, and Adam T Kalai. Man is to
computer programmer as woman is to homemaker? debiasing word embeddings. Advances in neural
information processing systems, 29, 2016.
[10] Emanuele Bugliarello, Ryan Cotterell, Naoaki Okazaki, and Desmond Elliott. Multimodal pretraining
unmasked: A meta-analysis and a unified framework of vision-and-language berts. Transactions of the
Association for Computational Linguistics, 9:978–994, 2021.
[11] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsu￾pervised learning of visual features by contrasting cluster assignments. Advances in neural information
processing systems, 33:9912–9924, 2020.
[12] Santiago Castro, Devamanyu Hazarika, Verónica Pérez-Rosas, Roger Zimmermann, Rada Mihalcea, and
Soujanya Poria. Towards multimodal sarcasm detection (an _obviously_ perfect paper). arXiv preprint
arXiv:1906.01815, 2019.
[13] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. In International conference on machine learning, pages
1597–1607. PMLR, 2020.
[14] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In Proceedings of the
IEEE/CVF conference on computer vision and pattern recognition, pages 15750–15758, 2021.
[15] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision
transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages
9640–9649, 2021.
[16] Pengyu Cheng, Weituo Hao, Shuyang Dai, Jiachang Liu, Zhe Gan, and Lawrence Carin. Club: A
contrastive log-ratio upper bound of mutual information. In International conference on machine learning,
pages 1779–1788. PMLR, 2020.
[17] Jianfeng Chi, William Shand, Yaodong Yu, Kai-Wei Chang, Han Zhao, and Yuan Tian. Conditional
supervised contrastive learning for fair text classification. arXiv preprint arXiv:2205.11485, 2022.
[18] Thomas M Cover and Joy A Thomas. Information theory and statistics. Elements of information theory, 1
(1):279–335, 1991.
[19] Li Deng. The mnist database of handwritten digit images for machine learning research. IEEE Signal
Processing Magazine, 29(6):141–142, 2012.
11
[20] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirec￾tional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
[21] Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, and Russ R Salakhutdinov. Contrastive learning
as goal-conditioned reinforcement learning. Advances in Neural Information Processing Systems, 35:
35603–35620, 2022.
[22] Meir Feder and Neri Merhav. Relations between entropy and error probability. IEEE Transactions on
Information theory, 40(1):259–266, 1994.
[23] Marco Federici, Anjan Dutta, Patrick Forré, Nate Kushman, and Zeynep Akata. Learning robust represen￾tations via multi-view information bottleneck. arXiv preprint arXiv:2002.07017, 2020.
[24] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning of sentence embeddings.
arXiv preprint arXiv:2104.08821, 2021.
[25] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya,
Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your
own latent-a new approach to self-supervised learning. Advances in neural information processing systems,
33:21271–21284, 2020.
[26] Qing Guo, Junya Chen, Dong Wang, Yuewei Yang, Xinwei Deng, Jing Huang, Larry Carin, Fan Li,
and Chenyang Tao. Tight mutual information estimation with contrastive fenchel-legendre optimization.
Advances in Neural Information Processing Systems, 35:28319–28334, 2022.
[27] Md Kamrul Hasan, Wasifur Rahman, AmirAli Bagher Zadeh, Jianyuan Zhong, Md Iftekhar Tanveer,
Louis-Philippe Morency, and Mohammed Ehsan Hoque. Ur-funny: A multimodal language dataset for
understanding humor. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),
pages 2046–2056, 2019.
[28] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised
visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern
recognition, pages 9729–9738, 2020.
[29] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders
are scalable vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 16000–16009, 2022.
[30] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir
Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational
framework. 2016.
[31] R Devon Hjelm, Alex Fedorov, Samuel Lavoie-Marchildon, Karan Grewal, Phil Bachman, Adam Trischler,
and Yoshua Bengio. Learning deep representations by mutual information estimation and maximization.
In International Conference on Learning Representations, 2018.
[32] Wei-Ning Hsu and James Glass. Disentangling by partitioning: A representation learning framework for
multimodal sensory data. arXiv preprint arXiv:1805.11264, 2018.
[33] Po-Yao Huang, Mandela Patrick, Junjie Hu, Graham Neubig, Florian Metze, and Alexander Hauptmann.
Multilingual multimodal pre-training for zero-shot cross-lingual transfer of vision-language models. arXiv
preprint arXiv:2103.08849, 2021.
[34] Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, and Longbo Huang. What makes
multi-modal learning better than single (provably). Advances in Neural Information Processing Systems,
34:10944–10956, 2021.
[35] HyeongJoo Hwang, Geon-Hyeong Kim, Seunghoon Hong, and Kee-Eung Kim. Variational interaction
information maximization for cross-domain disentanglement. Advances in Neural Information Processing
Systems, 33:22479–22491, 2020.
[36] Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang, and
Jason Baldridge. Mural: multimodal, multitask retrieval across languages. arXiv preprint arXiv:2109.05125,
2021.
[37] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung,
Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text
supervision. In International Conference on Machine Learning, pages 4904–4916. PMLR, 2021.
12
[38] Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi,
Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible
critical care database. Scientific data, 3(1):1–9, 2016.
[39] Jonathan Kahana and Yedid Hoshen. A contrastive objective for learning disentangled representations.
In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022,
Proceedings, Part XXVI, pages 579–595. Springer, 2022.
[40] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional
transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.
[41] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot,
Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in neural information processing
systems, 33:18661–18673, 2020.
[42] Byoungjip Kim, Sungik Choi, Dasol Hwang, Moontae Lee, and Honglak Lee. Transferring pre-trained
multimodal representations with cross-modal similarity matching. Advances in Neural Information
Processing Systems, 35:30826–30839, 2022.
[43] Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Cifar-10 (canadian institute for advanced research).
URL http://www.cs.toronto.edu/~kriz/cifar.html.
[44] Sangho Lee, Youngjae Yu, Gunhee Kim, Thomas Breuel, Jan Kautz, and Yale Song. Parameter efficient
multimodal transformers for video representation learning. arXiv preprint arXiv:2012.04124, 2020.
[45] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu, Michelle A
Lee, Yuke Zhu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Multibench: Multiscale benchmarks
for multimodal representation learning. NeurIPS Datasets and Benchmarks Track, 2021.
[46] Paul Pu Liang, Chiyu Wu, Louis-Philippe Morency, and Ruslan Salakhutdinov. Towards understanding
and mitigating social biases in language models. In International Conference on Machine Learning, pages
6565–6576. PMLR, 2021.
[47] Paul Pu Liang, Yiwei Lyu, Xiang Fan, Shengtong Mo, Dani Yogatama, et al. Highmmt: Towards modality
and task generalization for high-modality representation learning. arXiv preprint arXiv:2203.01311, 2022.
[48] Paul Pu Liang, Amir Zadeh, and Louis-Philippe Morency. Foundations and recent trends in multimodal
machine learning: Principles, challenges, and open questions. arXiv preprint arXiv:2209.03430, 2022.
[49] Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng,
Faisal Mahmood, Ruslan Salakhutdinov, and Louis-Philippe Morency. Quantifying & modeling feature
interactions: An information decomposition framework. arXiv preprint arXiv:2302.12247, 2023.
[50] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: pretraining task-agnostic visiolinguistic
representations for vision-and-language tasks. In Proceedings of the 33rd International Conference on
Neural Information Processing Systems, pages 13–23, 2019.
[51] Martin Q Ma, Yao-Hung Hubert Tsai, Paul Pu Liang, Han Zhao, Kun Zhang, Ruslan Salakhutdinov,
and Louis-Philippe Morency. Conditional contrastive learning for improving fairness in self-supervised
learning. arXiv preprint arXiv:2106.02866, 2021.
[52] Emily E Marsh and Marilyn Domas White. A taxonomy of relationships between images and text. Journal
of documentation, 2003.
[53] William McGill. Multivariate information transmission. Transactions of the IRE Professional Group on
Information Theory, 4(4):93–111, 1954.
[54] Yu Meng, Chenyan Xiong, Payal Bajaj, Paul Bennett, Jiawei Han, Xia Song, et al. Coco-lm: Correcting
and contrasting text sequences for language model pretraining. Advances in Neural Information Processing
Systems, 34:23102–23114, 2021.
[55] Sudipto Mukherjee, Himanshu Asnani, and Sreeram Kannan. Ccmi: Classifier based conditional mutual
information estimation. In Uncertainty in artificial intelligence, pages 1083–1093. PMLR, 2020.
[56] Arvind Neelakantan, Tao Xu, Raul Puri, Alec Radford, Jesse Michael Han, Jerry Tworek, Qiming Yuan,
Nikolas Tezak, Jong Wook Kim, Chris Hallacy, et al. Text and code embeddings by contrastive pre-training.
arXiv preprint arXiv:2201.10005, 2022.
13
[57] XuanLong Nguyen, Martin J Wainwright, and Michael I Jordan. Estimating divergence functionals and
the likelihood ratio by convex risk minimization. IEEE Transactions on Information Theory, 56(11):
5847–5861, 2010.
[58] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive
coding. arXiv preprint arXiv:1807.03748, 2018.
[59] Sherjil Ozair, Corey Lynch, Yoshua Bengio, Aaron Van den Oord, Sergey Levine, and Pierre Sermanet.
Wasserstein dependency measure for representation learning. Advances in Neural Information Processing
Systems, 32, 2019.
[60] Ben Poole, Sherjil Ozair, Aaron Van Den Oord, Alex Alemi, and George Tucker. On variational bounds of
mutual information. In International Conference on Machine Learning, pages 5171–5180. PMLR, 2019.
[61] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish
Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from
natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR,
2021.
[62] Anil Rahate, Rahee Walambe, Sheela Ramanna, and Ketan Kotecha. Multimodal co-learning: challenges,
applications with datasets, recent advances and future directions. Information Fusion, 81:203–239, 2022.
[63] Steffen Schneider, Alexei Baevski, Ronan Collobert, and Michael Auli. wav2vec: Unsupervised pre￾training for speech recognition. arXiv preprint arXiv:1904.05862, 2019.
[64] Bin Shan, Weichong Yin, Yu Sun, Hao Tian, Hua Wu, and Haifeng Wang. Ernie-vil 2.0: Multi-view
contrastive learning for image-text pre-training, 2022.
[65] Claude Elwood Shannon. A mathematical theory of communication. The Bell system technical journal, 27
(3):379–423, 1948.
[66] Yuge Shi, Brooks Paige, Philip Torr, et al. Variational mixture-of-experts autoencoders for multi-modal
deep generative models. Advances in Neural Information Processing Systems, 32, 2019.
[67] Ravid Shwartz-Ziv and Yann LeCun. To compress or not to compress–self-supervised learning and
information theory: A review. arXiv preprint arXiv:2304.09355, 2023.
[68] Jiaming Song and Stefano Ermon. Understanding the limitations of variational mutual information
estimators. CoRR, abs/1910.06222, 2019. URL http://arxiv.org/abs/1910.06222.
[69] Alessandro Sordoni, Nouha Dziri, Hannes Schulz, Geoff Gordon, Philip Bachman, and Remi Tachet
Des Combes. Decomposed mutual information estimation for contrastive representation learning. In
International Conference on Machine Learning, pages 9859–9869. PMLR, 2021.
[70] Karthik Sridharan and Sham M Kakade. An information theoretic framework for multi-view learning. In
Conference on Learning Theory, 2008.
[71] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. ECCV, 2020.
[72] Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What makes for
good views for contrastive learning? Advances in Neural Information Processing Systems, 33:6827–6839,
2020.
[73] Christopher Tosh, Akshay Krishnamurthy, and Daniel Hsu. Contrastive learning, multi-view redundancy,
and linear models. In Algorithmic Learning Theory, pages 1179–1206. PMLR, 2021.
[74] Yao-Hung Hubert Tsai, Tianqin Li, Weixin Liu, Peiyuan Liao, Ruslan Salakhutdinov, and Louis-Philippe
Morency. Learning weakly-supervised contrastive representations. In International Conference on Learning
Representations.
[75] Yao-Hung Hubert Tsai, Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency, and Ruslan Salakhutdinov.
Learning factorized multimodal representations. ICLR, 2019.
[76] Yao-Hung Hubert Tsai, Yue Wu, Ruslan Salakhutdinov, and Louis-Philippe Morency. Self-supervised
learning from a multi-view perspective. In International Conference on Learning Representations, 2020.
[77] Yao-Hung Hubert Tsai, Han Zhao, Makoto Yamada, Louis-Philippe Morency, and Russ R Salakhutdinov.
Neural methods for point-wise dependency estimation. Advances in Neural Information Processing
Systems, 33:62–72, 2020.
14
[78] Yao-Hung Hubert Tsai, Tianqin Li, Martin Q Ma, Han Zhao, Kun Zhang, Louis-Philippe Morency, and
Ruslan Salakhutdinov. Conditional contrastive learning with kernel. arXiv preprint arXiv:2202.05458,
2022.
[79] Michael Tschannen, Josip Djolonga, Paul K Rubenstein, Sylvain Gelly, and Mario Lucic. On mutual infor￾mation maximization for representation learning. In International Conference on Learning Representations,
2019.
[80] Jorge R Vergara and Pablo A Estévez. A review of feature selection methods based on mutual information.
Neural computing and applications, 24:175–186, 2014.
[81] Haoqing Wang, Xun Guo, Zhi-Hong Deng, and Yan Lu. Rethinking minimal sufficient representation
in contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 16041–16050, 2022.
[82] Paul L Williams and Randall D Beer. Nonnegative decomposition of multivariate information. arXiv
preprint arXiv:1004.2515, 2010.
[83] Mike Wu and Noah Goodman. Multimodal generative models for scalable weakly-supervised learning.
Advances in Neural Information Processing Systems, 31, 2018.
[84] Mike Wu, Chengxu Zhuang, Milan Mosse, Daniel Yamins, and Noah Goodman. On mutual information in
contrastive learning for visual representations. arXiv preprint arXiv:2005.13149, 2020.
[85] Jianwei Yang, Chunyuan Li, Pengchuan Zhang, Bin Xiao, Ce Liu, Lu Yuan, and Jianfeng Gao. Unified
contrastive learning in image-text-label space, 2022.
[86] Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi,
and Junzhou Huang. Vision-language pre-training with triple contrastive learning, 2022.
[87] Zesheng Ye and Lina Yao. Contrastive conditional neural processes. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 9687–9696, 2022.
[88] Ron Yosef, Yonatan Bitton, and Dafna Shahaf. Irfl: Image recognition of figurative language. arXiv
preprint arXiv:2303.15445, 2023.
[89] Xin Yuan, Zhe Lin, Jason Kuen, Jianming Zhang, Yilin Wang, Michael Maire, Ajinkya Kale, and Baldo
Faieta. Multimodal contrastive training for visual representation learning. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, pages 6995–7004, 2021.
[90] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo.
Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the
IEEE/CVF international conference on computer vision, pages 6023–6032, 2019.
[91] Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. Mosi: multimodal corpus of
sentiment intensity and subjectivity analysis in online opinion videos. arXiv preprint arXiv:1606.06259,
2016.
[92] Amir Zadeh, Paul Pu Liang, and Louis-Philippe Morency. Foundations of multimodal co-learning.
Information Fusion, 64:188–193, 2020.
[93] AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency.
Multimodal language analysis in the wild: Cmu-mosei dataset and interpretable dynamic fusion graph. In
Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pages 2236–2246, 2018.
[94] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Self-supervised
learning via redundancy reduction. In International Conference on Machine Learning, pages 12310–12320.
PMLR, 2021.
15
Appendix
A Broader Impact
Multimodal data and self-supervised models are ubiquitous in a range of real-world applications.
This paper is our attempt at broadening the applicability of self-supervised contrastive methods to a
wider range of multimodal tasks beyond those that exhibit multi-view redundancy. We believe that
special care must be taken to ensure that these models are safely deployed for real-world benefit:
Time and space complexity: Modern multimodal models are large and take up a significant amount
of carbon footprint during training and testing. As compared to heuristic combinations of cross-modal
and single-modality CL [33, 36, 44, 64, 81, 85, 89], we believe that FACTORCL does not significantly
increase complexity: (1) upper bounds on MI can be estimated “for free” by directly plugging in
the optimal critic from INCE, and (2) removal of task-irrelevant information via I(X1; X2∣X1
′
, X2
′
)
shares encoders with INCE, and (3) separate unimodal augmentations perform well enough in practice.
We also release our code and models so that they can be evaluated quickly on new tasks, which can
amortize complexity costs.
Privacy and security: There may be privacy risks associated with making predictions from multi￾modal data of recorded human behaviors and medical data (i.e., the datasets used in our experiments
for analysis of sentiment, emotions, personality, sarcasm, and humor, as well as disease prediction
from medical data). We have followed best practices in maintaining the privacy and safety of these
datasets: (1) the creators of these video datasets have taken the appropriate steps to only access
public data that participants or content creators have consented for public release (creative commons
license and following fair use guidelines of YouTube) [12, 27, 93], (2) MIMIC has been rigorously
de-identified in accordance with Health Insurance Portability and Accountability Act (HIPAA) such
that all possible personal information has been removed from the dataset [38], (3) all video data
was also anonymized and stripped of all personal (e.g., personally identifiable information) and
protected attributes (e.g., race, gender), (4) all models trained on affect recognition datasets use only
pre-extracted non-invertible features that rely on general visual or audio features such as the presence
of a smile or magnitude of voice which cannot be used to identify the speaker [91, 93], and (5) we
studied the videos collected in these affective computing datasets and found no offensive words used
or personal attacks recorded in the video. Finally, we only use these datasets for research purposes
and emphasize that any multimodal models trained to perform prediction should only be used for
scientific study and should not in any way be used for real-world harm.
Social biases: We acknowledge risks of social bias due to imbalanced datasets, resulting in potential
biases surrounding gender, race, and ethnicity, among others [9, 46]. We note that our FACTORCL
approach has a close link with conditional CL [51], which can also be adapted to condition on
sensitive attributes and therefore reduce bias. Studying these research questions is an important
direction for future work.
Future work: We discuss more limitations and potential future work in this direction. Firstly,
optimizing our objectives using better MI lower and upper bounds such as in Guo et al. [26] and
Sordoni et al. [69], could improve the performance for inputs of higher dimension and complex
modality. Next, the current data augmentation method requires one to pick augmentations to approxi￾mately satisfy Definition 4; future work could extend InfoMin [72] to automatically generate data
augmentations to satisfy Definition 4, or leverage future progress in multimodal generative models
for data augmentation. Lastly, future work could quantify whether shared or unique information is
more important for different tasks and reweight the terms in the FACTORCL objective to suit the
tasks.
B Analysis of Multi-view Contrastive Learning
Multi-view shared information describes the extent and dimensions in which information can be
shared across different views. The presence of shared information is often in contrast to unique
information that exists solely in a single modality, and can be formalized via information theory:
Definition 5. (Shared information) Given X1 and X2, I(X1; X2) = ∫ p(x1, x2) log p(x1,x2)
p(x1)p(x2)
measures the degree of information-theoretic shared information between X1 and X2.
Definition 6. (Task-relevant shared information) Given X1, X2, and a target Y , I(X1; X2; Y ) =
I(X1; X2) − I(X1; X2∣Y ) = ∫ p(x1, x2) log p(x1,x2)
p(x1)p(x2)
− ∫ p(x1, x2∣y) log p(x1,x2∣y)
p(x1∣y)p(x2∣y)
measures
16
the amount of task-relevant shared information between X1 and X2 for predicting Y . I(X1; X2∣Y )
represents the task-irrelevant shared information.
Learning shared information via contrastive learning: Current approaches for multi-view con￾trastive learning model shared information I(X1; X2) (and subsequently task-relevant shared infor￾mation I(X1; X2; Y ) during downstream task fine-tuning), without modeling unique information.
Z1 = arg max
Z1∶=fθ(X1)
I(Z1; X2), Z2 = arg max
Z2∶=fθ(X2)
I(X1;Z2). (19)
Optimizing for I(X1; X2) is performed via a surrogate loss during self-supervised pre-training
(where we do not have access to the label Y ) by maximizing the InfoNCE objective:
INFONCE = sup
f
Ex1,x
+
2∼p(x1,x2)
x
−
2∼p(x2)
[log exp f(x1, x+
2
)
∑k
exp f(x1, x−
2
)
] , (20)
Oord et al. [58] show that I(X1; X2) ≥ log k − LNCE(X1; X2) where LNCE(X1; X2) is negative
of INFONCE and is the loss to minimize (rather than maximize) in training. NCE falls into a
broader class of contrastive learning methods [13, 15, 28, 41, 61] that model the ratio between
joint densities p(x1, x2) and product of marginal densities p(x1)p(x2) using positive and negative
samples [57, 59, 60, 79, 84] or probabilistic classifiers [55, 77], all of which can also be used to
capture shared information.
Tian et al. [71] argues that the optimal view of contrastive learning is also minimal: the minimal
representations only extract relevant information of the contrastive task (maximizing the shared part)
and throw away other information. Therefore, from this minimal assumption, we have I(Z1; Y ∣X2) =
0 and I(Z2; Y ∣X1) = 0 as minimal Z1 and Z2 only captures task-relevant information from the shared
part. By conditioning on X1 or X2, the shared part is removed, and Z1 and Y (or Z2 and Y ) do not
share information.
Lastly, we restate the multi-view non-redundancy from Definition 2:
Definition 7. (Multi-view non-redundancy) ∃ϵ > 0 such that I(X1; Y ∣X2) > ϵ or I(X2; Y ∣X1) > ϵ.
We would like to compare and clarify the differences between the multiview redundancy assumption
in Eq.(1) and the multi-view nonredundancy in Def. 7. The multiview redundancy assumption in
Eq.(1) states that the task-relevant information from the unique part is minimal (≤ ϵ). The multiview
non-redundancy states the opposite: the task-relevant information from the unique part is nonzero
and nonminimal, as it is not bounded by ϵ. Next we briefly clarify the relationship between these
two assumptions and the InfoMin assumption: I(Z1; Y ∣X2) = I(Z2; Y ∣X1) = 0. InfoMin is about
representation Z while the redundancy assumptions are only about data X. InfoMin states that the
optimal (sufficient and minimal) representation learns task-relevant information only from the shared
part, as we discussed in the paragraph above. We empirically checked the two assumptions: Tables 1
and 4 in the main text show that the multiview non-redundancy assumption holds empirically, and
Table 11 shows that the InfoMin assumption holds empirically.
We now show the limitations of CL methods, first restating the Theorem here:
Theorem 4. (Suboptimality of standard CL) When there is multi-view non-redundancy as in Definition
7, given optimal representations {Z1, Z2} that satisfy Eq.(19 and I(Z1; Y ∣X2) = I(Z2; Y ∣X1) =
0 [72], we have that
I(Z1, Z2; Y ) = I(X1, X2; Y ) − I(X1; Y ∣X2) − I(X2; Y ∣X1) = I(X1; X2) − I(X1; X2∣Y ) < I(X1, X2; Y ).
(21)
Proof. Since Z1 and Z2 maximize I(X1; X2) we have that I(Z1; X2) = I(X1;Z2) = I(X1; X2) so
I(Z1; X2; Y ) = I(X1;Z2; Y ) = I(X1; X2; Y ) and I(Z1; X2∣Y ) = I(X1;Z2∣Y ) = I(X1; X2∣Y ).
We now show the relationship between I(X1, X2; Y ), which is the total information X1, X2 con￾tributes towards predicting Y in classic supervised learning, with I(Z1, Z2; Y ), which is the infor￾mation that our learned self-supervised representations can contribute towards Y during supervised
fine-tuning. We first derive the relationship between I(Z1; Y ) and I(X1; Y ):
I(Z1; Y ) = I(Z1; X2; Y ) + I(Z1; Y ∣X2) (22)
= I(X1; X2; Y ) + I(Z1; Y ∣X2) (23)
= I(X1; Y ) − I(X1; Y ∣X2) + I(Z1; Y ∣X2) (24)
= I(X1; Y ) − I(X1; Y ∣X2) (25)
17
Given X1, we further derive a relationship between I(Z2; Y ∣Z1) and I(X2; Y ∣X1):
I(Z2; Y ∣Z1) = I(Z2; X1; Y ∣Z1) + I(Z2; Y ∣Z1, X1) (26)
= I(X1; X2; Y ∣Z1) + I(Z2; Y ∣Z1, X1) (27)
= I(X1; X2; Y ∣Z1) + I(Z2; Y ∣X1) (28)
= I(X2; Y ∣Z1) − I(X2; Y ∣X1, Z1) + I(Z2; Y ∣X1) (29)
= I(X2; Y ∣Z1) − I(X2; Y ∣X1) + I(Z2; Y ∣X1) (30)
= I(X2; Y ) − I(Z1; X2; Y ) − I(X2; Y ∣X1) + I(Z2; Y ∣X1) (31)
= I(X2; Y ) − I(X1; X2; Y ) − I(X2; Y ∣X1) + I(Z2; Y ∣X1) (32)
= I(X2; Y ∣X1) − I(X2; Y ∣X1) + I(Z2; Y ∣X1) = 0 (33)
In Eqs.(28) and (30), we use the fact that conditioning on Z1 and X1 jointly reduces to conditioning
on X1 since Z1 is deterministically obtained from X1, and in Eq.(32) we use the definition of learning
Zs to maximize I(X1; X2) so I(Z1; X2; Y ) = I(X1;Z2; Y ). Finally, adding both terms up,
I(Z1, Z2; Y ) = I(Z1; Y ) + I(Z2; Y ∣Z1) (34)
= I(X1; Y ) − I(X1; Y ∣X2) (35)
= I(X1; X2; Y ) (36)
= I(X1, X2; Y ) − I(X1; Y ∣X2) − I(X2; Y ∣X1) (37)
= I(X1; X2) − I(X1; X2∣Y ) (38)
gives the desired result.
Bayes error rate. The Bayes error rate Pe(Z1, Z2) ∶= 1 − EPZ1,Z2
[maxy∈Y P (Yˆ = y ∣ z1, z2)] of
contrastive representations {Z1, Z2} is given by:
Pe ≤ 1 − exp [I(X1, X2; Y ) − I(X1; Y ∣X2) − I(X2; Y ∣X1) − H(Y )] (39)
= 1 − exp [I(X1; X2; Y ) − H(Y )] (40)
Proof. We use the inequality between Pe and H(Y ∣Z) [22, 76, 81]:
−ln(1 − Pe) ≤ H(Y ∣Z), or equivalently, Pe ≤ 1 − exp[−H(Y ∣Z)] (41)
If we regard Z as the joint of Z1 and Z2, then we have
Pe ≤ 1 − exp[−H(Y ∣Z1, Z2)] (42)
We further expand H(Y ∣Z1, Z2) by definition of mutual information, I(X; Y ) = H(X) − H(X∣Y ),
Theorem 4, and the I(X1; X2; Y ) = I(X1; X2) − I(X1; X2∣Y ):
H(Y ∣Z1, Z2) = H(Y ) − I(Z1, Z2; Y ) (43)
= H(Y ) − I(X1, X2; Y ) + I(X1; Y ∣X2) + I(X2; Y ∣X1) (44)
= H(Y ) − I(X1; X2) + I(X1; X2∣Y ) (45)
= H(Y ) − I(X1; X2; Y ) (46)
Plugging in Eq.(42), we have
Pe ≤ 1 − exp[−H(Y ∣Z1, Z2)] (47)
= 1 − exp[−(H(Y ) − I(X1, X2; Y ) + I(X1; Y ∣X2) + I(X2; Y ∣X1))] (48)
= 1 − exp[−H(Y ) + I(X1, X2; Y ) − I(X1; Y ∣X2) − I(X2; Y ∣X1)] (49)
and
Pe ≤ 1 − exp[−H(Y ∣Z1, Z2)] (50)
= 1 − exp[−(H(Y ) − I(X1; X2; Y ))] (51)
= 1 − exp[−H(Y ) + I(X1; X2; Y )] (52)
resulting in the Bayes error rate as desired.
18
C FACTORIZED CONTRASTIVE LEARNING
C.1 Contrastive estimators
Theorem 5. (Contrastive estimators for I(X1; X2)) Defining the NCE estimator and NCE-CLUB
estimator as follows,
INCE(X1; X2) = Ex1,x
+
2∼p(x1,x2)
x
−
2∼p(x2)
[log exp f(x1, x+
2
)
∑k
exp f(x1, x−
2
)
] (53)
INCE-CLUB(X1; X2) = Ex1,x+
2
∼p(x1,x2) [f
∗
(x1, x+
2
)] − Ex1∼p(x1)
x
−
2∼p(x2)
[f
∗
(x1, x−
2
)] (54)
where f
∗
(x1, x2) is the optimal critic from INCE plugged into the ICLUB objective [16]. We call the
proposed plug-in objective Eq.(11) INCE-CLUB, and obtain lower and upper bounds on I(X1; X2):
INCE(X1; X2) ≤ I(X1; X2) ≤ INCE-CLUB(X1; X2). (55)
Proof. The lower bound INCE(X1; X2) ≤ I(X1; X2) follows from Oord et al. [58]: optimizing the
objective leads to an optimal critic f
∗ = log p(x2∣x1) + c(x2) [60] or without loss of generality
f
∗ = log p(x1∣x2) + c(x1), where c(⋅) is an arbitrary deterministic function. Plugging the optimal
critic into the INCE(X1; X2) gives the result: INCE(X1; X2) + log N ≤ I(X1; X2) [58, 60].
Next, the original ICLUB(X1; X2) [16] is defined as:
ICLUB(X1; X2) ∶= Ep(x1,x2) [log p(x2∣x1)] − Ep(x1)p(x2) [log p(x2∣x1)]. (56)
As mutual information is symmetric w.r.t x1 and x2: I(X1; X2) = I(X2; X1) , without loss of
generality, we have:
ICLUB(X1; X2) = ICLUB(X2; X1) = Ep(x1,x2) [log p(x1∣x2)] − Ep(x1)p(x2) [log p(x1∣x2)] (57)
The formulation above assumes p(x1∣x2) is known, which is satisfied when we use the optimal critic
f
∗ = log p(x1∣x2) + c(x1) from INCE(X1; X2). Plugging the optimal critic f
∗
into ICLUB(X1; X2),
we obtain a desired upper bound INCE-CLUB(X1; X2) of I(X1; X2):
INCE-CLUB(X1; X2) = Ep(x1,x2) [log p(x1∣x2) + c(x1)] − Ep(x1)p(x2) [log p(x1∣x2) + c(x1)] (58)
= Ep(x1,x2) [log p(x1∣x2)] + Ep(x1,x2) [c(x1)] − Ep(x1)p(x2) [log p(x1∣x2)] − Ep(x1)p(x2) [c(x1)]
(59)
= Ep(x1,x2) [log p(x1∣x2)] − Ep(x1)p(x2) [log p(x1∣x2)] (60)
≥ I(X1; X2). (61)
Eq.(59) is from the linearity of expectation, Eq.(60) is from the fact that c(x1) is not a function of x2
and therefore Ep(x1,x2) [c(x1)] = Ep(x1)p(x2) [c(x1)] = Ep(x1) [c(x1)], and Eq.(61) is directly from
the original ICLUB(X1; X2) paper [16].
C.2 Unimodal and multimodal augmentations
We first restate the definitions of optimal single-view and multi-view augmentation:
Definition 8. (Optimal unimodal augmentation) X1
′
is an optimal unimodal augmentation for X1
when I(X; X′
) = I(X; Y ), which implies that the only information shared between X and X′
is
task-relevant with no irrelevant noise.
Definition 9. (Optimal multimodal augmentation) X1
′
and X2
′
are optimal multimodal augmentation
for X1 and X2 when I(X1, X2; X1
′
, X2
′
) = I(X1, X2; Y ), which implies that the only information
shared between X1, X2 and X1
′
, X2
′
is task-relevant with no irrelevant noise.
When are these assumptions satisfied? I(X; X′
) = I(X; Y ) holds when all information shared be￾tween X and X′
is task-relevant, which implies that the augmentation keeps task-relevant information
constant while changing task-irrelevant information. In the case of image classification, task-relevant
information is the object in the picture, while task-irrelevant information is the background. To satisfy
I(X1, X2; X1
′
, X2
′
) = I(X1, X2; Y ), by the chain rule of MI, we augment in two steps:
Unimodal aug: X1
′
s.t. I(X1; X1
′
) = I(X1; Y ), (62)
Unique aug: X2
′
s.t. I(X2; X2
′
∣X1) = I(X2; Y ∣X1). (63)
19
Table 5: More examples of optimal single-view and multi-view augmentations.
Standard Aug Unique Aug
Dataset X1 X2 X1
′ X2
′ X2
′′
Cartoon Caption Image Word Masking Crop + Flip + Resize Flip + Resize
MIMIC Signals Tables Time Warping CutMix [90] on All Features CutMix on Nonclinical Features
MOSEI Transcripts Video+Audio Word Masking Noise Injection on Any Frames Noise Injection on Silent Frames
UR-FUNNY Transcripts Video+Audio Word Masking Noise Injection on Any Frames Noise Injection on Silent Frames
MUsTARD Transcripts Video+Audio Word Masking Noise Injection on Any Frames Noise Injection on Silent Frames
the second step is the unique augmentation: after observing X1, we create augmented X2
′
from X2
to keep the task-relevant information but meanwhile do not affect any information from X1. In Table
5, we include some more examples of how unique augmentations could be designed across different
datasets.
Final objectives: If Definitions 8 and 9 are both satisfied, we can substitute contrastive estimators in
the following equations:
INCE(X1; X2∣Y ) = Ep(y)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x
+
2
∼p(x1,x2∣y)
x
−
2
∼p(x2∣y)
[log exp f(x1, x
+
2 , y)
∑k
exp f(x1, x−
2
, y)
]
⎤
⎥
⎥
⎥
⎥
⎦
(64)
INCE-CLUB(X1; X2∣Y ) = Ep(y)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x+
2
∼p(x1,x2∣y)
[f
∗
(x1, x
+
2 , y)] − Ex1∼p(x1∣y)
x
−
2
∼p(x2∣y)
[f
∗
(x1, x
−
2 , y)]
⎤
⎥
⎥
⎥
⎥
⎦
(65)
by replacing I(Xi
; Y ) terms with I(Xi
; Xi
′
) and replacing I(X1; X2∣Y ) terms with
I(X1; X2∣X1
′
, X2
′
):
INCE(X1; X2∣X1
′
, X2
′
) = Ep(x′
1
,x′
2
)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x
+
2
∼p(x1,x2∣x
′
1
,x
′
2
)
x
−
2
∼p(x2∣x
′
1
,x
′
2
)
[log exp f(x1, x
+
2 , x
′
1, x
′
2)
∑k
exp f(x1, x−
2
, x′
1
, x′
2
)
]
⎤
⎥
⎥
⎥
⎥
⎦
(66)
INCE-CLUB(X1; X2∣X1
′
, X2
′
) = Ep(x′
1
,x′
2
)[Ex1,x+
2
∼p(x1,x2∣x′
1
,x′
2
)
[f
∗
(x1, x
+
2 , x
′
1, x
′
2)]
− Ex1∼p(x1∣x
′
1
,x
′
2
)
x
−
2
∼p(x2∣x
′
1
,x
′
2
)
[f
∗
(x1, x
−
2 , x
′
1, x
′
2)]] (67)
C.2.1 Implementing conditional CL via kernel
We restate our objectives below:
INCE(X1; X2∣X1
′
, X2
′
) = Ep(x′
1
,x′
2
)
⎡
⎢
⎢
⎢
⎢
⎣
Ex1,x
+
2
∼p(x1,x2∣x
′
1
,x
′
2
)
x
−
2
∼p(x2∣x
′
1
,x
′
2
)
[log exp f(x1, x
+
2 , x
′
1, x
′
2)
∑k
exp f(x1, x−
2
, x′
1
, x′
2
)
]
⎤
⎥
⎥
⎥
⎥
⎦
(68)
INCE-CLUB(X1; X2∣X1
′
, X2
′
) = Ep(x′
1
,x′
2
)[Ex1,x+
2
∼p(x1,x2∣x′
1
,x′
2
)
[f
∗
(x1, x
+
2 , x
′
1, x
′
2)]
− Ex1∼p(x1∣x
′
1
,x
′
2
)
x
−
2
∼p(x2∣x
′
1
,x
′
2
)
[f
∗
(x1, x
−
2 , x
′
1, x
′
2)]] (69)
However, sampling from p(⋅∣x
′
1
, x′
2
) is hard. Since X1
′
, X2
′
are continuous variables, directly sam￾pling from the conditional distributions p(⋅∣x
′
1
, x′
2
) may be difficult; training a generative model
pθ(x1, x2∣x
′
1
, x′
2
) from augmented data x
′
1
, x′
2
to original data x1, x2 can be expensive and nontrivial
in a multimodal setup. In this work, we implement the conditioning in p(x1, x2∣x
′
1
, x′
2
) through con￾catenation and the details are in Appendix D.1. Here we discuss an alternative solution to this problem
introduced by Tsai et al. [78]. It leverages kernel methods for conditional sampling in contrastive
learning by assigning weights to each sampled data given the kernel similarity between conditioned
variables, avoiding directly sampling from the conditional distributions or training generative models.
In our formulation, given multimodal input (x1, x2) with their augmentation (x
′
1
, x′
2
), we can sim￾ply use the technique from [78] to estimate INCE(X1; X2∣X1
′
, X2
′
) and INCE-CLUB(X1; X2∣X1
′
, X2
′
),
where the kernel measures the similarity between different pairs (x
′
1
, x′
2
) of the conditional variable
20
X1, X2. Specifically,
INCE(X1; X2∣X1
′
, X2
′
) = Ep(x1,x2,x′
1
,x′
2
)
⎡
⎢
⎢
⎢
⎢
⎣
log exp f(x1, x+
2
)
exp f(x1, x2) + n ∗ [KX1⊥⊥X2∣X1
′
,X2
′ ]
ii
⎤
⎥
⎥
⎥
⎥
⎦
(70)
INCE-CLUB(X1; X2∣X1
′
, X2
′
) = Ep(x1,x2,x′
1
,x′
2
) [f
∗
(x1, x2) − log [KX1⊥⊥X2∣X1
′
,X2
′ ]
ii] (71)
where KX1⊥⊥X2∣X′
1
,X2
′ = KX1X2
(KX1
′ X2
′ + λI)
−1KX′
1X2
′ and [KX1⊥⊥X2∣X1
′
,X2
′ ]
ii
is the ith row and
ith column of KX1⊥⊥X2∣X′
1
,X2
′ . KX1X2
is a kernel similarity matrix between X1 and X2, and KX1
′ X2
′
is a separate kernel similarity matrix between X1
′
and X2
′
. f
∗
is the optimal solution of Eq.(70).
By leveraging the similarity KX′
1X2
′ between conditional variables X1
′
and X2
′
, KX1⊥⊥X2∣X1
′
,X2
′
transforms the similarity scores between X1 and X2 under unconditional distributions into similarity
scores under conditional distributions. Note that the expectations in Eqs.(70) and (71) are taken
over the joint distribution p(x1, x2, x′
1
, x′
2
), which comes naturally after augmenting both modalities
X1 and X2. This method could effectively alleviate the problem of sampling from conditional
distributions in our formulation. We refer the reader to Tsai et al. [78] for more details.
C.3 Final estimators in FACTORCL
Theorem 6. (Contrastive estimators for shared and unique information). Under assumptions on
single-view augmentations I(X1; Y ) = I(X1, X1
′
) (Definition 8) and optimal multi-view augmen￾tation X2
′
such that I(X1, X2; X1
′
, X2
′
) = I(X1, X2; Y ) (Definition 9), we can define contrastive
objectives for task-relevant shared and unique information with:
S = I(X1; X2; Y ) ≥ INCE(X1; X2) − INCE-CLUB(X1; X2∣X1
′
, X2
′
) (72)
Ui = I(Xi
; Y ∣X−i) ≥ INCE(Xi
; Xi
′
) − INCE-CLUB(X1; X2) + INCE(X1; X2∣X1
′
, X2
′
) (73)
Proof. The objectives follow from the fact that INCE(X1; X2) and INCE(X1; X2∣X1
′
, X2
′
)
are lower bounds of I(X1; X2) and I(X1; X2∣Y ) respectively, and INCE-CLUB(X1; X2) and
INCE-CLUB(X1; X2∣X1
′
, X2
′
) are upper bounds of I(X1; X2) and I(X1; X2∣Y ) respectively:
S = I(X1; X2; Y ) = I(X1; X2) − I(X1; X2∣Y ) (74)
≥ INCE(X1; X2) − INCE-CLUB(X1; X2∣X1
′
, X2
′
) (75)
Ui = I(Xi
; Y ∣X−i) = I(Xi
; Y ) − (I(X1; X2) − I(X1; X2∣Y )) (76)
≥ INCE(Xi
; Xi
′
) − (INCE-CLUB(X1; X2) − INCE(X1; X2∣X1
′
, X2
′
)) (77)
and symmetrically for U2.
Now we show that FACTORCL learns both shared and unique task-relevant information. First, we
restate the definition of the factorized representations:
ZS1 = arg max
Z1=fθ(X1)
I(Z1; X2; Y ), ZS2 = arg max
Z2=fθ(X2)
I(Z2; X1; Y ), (78)
ZU1 = arg max
Z1=fθ(X1)
I(Z1; Y ∣X2), ZU2 = arg max
Z2=fθ(X2)
I(Z2; Y ∣X1). (79)
where I(Z1; X2; Y ) = I(Z1; X2) − I(Z1; X2∣Y ) is the shared information and I(Z2; X1; Y ) =
I(Z2; X2) − I(Z2; X1∣Y ) is the unique information.
Theorem 7. (Optimality of FACTORCL) If ZS1
, ZS2
, ZU1
, ZU2 perfectly maximize Eqs.(78-79)
and the estimations in Eqs.(13-67) are tight, we obtain I(X1, X2; Y ) = I(ZS1
;ZS2
; Y ) +
I(ZU1
; Y ∣ZS2
) + I(ZU2
; Y ∣ZS1
), suggesting that FACTORCL learns both shared and unique task￾relevant information.
Proof. Because I(X1, X2; Y ) = I(X1; X2; Y ) + I(X1; Y ∣X2) + I(X2; Y ∣X1), it is sufficient to
show that I(X1; X2; Y ) = I(ZS1
;ZS2
; Y ), I(X1; Y ∣X2) = I(ZU1
; Y ∣ZS2
) and I(X2; Y ∣X1) =
I(ZU2
; Y ∣ZS1
).
21
First we show I(X1; X2; Y ) = I(ZS1
;ZS2
; Y ). Crucially, by definition of how ZS1
and ZS2
are
optimized to maximize I(X1; X2; Y ), we have that:
I(X1; X2; Y ) = I(ZS1
; X2; Y ) = I(ZS2
; X1; Y ). (80)
We can then obtain
I(X1; X2; Y ) = I(X1;ZS2
; Y ) (81)
= I(X1;ZS2
; Y ∣ZS1
) + I(ZS1
;ZS2
; X1; Y ) (82)
= I(ZS2
; Y ∣ZS1
) − I(ZS2
; Y ∣ZS1
, X1) + I(ZS1
;ZS2
; X1; Y ) (83)
= I(ZS2
; Y ∣ZS1
) − I(ZS2
; Y ∣X1) + I(ZS1
;ZS2
; X1; Y ) (84)
= I(ZS2
; Y ∣ZS1
) − I(ZS2
; Y ∣X1) + I(ZS1
;ZS2
; Y ) (85)
= I(ZS2
; Y ∣ZS1
) + I(ZS1
;ZS2
; Y ) (86)
= I(ZS1
;ZS2
; Y ) (87)
where Eq.(84) is because ZS1
are deterministically obtained from S1 and Eq.(85) is because ZS1
maximizes the shared information. Finally, we go to Eq.(87) I(ZS2
; Y ∣ZS1
) = 0 as shown in
Eqs.(26-33) using the fact that ZS1
is learned to maximize I(X1; X2; Y ) and I(ZS1
; X2; Y ) =
I(X1;ZS2
; Y ).
Next, we show I(X1; Y ∣X2) = I(ZU1
; Y ∣ZS2
):
I(ZU1
; Y ∣ZS2
) = I(ZU1
; Y ∣ZS2
, ZU2
) + I(ZU1
; Y ;ZU2
∣ZS2
), (88)
which is by the chain rule of conditional mutual information. Then we show I(ZU1
; Y ;ZU2
∣ZS2
) = 0:
I(ZU1
; Y ;ZU2
∣ZS2
) = I(ZU1
;ZU2
∣ZS2
) − I(ZU1
;ZU2
∣Y ;ZS2
) = 0 − 0 = 0 (89)
This is because Eq.(79) leads to I(ZU1
; Y ∣X2) = I(X1; Y ∣X2) and I(ZU2
; Y ∣X1) = I(X2; Y ∣X1).
If the estimations in Eqs.(13-67) are tight, by conditioning and by the previously stated
I(ZU1
; Y ∣X2) = I(X1; Y ∣X2), ZU1
tightly captures information from only X1 and not in X2.
The same applies to ZU2
. We have I(ZU1
; X2) = I(ZU2
; X1) = I(ZU1
;ZU2
) = I(ZU1
;ZU2
∣T) = 0
with T being an arbitrary random variable because no shared information exists between ZU1
and
ZU2
. Then we obtain:
I(ZU1
; Y ∣ZS2
, ZU2
) = I(ZU1
; Y ∣ZS2
, ZU2
, X2) + I(ZU1
; Y ; X2∣ZS2
, ZU2
) (90)
= I(ZU1
; Y ∣X2) (91)
We use the fact that conditioning on ZS2
, ZU2
and X2 jointly reduces to conditioning on X2 since
ZS2
and ZU2
are deterministically obtained from X2. Lastly, since Eqs.(78-79) are satisfied, ZU1 =
arg maxZ1=fθ(X1)
I(Z1; Y ∣X2) therefore I(ZU1
; Y ∣X2) = I(X1; Y ∣X2). We have:
I(ZU1
; Y ∣ZS2
) = I(ZU1
; Y ∣X2) = I(X1; Y ∣X2). (92)
The proof for I(X2; Y ∣X1) = I(ZU2
; Y ∣ZS1
) is similar. We now have shown that I(X1; X2; Y ) =
I(ZS1
;ZS2
; Y ), I(X1; Y ∣X2) = I(ZU1
; Y ∣ZS2
) and I(X2; Y ∣X1) = I(ZU2
; Y ∣ZS1
), adding up all
LHS and RHS we have the theorem.
C.4 Extensions to masking and non-contrastive learning
We now show how similar ideas can be extended to other popular self-supervised learning objectives,
such as non-contrastive learning [6, 94] and masked pre-training [20, 29]. Importantly, this paper pro￾vides a new principle for multimodel self-supervised learning: (1) learning task-relevant information
and (2) removing task-irrelevant information from both shared and unique parts across modalities.
Our paper focuses on realizing this principle via multi-view information theory and contrastive
learning. Below we provide two potential alternatives to realize this principle on non-contrastive and
masking methods, respectively:
Non-contrastive learning: Methods such as Barlow Twins [94] and VICReg [6] use invariance and
covariance regularizations to maximally preserve shared information in the embeddings across two
modalities. However, the embeddings learned may contain only contain task-relevant information
from the shared part and not unique parts. To use the principle in this paper to capture more task￾relevant information from unique parts, one should perform VIC-regularization on X1 features, on
22
X2 features, and on X1, X2 cross-modal features. When performing VICReg on unimodal features,
one should condition on the other modality when performing augmentation. Specifically, similar to
the idea of multimodal augmentation in this paper, the augmentation of the second modality should
not interfere with the shared part (i.e., do not augment regions referred to by the first modality),
making the invariance and covariance regularization of the second modality focus on the augmented
modality-unique features. This makes the model learn unique modality features that are not captured
by the joint embedding from standard independent augmentations.
Masking: Conceptually, masking [20, 29] can be interpreted as leveraging unmasked regions in the
same modality to predict masked regions or leveraging the other modality to predict the masked
region. However, the learned representation may not be all task-relevant. To use the principle in this
paper to exclude task-irrelevant information and capture more task-relevant information from unique
parts, we can perform conditional masking, where masking is conditioned on augmented views
(similar to the multimodal augmentation in the paper, where the conditioned views are approximating
the labels). As a result, only unique regions in the second modality can be masked out, making the
model capture more unique information from the second modality by masked prediction. Here we
have only provided high-level intuitions of extensions to these methods, and future work should
explore these ideas in more detail.
D Experimental Details
D.1 Implementation details
Objective Formulation and Architecture
In Algorithm 2 in the main text, we see the sketch for doing contrastive learning with our proposed
objectives. To implement all algorithms used in our ablation experiments, we start with two encoders
e1(⋅) and e2(⋅), which takes samples x1 and x2 from the modalities X1 and X2, and outputs
corresponding representations z1 and z2. We also have a critic function fθ(⋅, ⋅) parametrized by θ
which takes z1 and z2 as inputs and returns a scalar. A popular way to perform contrastive learning
aims to maximize INCE(X1; X2), where
INCE(X1; X2) = Ex1,x
+
2∼p(x1,x2)
x
−
2∼p(x2)
[log exp fθ(e1(x1), e2(x
+
2
))
∑k
exp fθ(e1(x1), e2(x
−
2
)) ] . (93)
In our algorithms, we follow the derivations in Eqs.(8-9) to maximize each INCE objective and
minimize each INCE-CLUB objective. Therefore, for each objective, we add two additional MLP heads
on top of the two encoders and create a separate critic which takes in the outputs of the MLP heads
instead of the encoders. In all the experiments, we adopt the concat critic design [58, 60, 68], where
fθ(x, y) = hθ([x, y]) with hθ being an MLP.
FACTORCL-SUP: In the supervised version of CL which uses label Y , the objective we aim to
maximize is formulated as
LFACTORCL−SUP = INCE(X1; X2) − INCE-CLUB(X1; X2∣Y ) (94)
+ INCE(X1; Y ) + INCE(X2; Y ) (95)
− INCE-CLUB(X1; X2) + INCE(X1; X2∣Y ). (96)
Each INCE and INCE-CLUB term in this objective is calculated using its own critic as discussed above.
The conditional terms involving the label Y are implicitly modeled by directly concatenating Y
to the outputs of both heads before feeding into the critic. To obtain the learned representations
ZS1
, we concatenate the outputs of the heads on top of the encoder e1 that correspond to the terms
INCE(X1; X2) and INCE-CLUB(X1; X2∣Y ). To obtain ZU1
, we concatenate e1’s head outputs from
the terms INCE(X1; Y ), INCE-CLUB(X1; X2), and INCE(X1; X2∣Y ). ZS2
and ZU2
are obtained in a
similar fashion, except we use the outputs from e2’s heads instead of e1.
FACTORCL-SSL: In the self-supervised version of CL which uses augmentations X1
′
and X2
′
of the
input modalities, the objective we aim to maximize is formulated as
LFACTORCL−SSL = INCE(X1; X2) − INCE-CLUB(X1; X2∣X1
′
, X2
′
) (97)
+ INCE(X1; X1
′
) + INCE(X2; X2
′
) (98)
− INCE-CLUB(X1; X2) + INCE(X1; X2∣X1
′
, X2
′
). (99)
23
Figure 5: An illustration of conditioning by concatenation in the implementation of FACTORCL. Conditioning
is done by concatenating Z1, the encoded representation of X1, and Z1
′
, the encoded representation of X1
′
. A
similar operation is performed for X2 and X2
′
. The concatenated vectors are then fed to MI estimators, such as
INCE and INCE-CLUB (the figure illustrates INCE).
Here the conditional terms are conditioned on the augmentations X1
′
and X2
′
, and we can similarly
model it by concatenating the head outputs of X1
′
to X1 and the head outputs of X2
′
to X2 before
feeding into the critic. We use Figure 5 to illustrate this. The way to obtain the learned representations
is the same as described in FACTORCL-SUP.
Estimation of CMI: To estimate the conditional mutual information (CMI) I(X1; X2∣X1
′
, X2
′
), we
can estimate the lower or upper bounds of true CMI [51, 55, 69]. However, direct sampling from the
conditional distribution p(x1, x2∣x
′
1
, x′
2
) can be expensive because we should consider a different
conditional distribution p(x1, x2∣x
′
1
, x′
2
) for each data pair x
′
1
, x′
2
. Sordoni et al. [69] address this by
concatenating the conditioning variable with the input in the critic: ϕ(x1, x2, c), and showing that
Conditional InfoNCE (Eq.(15) is a lower bound and estimator of CMI. This estimator can be made
more exact by further importance sampling [69]. However, adding importance sampling [69] or using
more accurate estimators [26] comes with a trade-off in complexity. Since we focus on capturing
unique information to learn a scalable multimodal representation instead of accurately estimating the
CMI, we leveraged a simpler version of the estimator from Sordoni et al. [69]: generating multiple
augmented pairs from x1, x2, and concatenating the input x1, x2 and each augmented pair x
′
1
, x′
2
to
define samples from the conditional distribution p(x1, x2∣x
′
1
, x′
2
). We argue that since augmentations
do not significantly change the semantics of images, p(x1, x2∣x
′
1
, x′
2
) could be approximated by
p(x
′′
1
, x′′
2
∣x
′
1
, x′
2
) where x
′′
1
, x′′
2
are other augmented pairs in addition to x
′
1
, x′
2
. In this submission,
we use one pair of augmented samples for consistency, but our code easily supports increasing the
number of augmented pairs that can improve the accuracy of CMI estimation.
Regardless, our existing one-pair implementation can already show that our estimators are empirically
comparable to CMI estimators with guarantees such as Mukherjee et al. [55] (Table 9), and our
estimators empirically satisfy that the lower bound is smaller than the true CMI, and the true CMI
smaller than the upper bound, i.e., Conditional InfoNCE ≤ true CMI ≤ Conditional InfoNCE-CLUB
(also in Table 9). We refer the reader to Sordoni et al. [69] for tighter bounds for CMI.
OurCL-SUP: For this ablation, we remove the factorization and only learn Z1 for X1 and Z2 for X2.
The objective we use is the same as that of FACTORCL-SUP. The only difference is that we now take
e1(x1) and e2(x2) as the learned representations for inputs x1 and x2.
OurCL-SSL: This is a similar ablation for FACTORCL-SSL where we remove the factorization.
The objective is the same as that of FACTORCL-SSL and we use e1(x1) and e2(x2) as the learned
representations for inputs x1 and x2.
Training Strategy: In regular contrastive learning using INCE as the only objective, we can simply
perform gradient descent to minimize INCE, updating all parameters in the encoders, MLP heads,
and critics. However, training any of the four methods above also involves the minimization of
the INCE-CLUB objectives, which require the optimal critic f
∗
from INCE, as stated in Eq.(11).
Therefore, within each iteration during our training, we need to first obtain the optimal critics for the
INCE-CLUB terms using the INCE objective. We outline the training strategy using a sampling method
in Algorithm 3. In this algorithm, LFACTORCL can be either LFACTORCL−SUP or LFACTORCL−SSL, and
LNCE is the summation of INCE objectives for the INCE-CLUB terms. In particular, we have
LNCE = {
INCE(X1; X2∣Y ) + INCE(X1; X2), if L = LFACTORCL−SUP ;
INCE(X1; X2∣X′
1
, X′
2
) + INCE(X1; X2), if L = LFACTORCL−SSL .
(100)
24
Algorithm 3 CL training with sampling
Require: Multimodal dataset {X1, X2}.
θ, ϕ ← Initialize network parameters.
while
{x1
not converged
,x2} ← Sampled batch from
do
{X1, X2}
θ ← Update parameters by maximizing LFACTORCL
for
{x
i =
′
1
,
1
x
to
′
2}
k
←
do
Sampled batch from {X1, X2}
ϕ ← Update parameters by maximizing LNCE
end for
end while
return θ, ϕ
We define ϕ to be the parameters of critics for the INCE-CLUB terms, and θ corresponds to all the rest
parameters in the network (parameters of encoders, heads, and critics for INCE terms). In the outer
loop, we update θ using the main learning objective L. In the inner loop, we update ϕ using the LNCE
objective, which learns the optimal critics f
∗
needed to compute the INCE-CLUB terms. Ideally in
the inner loop we would update ϕ until convergence so we get a good approximation to the optimal
critic. In practice we found sampling just one batch by setting k = 1 in Algorithm 3 works pretty
well. Using only one iteration does not have a big impact on the convergence and still produces
promising results. More importantly, it significantly reduces the time required for training, and allows
our algorithms to have comparable running time to existing contrastive learning methods.
D.2 Datasets
Gaussian datasets for MI estimation: As shown in Figure 3 in the main text, we first demonstrate
the quality of our proposed upper bounds INCE-CLUB(X1; X2) on a toy Gaussian dataset. We obtain
the samples {(xi
, yi)} from 4 multivariate Gaussian distribution with dimensions {20, 50, 100, 200}.
In each dataset, we set the ground truth MI values to be {2, 4, 6, 8, 10}, and so we can compute the
correlation ρ needed for achieving these MI values using the ground truth MI formula for Multivariate
Gaussian: I(X, Y ) = −
d
2
log(1 − ρ
2
). At each true MI value we sample 4000 times using a batch
size of 64.
Synthetic dataset with controlled generation: We generate data with controllable ratios of task￾relevant shared and unique information to analyze the behavior of each contrasive learning objective
in Figure 1 in the main text. Starting with a set of latent vectors w1, w2, ws ∼ N (0d, Σ
2
d
), d = 50
representing information unique to
[w1, ws] is transformed into high-dimensional
X1, X2 and common to both respectively, the concatenated vector
x1 using a fixed full-rank transformation T1 and
likewise [w2, ws] to x2 via T2. The label y is generated as a function (with nonlinearity and noise)
of varying ratios of ws, w1, and w2 to represent shared and unique task-relevant information. For
experiments, we used 1-layer MLPs with 512 hidden size as encoders, and the embedding dimensions
are 128 for both modalities. The heads on top of encoders are also 1-layer MLPs with the same
hidden and output dimension as the input, and all critics are 1-layer MLPs with 512 hidden size.
Multimodal fusion datasets: We use a collection of 5 real-world datasets provided in Multi￾Bench [45] and the IRFL dataset to test our method in the context of varying ratios of shared and
unique information that is important for the task. In all the datasets below, the heads added on top
of the encoders are 1-Layer MLPs with ReLU activations that map the encoder outputs to the same
dimensions. All critics are also MLPs with 1 hidden layer of size 512 and ReLU activation.
1. MIMIC-III [38] (Medical Information Mart for Intensive Care III) is a large-scale dataset for
healthcare which contains records of over 40,000 ICU patients in both forms of times-series data
measured by hours and static data (age, gender, ethnicity) in the tabular numerical form. We use
the preprocessed data provided in MultiBench [45], where the time-series data is measured every 1
hour in a 24-hour period and consists of vectors of size 12, and the tabular data consists of vectors
of size 5. The task we use in the experiment is a binary classification on whether the patient fits
any ICD-9 code in group 7 (460-519).
25
Table 6: Results on MultiBench [45] datasets with varying shared and unique information: FACTORCL achieves
strong results vs self-supervised (top 5 rows) and supervised (bottom 3 rows) baselines that do not have unique
representations, factorization, upper-bounds to remove irrelevant information, and multimodal augmentations.
Model MIMIC MOSEI MOSI UR-FUNNY MUSTARD
SimCLR [13] 66.7 ± 0.0% 71.9 ± 0.3% 47.8 ± 1.8% 50.1 ± 1.9 % 53.5 ± 2.9%
Cross+Self [81] 65.2 ± 0.0% 71.1 ± 0.2% 48.6 ± 1.2% 56.5 ± 0.7% 53.9 ± 4.5%
Cross+Self+Fact [89] 65.5 ± 0.0% 71.9 ± 0.2% 49.0 ± 1.1% 59.9 ± 0.9% 53.9 ± 4.0%
OurCL-SSL 65.2 ± 0.0% 71.2 ± 0.2% 49.0 ± 0.8% 58.8 ± 1.3% 54.0 ± 2.5%
FACTORCL-SSL 67.3 ± 0.0% 74.5 ± 0.1% 51.2 ± 1.6% 60.5 ± 0.8% 55.8 ± 0.9%
SupCon [41] 67.4 ± 0.0% 71.0 ± 0.1% 47.2 ± 1.2% 50.1 ± 2.0% 52.7 ± 2.2%
OurCL-SUP 68.2 ± 0.0% 71.1 ± 0.2% 65.3 ± 0.8% 58.3 ± 1.1% 65.1 ± 1.8%
FACTORCL-SUP 76.8 ± 0.0% 77.8 ± 0.3% 69.1 ± 0.6% 63.5 ± 0.8% 69.9 ± 1.9%
Table 7: Continued pre-training on CLIP with our FACTORCL objectives on classifying images and figurative
language. Our approach shows strong results as compared to standard fine-tuning and continued pre-training.
Task IRFL
Zero-shot CLIP [61] 89.2 ± 0.0%
SimCLR [13] 91.6 ± 0.0%
Cross+Self [81, 89] 91.1 ± 1.2%
FACTORCL-IndAug 91.6 ± 1.3%
FACTORCL-SSL 93.8 ± 1.4%
Fine-tuned CLIP [61] 96.4 ± 0.0%
SupCon [41] 87.7 ± 4.7%
FACTORCL-SUP 98.3 ± 1.2%
In the experiments, we use a 2-layer MLP with 10 hidden layer size for the tabular data modality,
and map it to a vector of size 10. The time-series modality is encoded using a GRU with hidden
size 30 and followed by a linear layer which projects the output to embeddings of size 15. We
train the model for 100 epochs using the Adam optimizer with a 1e-4 learning rate.
2. CMU-MOSEI [93] is the largest sentence-level multimodal sentiment and emotion benchmark
with 23, 000 monologue videos. It contains more than 65 hours of annotated video from more
than 1,000 speakers and 250 topics. Each video is labeled with a sentiment intensity ranging from
-3 to 3. In our experiments, we cast the intensity values to a binary classification on whether the
sentiment is positive or negative. MultiBench [45] provides access to the extracted features of the
vision, text, and audio modalities, and in our experiments, we use the vision and text features for
doing contrastive learning.
In our experiments, we encode both the vision and text modalities using Transformer encoders
with 5 heads and 5 layers, and map them to 40-dimensional embeddings. We train the model for
100 epochs using the Adam optimizer with a 1e-4 learning rate.
3. CMU-MOSI [91] is a similar dataset for multimodal sentiment analysis created from 2, 199
YouTube videos clips. The data focuses on videos that reflect the real-world distribution of
speakers expressing their opinions in the form of monologues. The sentiment intensities are
labeled continuously from -3 to 3. Again we cast the label into a binary classification on whether
the sentiment is positive or negative, and we used the extracted vision and text features for
contrastive learning.
In our experiments we encode both the vision and text modalities using Transformer encoders
with 5 heads and 5 layers, and map them to 40-dimensional embeddings. We train the model for
100 epochs using the Adam optimizer with a 1e-4 learning rate.
4. UR-FUNNY [27] is the first large-scale dataset for humor detection in human speech. The
dataset consists of samples from more than 16, 000 TED talk videos with speakers from diverse
backgrounds sharing their ideas. The laughter markup is used to filter out 8,257 humorous
punchlines from the transcripts. The context is extracted from the prior sentences to the punchline.
Using a similar approach, 8,257 negative samples are chosen at random intervals where the last
sentence is not immediately followed by a laughter marker. The task is to classify whether there is
humor or not using the vision and text modalities.
26
Table 8: Additional experiments on CIFAR10 [43] and MNIST [19] datasets using our FACTORCL objectives
on image classification.
Task CIFAR10 MNIST
SimCLR [13] 87.0% 98.84%
SupCon [41] 92.7% 99.38%
FACTORCL-SUP 91.3% 99.21%
Table 9: We verify our conditional lower and upper bound estimators on a synthetic dataset with fixed dimension
of representation dz and varying number of samples n.
Number of samples (×103
), dz = 20 5 10 20 50
CCMI (MI-Diff + Classifier) 2.03 2.06 2.15 2.20
Conditional InfoNCE 2.19 2.20 2.20 2.20
Conditional InfoNCE-CLUB 3.45 3.53 2.98 2.86
True CMI 2.32 2.32 2.32 2.32
In our experiments, we encode both the vision and text modalities using Transformer encoders
with 5 heads and 5 layers, and map them to 40-dimensional embeddings. We train the model for
100 epochs using the Adam optimizer with a 1e-4 learning rate.
5. MUSTARD [12] is a corpus of 690 videos for research in sarcasm detection from popular TV
shows including Friends, The Golden Girls, The Big Bang Theory, and Sarcasmaholics Anony￾mous. It contains audiovisual utterances together with the textual context. We use the preprocessed
features of the vision and text modalities for doing contrastive learning and performing sarcasm
detection.
In our experiments, we encode both the vision and text modalities using Transformer encoders
with 5 heads and 5 layers, and map them to 40-dimensional embeddings. We train the model for
100 epochs using the Adam optimizer with a 1e-4 learning rate.
6. IRFL [88] is a dataset for understanding multimodal figurative languages. It contains 6, 697
matching images and figurative captions (rather than literal captions) of three types of figurative
languages: idiom, simile, and metaphor. The original data for the matching task is provided in the
form of 1 caption, 3 distractor images, and 1 matching image. We convert it into a fusion task by
only collecting the matching image and text pairs and assigning labels using the type of figurative
language it belongs to.
For this dataset, we do not train from scratch. Instead, we performed continued pretraining using
our proposed objectives on pretrained CLIP [61] models. We used the CLIP-VIT-B/32 model
and its pretrained image and text encoders. We performed training for 10 epochs using the Adam
optimizer with a 1e-6 learning rate.
D.3 Additional analysis and results
Fusion experiments: In Table 6 and 7 we present more detailed results on the Multibench [45]
and IRFL [88] datasets computed from 5 independent runs. FACTORCL significantly outperforms
the baselines that do not capture both shared and unique information in both supervised and self￾supervised settings, particularly on MUSTARD (where unique information expresses sarcasm, such
as sardonic facial expressions or ironic tone of voice), and on MIMIC (with unique health indicators
and sensor readings). There are also big improvements on the two sentiment analysis datasets MOSEI
and MOSI, with 6.8% and 21.9% increases respectively when compared to SupCon [41].
In Table 7, we also see that FACTORCL substantially improves the state-of-the-art in classifying
images and figurative captions which are not literally descriptive of the image on IRFL, outperforming
zero-shot and fine-tuned CLIP [61] as well as continued pre-training baselines on top of CLIP. While
the supervised version gives the best results overall, the self-supervised version with our proposed
unique augmentations also performs better than independent augmentations, indicating that in the
case without label information, we should always try to find and use unique augmentations when
possible. In our experiments, we use word masking for text augmentations. For independent image
27
Table 10: We verify our conditional lower and upper bound estimators on a synthetic dataset with varying
dimensions of representation dz and fixed number of samples n.
Dimension dz, n = 2 × 104
1 10 20 50 100
CCMI (MI-Diff + Classifier) 2.30 2.18 2.15 1.98 1.67
Conditional InfoNCE 2.18 2.20 2.20 2.26 2.30
Conditional InfoNCE-CLUB 3.70 2.95 2.98 2.79 2.86
True CMI 2.32 2.32 2.32 2.32 2.32
Table 11: We probe whether the InfoMin assumption from Tian et al., I(Z1; Y ∣X2) = 0 and I(Z2; Y ∣X1) = 0,
is reasonable for Theorem 1. Compared to the shared information I(X1; X2), I(Z1; Y ∣X2) is much smaller
and closer to zero, indicating that the InfoMin assumption is reasonable, and Theorem 1 holds in practice.
I(X1, Y ; X2) I(X1; X2) I(Z1; Y ∣X2) I(X2, Y ; X1) I(X2; X1) I(Z2; Y ∣X1)
12.69 12.29 0.40 11.31 10.92 0.38
augmentations, we use cropping, flipping, and color jittering. The unique augmentation simply
removes the cropping operation, as illustrated in Figure 4 in the main text.
Additional experiments on high shared information and low unique information: In Table 8 we
include additional results using our method on the CIFAR10 [43] and MNIST [19] datasets. Our
method outperforms the self-supervised contrastive learning on both datasets as expected, and roughly
maintains the same performance as supervised contrastive learning. Therefore, in cases with abundant
shared information (two modalities with high shared information or two different views generated
from augmentations), our method recovers the performance of existing methods that do not capture
unique information.
Experiments on CMI estimator verification: In Table 9 and Table 10, we include experiment results
which verify that computing the conditional MI lower and upper bounds via concatenation indeed
yields reliable estimates. In particular, we aim to verify that the the Conditional InfoNCE objective
gives a lower bound of the CMI, and the Conditional InfoNCE-CLUB objective gives an upper bound
of the CMI. We follow the experiment setups in [55], presenting the true CMI and results of our
estimators on synthetic data with fixing dimension of representation Z and varying samples n, and
fixing samples n and varying dz. The specific implementations used for conditional InfoNCE and
conditional InfoNCE-CLUB can be found in Equation 13 and Equation 14, respectively. The results
indicate that our Conditional InfoNCE gives estimations smaller than the true CMI, and Conditional
InfoNCE-CLUB gives estimations greater than the true CMI. The performances are comparable to
estimators in [55], suggesting that our method yields valid and competitive lower and upper bounds
for CMI.
Empirical verification on InfoMin assumption: To verify the InfoMin assumption [72]
(I(Z1; Y ∣X2) = I(Z2; Y ∣X1) = 0), we use the same synthetic dataset as in Table 1 and measure
I(Z1; Y ∣X2). The results are shown in Table 11: we get I(X1; X2) = 12.29 and I(Z1; Y ∣X2) = 0.4.
I(Z1; Y ∣X2) is much smaller and closer to zero than I(Z1; Y ∣X2), indicating that the InfoMin
assumption holds in practice.
Compute resources: All experiments in this paper are run on a single NVIDIA A100 GPU. It takes
about 10 to 12 GPU hours to train the model on the CIFAR10 [43] for 300 epochs, and all the other
experiments can be finished within 1 GPU hour using our specified hyperparameters.
28

以上是第一篇文章的内容

---

好的，我已经仔细阅读并分析了您提供的第一篇文章《FACTORIZED CONTRASTIVE LEARNING: Going Beyond Multi-view Redundancy》。

以下是我对该文章核心思想、技术细节和主要贡献的总结，重点关注您提到的信息分解和模态增强方法。

---

### **第一篇文章总结: FACTORIZED CONTRASTIVE LEARNING (FactorCL)**

#### **1. 核心问题**

传统的 多模态对比学习（Multimodal Contrastive Learning）方法存在一个根本性的局限，即它们依赖于“**多视图冗余假设**”（Multi-view Redundancy Assumption）。该假设认为，对于下游任务来说，模态间共享的信息是必要且充分的。

然而，在许多现实世界的应用中，这个假设不成立：
*   **低共享信息**：例如，在比喻性图文对中（图片与非字面描述的文本），两个模态共享的信息很少。
*   **高独有信息**：例如，在医疗诊断中，传感器读数（模态1）和病历文本（模态2）各自包含了对诊断至关重要的、对方所没有的独有信息。

在这种情况下，只关注最大化共享信息的传统对比学习方法会丢弃任务相关的“独有信息”，从而导致性能下降。

#### **2. 理论基础与信息分解**

文章从信息论出发，将多模态数据 (`X1`, `X2`) 中与任务标签 (`Y`) 相关的总信息 `I(X1, X2; Y)` 分解为三个互斥的部分：

`I(X1, X2; Y) = S + U1 + U2`

其中：
*   **S = I(X1; X2; Y)**：**任务相关的共享信息** (Task-relevant Shared Information)，即两个模态都包含的、且对任务有用的信息。
*   **U1 = I(X1; Y | X2)**：**模态 X1 的任务相关独有信息** (Task-relevant Unique Information in X1)，即在已知模态 X2 的情况下，X1 仍然能为任务提供的额外信息。
*   **U2 = I(X2; Y | X1)**：**模态 X2 的任务相关独有信息** (Task-relevant Unique Information in X2)。

传统对比学习只优化 `S`，而 FactorCL 的目标是同时学习并利用 `S`, `U1`, 和 `U2`。

#### **3. FactorCL 方法的核心思想**

为了捕获上述三种信息，FactorCL 提出了一个全新的框架，包含三个关键创新：

**a) 表示分解 (Representation Factorization):**
模型不再为每个模态学习单一的表示，而是学习四种分解后的表示：
*   `ZS1` 和 `ZS2`：分别从 `X1` 和 `X2` 中提取的**共享信息**表示。
*   `ZU1`：从 `X1` 中提取的**独有信息**表示。
*   `ZU2`：从 `X2` 中提取的**独有信息**表示。

**b) 通过互信息上下界进行优化:**
为了得到更纯粹的表示，FactorCL 不仅要**捕获**任务相关信息，还要主动**移除**任务无关信息。这是通过同时优化互信息（MI）的**下界**（最大化）和**上界**（最小化）来实现的：
*   **捕获信息**：通过最大化互信息的**下界**（如 InfoNCE 损失 `INCE`）来鼓励表示学习到目标信息。
*   **移除信息**：通过最小化互信息的**上界**（文章提出了一个新的、更紧的 `INCE-CLUB` 上界）来剔除无关信息。这个上界估计器可以“免费”地从 `INCE` 的优化过程中获得，非常高效。

**c) 通过多模态增强实现自监督学习:**
在没有任务标签 `Y` 的自监督场景下，如何定义“任务相关性”是关键。FactorCL 的核心贡献在于使用**数据增强**来近似 `Y`。

#### **4. 核心机制：模态增强的应用**

这是 FactorCL 最具特色的部分，它解决了如何在自监督设定下有意义地分解信息的问题。

**a) 从监督到自监督的转换:**
在监督学习中，可以直接使用标签 `Y` 来计算条件互信息项，例如 `I(X1; X2 | Y)`（任务无关的共享信息）。在自监督学习中，FactorCL 用精心设计的**增强视图** (`X1'`, `X2'`) 来替代 `Y`。其理论基础是“最优增强假设”：一个好的增强应该保留所有任务相关信息，同时改变任务无关信息，因此 `I(X; X') ≈ I(X; Y)`。

**b) 提出“最优多模态增强” (Optimal Multimodal Augmentation):**
文章指出，简单地对每个模态独立进行标准增强是不够的。为了准确地近似条件互信息 `I(X1; X2 | Y)`，需要一种联合的增强策略。FactorCL 提出了一个两步增强过程：

1.  **单模态增强 (Unimodal Augmentation):** 对模态 `X1` 进行增强得到 `X1'`，使其满足 `I(X1; X1') ≈ I(X1; Y)`。
2.  **独有信息增强 (Unique Augmentation):** 对模态 `X2` 的增强**需要考虑 `X1` 的信息**。具体来说，在增强 `X2` 得到 `X2'` 时，应避免破坏或改变那些与 `X1` 共享的信息。

**示例（来自论文图4）：**
*   **文本 (X1):** "这辆车很快。"
*   **图片 (X2):** 一辆在高速公路上行驶的汽车。
*   **不好的增强:** 对图片进行随机裁剪，可能会把背景中的“高速公路”裁掉。而“高速公路”是体现“快”这个共享信息的关键视觉元素。
*   **好的独有信息增强:** 只进行翻转、色彩抖动等不影响“高速公路”这个关键元素的增强。这样，模型在学习 `X2` 的独有信息时，增强过程不会干扰到与 `X1` 共享的部分，从而更精准地分离出独有信息。

通过这种方式，`I(X1; X2 | X1', X2')` 就能更好地近似任务无关的共享信息 `I(X1; X2 | Y)`，使得整个信息分解框架在自监督下变得可行且有意义。

#### **5. 总结与主要贡献**

*   **问题定义**: 指出了传统对比学习在“多视图非冗余”场景下的“独有性差距”（uniqueness gap）问题，并用信息论进行了形式化定义。
*   **框架创新**: 提出了一个全新的 **FactorCL** 框架，通过**分解表示**来显式地学习共享和独有信息。
*   **优化方法**: 创造性地结合了互信息的**下界（捕获）**和**上界（移除）**进行优化，从而学习到更纯粹的表示。
*   **核心机制**: 提出了“**最优多模态增强**”和“**独有信息增强**”的概念，为在自监督条件下进行有效的信息分解提供了理论和实践指导。这是将信息分解理论应用于现实多模态学习的关键桥梁。

---

现在我已经对第一篇文章有了深入的理解。请您提供第二篇文章，我将在此基础上进行详细的比较分析。

---

Published as a conference paper at ICLR 2025
WHAT TO ALIGN IN MULTIMODAL CONTRASTIVE
LEARNING?
Benoit Dufumier1,2∗ Javiera Castillo Navarro1,3∗ Devis Tuia1 Jean-Philippe Thiran1,4
1 EPFL 2 NeuroSpin, CEA 3 CEDRIC, CNAM 4 Radiology Department, CHUV
ABSTRACT
Humans perceive the world through multisensory integration, blending the infor￾mation of different modalities to adapt their behavior. Alignment through con￾trastive learning offers an appealing solution for multimodal self-supervised learn￾ing. Indeed, by considering each modality as a different view of the same entity,
it learns to align features of different modalities in a shared representation space.
However, this approach is intrinsically limited as it only learns shared or redun￾dant information between modalities, while multimodal interactions can arise in
other ways. In this work, we introduce CoMM, a Contrastive Multimodal learn￾ing strategy that enables the communication between modalities in a single multi￾modal space. Instead of imposing cross- or intra- modality constraints, we propose
to align multimodal representations by maximizing the mutual information be￾tween augmented versions of these multimodal features. Our theoretical analysis
shows that shared, synergistic and unique terms of information naturally emerge
from this formulation, allowing to estimate multimodal interactions beyond re￾dundancy. We test CoMM both in a controlled and in a series of real-world set￾tings: in the former, we demonstrate that CoMM effectively captures redundant,
unique and synergistic information between modalities. In the latter, we show
that CoMM learns complex multimodal interactions and achieves state-of-the-art
results on seven multimodal tasks. Code is available here.
1 INTRODUCTION
Multisensory or multimodal learning (Baltrusaitis et al. ˇ , 2018) involves extracting and processing
information from multiple sources (e.g. text, audio, images, tabular data, etc.). The whole human
experience is inherently multimodal: we simultaneously see, hear, smell, taste and feel, and these
different sensory signals are combined to give us the necessary information to explore our environ￾ment. Many of the simplest tasks we tackle in our daily lives are multimodal:the way we perceive
the flavor of our food or drinks does not depend solely on our taste, but also on what we see (Morrot
et al., 2001) or what we hear (Woods et al., 2011) while we eat. McGurk & MacDonald (1976) have
also shown that visual stimuli interact with audio signals to perform human speech recognition.
Despite the inherent multimodality of sensory systems, machine learning has largely concentrated
on single-modality models, with few exceptions in areas like audio-visual speech recognition (Yuhas
et al., 1989; Ngiam et al., 2011), multimedia content retrieval (Atrey et al., 2010; Snoek & Worring,
2005), and video-based human behavior analysis (Kraaij et al., 2005). Nowadays, with the emer￾gence of self-supervised strategies and their impressive capacities for learning representations in
computer vision (Chen et al., 2020a; He et al., 2020; Caron et al., 2021), NLP (Radford et al., 2018;
Devlin et al., 2019) or audio (Oord et al., 2018; Niizumi et al., 2021), the paradigm has shifted
to learning general multimodal representations from unlabeled data and then fine-tune the models
to specific multimodal tasks. Recent works have shown success at training multimodal represen￾tations by using cross-modal contrastive objectives (Radford et al., 2021; Jia et al., 2021) to align
the representations in a shared embedding space. However, this training strategy only works under
the multiview redundancy assumption, i.e., considering that all task-relevant information is shared
between modalities and redundantly contained in either one of them separately. In particular, for
vision-language tasks, this can be seen as a clever way to perform supervised learning on a visual
encoder, which explains their success to transfer to visual classification tasks.
∗
denotes equal contribution. Contact information: {name.surname}@epfl.ch
1
arXiv:2409.07402v2 [cs.LG] 5 Mar 2025
Published as a conference paper at ICLR 2025
a) Multimodal models b) Multimodal interactions
Figure 1: a) We propose CoMM, a contrastive multimodal approach that allows the interplay
of multiple modalities and learns multimodal interactions. Unlike previous multimodal models
(Cross) that align cross-modal features, CoMM aligns multimodal features in a shared representation
space. b) Multimodal interactions are task-dependent, thus a model needs to capture all of them to
generalize to any multimodal task. CoMM’s new paradigm captures multimodal interactions
beyond redundancy.
Nonetheless, these solutions are insufficient in many cases, as the interactions between modalities
can arise in several ways to perform a specific task (Bertschinger et al., 2014): redundancy (R)
arises when the task can be performed using either of the modalities because they contain redundant
information; uniqueness (U) appears when only one of the modalities contains all the necessary
information to complete the task; synergy (S) emerges when both modalities have complementary
information, and they are needed simultaneously to fulfill the task. Modeling these interactions to
perform multimodal learning is highly challenging as the interplay between R, S and U is task￾dependent and difficult to measure in complex real-life scenarios. Fig. 1b shows simple tasks where
the predominant type of interaction can be easily identified. We observe that we need to model
specific interactions for the same input modalities to perform a specific task. Therefore, a model
must capture all these terms to learn task-agnostic multimodal representations.
To achieve task-agnostic multimodal learning, we propose a Contrastive MultiModal self￾supervised pre-training method (CoMM) that enables the communication of modalities in a single
multimodal space. Unlike previous contrastive multimodal methods that impose cross-modal con￾straints to align unimodal representations, we propose to leverage a simple multimodal architecture
to fuse multimodal inputs into a common representation and then align the multimodal features by
maximizing the mutual information between augmented versions of these features (see Fig. 1a).
CoMM enables to model multimodal interactions –including redundancy, uniqueness and synergy–
in the context of multimodal representation learning, as these terms naturally emerge from our con￾trastive multimodal formulation.
CoMM’s formulation is well-aligned with the global workspace theory (Baars, 1988; Goyal & Ben￾gio, 2022) in cognitive neuroscience, which considers the nervous system as a set of multiple spe￾cialized processors working in parallel and claims the existence of a shared representation, which
can be modified by any selected processor and whose content is broadcast to all processors. By
analogy, CoMM considers a shared representation space built from parallel streams of modality￾specific processors. When a task requires knowledge from a given modality or interactions between
modalities, only these parts of the representation space should be used.
Based on Partial Information Decomposition (Williams & Beer, 2010; Bertschinger et al., 2014), we
built a strong theoretical basis for CoMM to learn multimodal interactions in a self-supervised way.
Empirically, we show that CoMM effectively captures redundant, unique and synergistic information
between modalities in a controlled environment, where the type of interaction is known.
Then, in a series of real-world datasets –from different domains (healthcare, robotics, etc.) and
including diverse data types (image, text, audio, etc.)–, CoMM achieves state-of-the-art results on
seven multimodal tasks with two or three modalities. In all cases, CoMM showed to be a versatile
framework capable of handling any number of modalities, various data types, and different domains.
2
Published as a conference paper at ICLR 2025
2 QUANTIFYING MULTIMODAL INTERACTIONS
Problem setup. Let X1, X2, . . . , Xn be random variables representing n different data modalities
(e.g. images, text, audio, tabular data, etc.), and a given task Y . Our goal is to learn a latent variable
Z = f(X) that is a good representation of X = (X1, . . . , Xn) for Y .
For our theoretical analysis, we set n = 2, as multimodal interactions have not been characterized
yet by PID for larger n. In practice, CoMM’s implementation for n > 2 is straightforward and tested
in Section 4.3. All proofs can be found in Appendix G.
For Z to be a good representation of X it should capture the task-relevant information that X
contains. Therefore, we need to model the information between the joint variable X and the task Y :
I(X; Y ) = I(X1, X2; Y ).
Partial information decomposition (PID) (Williams & Beer, 2010; Bertschinger et al., 2014) states
that multivariate mutual information I(X1, X2; Y ) is decomposed into three forms of interactions:
(i) Uniqueness. This term appears when the task Y can be completed by leveraging only one of
the modalities. U1 (resp. U2) refers to the case when X1 (resp. X2) contains all task-relevant
information.
(ii) Redundancy. When X1 and X2 contain the same information about Y . R corresponds to this
redundant or shared information.
(iii) Synergy. Noted by S, this term only emerges when X1 and X2 are simultaneously present,
because they bring different and complementary task-relevant information.
Thus, the information that (X1, X2) has about Y can be written as the contribution of four terms:
I(X1, X2; Y ) = R + S + U1 + U2 (1)
Moreover, Eq. (1) and the application of the chain rule of mutual information yield the following
consistency equations between R, S, U1 and U2:
I(X1; Y ) = R + U1, I(X2; Y ) = R + U2, I(X1; X2; Y ) = R − S (2)
Existing methods using contrastive objectives to learn multimodal representations (Jia et al., 2021;
Radford et al., 2021; Tian et al., 2020a) impose cross-modal constraints by maximizing an estima￾tor of I(X1; X2) as an approximation of I(X1, X2; Y ). However, this strategy is limited by the
multiview redundancy assumption.
Definition 1 (Multi-view redundancy) ∃ ε > 0 such that I(Y ; X1|X2) < ε and I(Y ; X2|X1) < ε.
This assumption states that most task-relevant information is shared across modalities and the non￾shared information is (at most) a small ε. In other words, any of the modalities contains enough
information to fulfill the downstream task Y , and they can provide some kind of “supervision” to
one another, which explains their success for zero-shot image classification (El Banani et al., 2023).
Lemma 1 Under the multiview redundancy assumption, cross-modal contrastive learning methods
are limited to only learn the redundant information R.
What happens when other sources of multimodal information intervene? FactorCL (Liang et al.,
2023b) is a good initial step to integrate uniqueness and redundancy into multimodal contrastive
learning by applying multimodal augmentations. However, it heavily relies on the assumption that
optimal multimodal augmentations can be obtained by applying a two-step process based on condi￾tional augmentations. We argue that this hypothesis is unrealistic, as the first unique augmentation
is strongly related to task-relevant information of different modalities. For example, if the text cap￾tion is “a yellow flower”, then color jittering should not be applied to an image depicting a flower.
Besides, the factorized formulation of FactorCL is impractical as it is prone to cumulative errors.
Finally, this method does not consider the synergy between modalities.
In contrast, we propose a model that relies solely in the main hypothesis of contrastive learning,
extended to the multimodal case, without relying on strong assumptions about multimodal relation￾ships nor conditional augmentations.
Assumption 1 (Minimal label-preserving multimodal augmentations) We assume the existence of
T
⋆
, a set of multimodal augmentations such that for any t ∈ T ⋆ and X′ = t(X), we have
I(X, X′
) = I(X, Y ).
3
Published as a conference paper at ICLR 2025
Even if Assumption 1 might seem strong at first glance, it actually makes sense in the context of
multimodal representation learning. Indeed, coming back to the example of the flower image with
“a yellow flower” as a caption, applying color jittering to the image would allow the model to focus
on other interactions (uniqueness or synergy) rather than color redundancy or even to refocus on
other features (like the flower shape). This is discussed more in-depth in Appendix C.4.
Moreover, our assumption allows for a larger spectrum of augmentations, without being constrained
to the set Tc
⋆ = {t(X) = (t1(X1), t2(X2))} of transformations that can be decomposed in indepen￾dent unimodal augmentations.
3 COMM: CONTRASTIVE MULTIMODAL LEARNING
We aim to learn multimodal representations that are transferable to any multimodal task. Contrastive
learning has shown promising results in multimodal learning. However, current approaches fail to
capture multimodal interactions other than redundancy, as shown in Section 2.
Our strategy builds upon multiview contrastive learning theory and extends it to the multimodal
case. It is based on two main components:
(i) A multimodal architecture, with specialized encoders to process any data type, and an
attention-based fusion module to obtain a final multimodal representation.
(ii) A contrastive objective that naturally captures unique, redundant, and synergistic interactions
between different data modalities.
3.1 TOWARDS EFFECTIVE MULTIMODAL REPRESENTATIONS
To obtain robust, task-agnostic and common representations Z that capture uniqueness, redundancy
and synergy from different input modalities, we design fθ –a neural network parameterized by θ–
such that Zθ = fθ(X) = fθ(X1, X2).
We define X′ = t(X) with t ∈ T a stochastic mapping1
(multimodal augmentation) of X and
Zθ
′ = fθ(X′
).
Given data processing inequalities for Markov chains X → X′ → Zθ
′ and Zθ
′ → X → Zθ, we have:
I(Zθ;Zθ
′
) ≤ I(X, Zθ
′
) ≤ I(X, X′
) (3)
With these inequalities, we can prove the following lemmas:
Lemma 2 By optimizing fθ to maximize I(Zθ;Zθ
′
), and if we assume an expressive enough network
fθ, we have at optimum:
I(Zθ
⋆ , Zθ
′
⋆ ) = I(X, X′
) (4)
Lemma 3 Let fθ
⋆ be optimal, i.e. fθ
⋆ maximizes I(Zθ, Zθ
′
). Then, we have the equality
I(Zθ
′
⋆ ; Y ) = I(X′
; Y ). If we consider the special case T = {ti} such that X′ = ti(X) = Xi
and Zθ
′
⋆ = fθ
⋆ (Xi) = Zi
for i ∈ {1, 2}, then it follows:
I(Zi
; Y ) = I(Xi
; Y ) = R + Ui (5)
Lemma 3 implies that optimal representations Zi preserve all the task-relevant information contained
in modality i. Interestingly, we do not require Assumption 1 for this equality to hold.
The previous theoretical developments lead us to the key ingredients for CoMM’s contrastive objec￾tives to succeed at capturing multimodal interactions (see Section 3.3 for practical implementation):
(i) Following Lemma 2, U1 + U2 + R + S = I(X, Y ) can be learned by optimizing the term
I(Zθ, Zθ
′
) for T = T
⋆
, since I(X, X′
) = I(X, Y ) by Assumption 1;
(ii) Thanks to Lemma 3, R + Ui for i ∈ {1, 2} can be directly learned by optimizing the term
I(Zθ, Zθ
′
) for T = {ti}.
1Here, T can be any set of mappings, e.g., T ̸= T
⋆
.
4
Published as a conference paper at ICLR 2025
3.2 MULTIMODAL ARCHITECTURE
Figure 2: CoMM’s model architecture. Inputs from dif￾ferent modalities X = (X1, ..., Xn) are first encoded by
modality-specific encoders. Modality-specific features are
processed by latent converters to map them into sequences
of embeddings which are concatenated and fused by a trans￾former block. The output is a single multimodal feature Zθ.
Our architecture for multimodal rep￾resentation learning is presented in
Fig. 2. In order to capture multi￾modal interactions, the model con￾sists of mainly three components:
Encoders. Each modality is en￾coded independently by one of the
n modality-specific encoders.
Latent converters. Linear mod￾ules that transform features into
modality-specific sequences of em￾beddings. After the latent convert￾ers, a concatenation operation gath￾ers these sequences to be fed into a
transformer architecture.
Transformer block. The goal of
this module is to perform the fusion of the modality-specific embeddings through the multihead
self-attention layers in the transformer block, obtaining the final multimodal embedding Z.
More information about the specific modules used as modality encoders, about the latent converters
and the transformer block architecture can be found in Appendix B.
3.3 TRAINING
Given a multimodal input X = (X1, ..., Xn) and a set of label-preserving multimodal transfor￾mations T
⋆
, two augmentations t
′
, t′′ are drawn from T
⋆
and applied to X, obtaining X′ =
t
′
(X) and X′′ = t
′′(X). We also consider projections (with a slight abuse of notation) Xi =
([MSK], ..., Xi
, ..., [MSK]) for i ∈ {1, ..., n}, where every modality is masked except for the i-th.
These terms are encoded by the network to obtain (n+2) embeddings, namely: Z
′
, Z′′, and {Zi}
n
i=1.
(2n + 1) mutual information terms are then optimized through backpropagation: I(Z
′
, Z′′) to max￾imize I(Z, Z′
) in Eq. (3) and both I(Zi
, Z′
) and I(Zi
, Z′′) to better approximate R + Ui
in Eq. (5)
for i ∈ {1, ..., n}.
We use the InfoNCE (Oord et al., 2018) estimator of mutual information due to its simplicity and
strong results in the self-supervised learning literature:
ˆINCE(Z, Z′
) = Ez,z′
pos∼p(Z,Z′
)
z
′
neg∼p(Z
′
)
"
log
exp sim(z, z′
pos)
P
z
′
neg
exp sim(z, z′
neg)
#
(6)
Given this estimator, our final training loss can be written as:
LCoMM = − ˆINCE(Z
′
, Z′′)
≈
|R+S+
{z
P n
i=1 U
}i
−
nX
i=1
1
2

I
ˆ
NCE(Zi
, Z′
) + I
ˆ
NCE(Zi
, Z′′)

|
{z
}
≈R+Ui
=: L +
nX
i=1
Li (7)
Fig. 3 illustrates CoMM’s training process for the case n = 2. The pseudocode for the general case
n ≥ 2 is available in Appendix D. It is worth to notice that the loss terms in Eq. (7) grow linearly
with the number of modalities n.
At inference, no augmentations are applied. The multimodal input X = (X1, ..., Xn) is processed
through the network to obtain the multimodal feature Zθ = fθ(X). This multimodal representation
can then be directly transferred to any task either by performing linear probing or by fine-tuning the
whole architecture.
4 EXPERIMENTS
We design different sets of experiments to evaluate our model: first, over a controlled environment
inspired by the Trifeature dataset (Hermann & Lampinen, 2020), we carefully formulate tasks that
5
Published as a conference paper at ICLR 2025
Figure 3: CoMM training for n = 2. Two multimodal augmentations are applied to X to obtain
X′
and X′′. We also consider the projection operators to get {Xi}
n
i=1. These n + 2 transformed
versions of X are processed by the network fθ, trained to maximize the agreement between these
n + 2 terms using contrastive objectives.
need a specific kind of interaction (uniqueness, redundancy, or synergy) to assess the model’s ability
to capture them separately. Second, in order to evaluate the capacities of the model to learn multi￾modal representations for complex tasks requiring different levels of shared, unique, and synergistic
information, we use real-world benchmark datasets with 2 or 3 modalities.
Experimental settings. We report mean and standard deviation over 5 runs for all our results (except
when using public model weights). All hyper-parameter details can be found in Appendix B.
Evaluation. Given a pre-trained model f, we perform evaluation through linear probing, i.e., we
train a linear layer gW (x) = W f(x) (f fixed) to minimize the classification or regression loss
(depending on the task). We also report results obtained with fine-tuning, i.e. after further training f
in a supervised way.
4.1 CONTROLLED EXPERIMENTS ON THE BIMODAL TRIFEATURE DATASET
Figure 4: Linear probing accuracy of redun￾dancy (shape), uniqueness (texture) and synergy
(color and texture) on bimodal Trifeature dataset.
CoMM is the only model capturing all three task￾related interactions between modalities.
To evaluate whether CoMM learns redundant,
synergistic and unique information for a given
task, we design a synthetic bimodal dataset
based on Trifeature (Hermann & Lampinen,
2020). Briefly, we generate a Trifeature dataset
(as first modality) containing images of one of
ten shapes, for one of ten textures and one of
ten colors (1 000 combinations in total). We
augment each image three times using rotations
and translations. Then, we pair each image
with a second one (as second modality) from
the same Trifeature dataset, allowing us to con￾trol the shared, unique and synergistic attributes
between the two modalities. As Hermann &
Lampinen (2020), we use AlexNet (Krizhevsky
et al., 2012) as modality-specific encoder for
both modalities in all experiments with an embedding dimension d = 512.
Experiment 1 – Modeling shared and unique interactions. We choose the shape as shared fea￾ture and texture as unique feature for each modality by selecting only paired images with the same
shape. Training and test set follow the same distribution, with 10 000 and 4 096 images, respectively.
We measure the linear probing accuracy of shape (respectively texture) to test whether redundancy
(resp. uniqueness) has been captured in the latent representation of the model (chance level=10%).
Experiment 2 – Modeling synergy. In the previous experiment, texture and color features are in￾dependent between the two modalities. To introduce a synergy between these features, we bias the
training set by defining a mapping M between the ten textures and ten colors (e.g. stripes=red,
dots=green, etc.). Then, we select randomly 10 000 pairs of images that respect this mapping,
thus artificially introducing a strong correlation between texture from the first modality and color
from the second modality in the training set. The test set is left unchanged from previous Exper-
6
Published as a conference paper at ICLR 2025
Model Regression Classification
V&T EE ↓ MIMIC ↑ MOSI ↑ UR-FUNNY ↑ MUsTARD ↑ Average∗ ↑
Cross†
(Radford et al., 2021) 33.09±3.67 66.7±0.0 47.8±1.8 50.1±1.9 53.5±2.9 54.52
Cross+Self†
(Yuan et al., 2021) 7.56±0.31 65.49±0.0 49.0±1.1 59.9±0.9 53.9±4.0 57.07
FactorCL†
(Liang et al., 2023b) 10.82±0.56 67.3±0.0 51.2±1.6 60.5±0.8 55.80±0.9 58.7
CoMM (ours) 4.55±0.30 66.4±0.32 67.5±1.30 63.1±0.65 63.9±3.01 65.22
SupCon†
(Khosla et al., 2020) - 67.4±0.0 47.2±1.2 50.1±2.0 52.7±2.2 54.35
FactorCL-SUP†
(Liang et al., 2023b) 1.72±0.03 76.8±0.0 69.1±0.6 63.5±0.8 69.9±1.9 69.82
CoMM (fine-tuned) 1.34±0.01 68.18±0.23 74.98±0.43 65.96±0.44 70.42±0.15 69.88
Table 1: Linear evaluation top-1 accuracy (in %) for classification tasks and MSE (×10−4
) for
regression task (V&T End Effector) on MultiBench after 100 epochs. †Results obtained from (Liang
et al., 2023b). ∗ Average is taken over classification results only. Rows in color are supervised.
iment 1 and the task Y is to detect whether a given pair of images respects the mapping M, i.e.
Y = 1(texture(X1), color(X2) ∈ M) (chance level=50%).
In Fig. 4, we show that cross-modality constraints with InfoNCE (Radford et al., 2021) (“Cross”
model) allow to perfectly capture redundant information but completely remove unique and syn￾ergistic information, as predicted in Lemma 1. Self-supervised constraints on each encoder
(“Cross+Self” (Yuan et al., 2021)) capture accurately unique information but fail at preserving syn￾ergy, FactorCL, the method most closely related to our work, also performs poorly on synergy.
CoMM is the only model learning all three interactions related to the task.
4.2 EXPERIMENTS WITH 2 MODALITIES ON REAL-WORLD DATASETS
MultiBench. Following (Liang et al., 2023b), we use a subset of real-world multimodal datasets
from MultiBench (Liang et al., 2021), with different degrees of shared and unique task-relevant in￾formation, including: Vision&Touch (Lee et al., 2020) a robotics dataset that includes images, force,
and proprioception data for end-effector position prediction (regression) and contact prediction (bi￾nary classification), MIMIC (Johnson et al., 2016), a dataset for mortality and disease prediction
from medical records, including tabular data and medical time series from ICU; MOSI (Zadeh
et al., 2016), a dataset for sentiment analysis from videos (vision, audio, and language); UR￾FUNNY (Hasan et al., 2019), humor detection from videos (vision, audio and language); and MUs￾TARD (Castro et al., 2019), a dataset for sarcasm detection from TV shows (vision, audio, and
language). For fair comparisons, we run our experiments using the same data pre-processing steps
as previous works, using pre-extracted text, video and audio features for training (Liang et al., 2021;
2023b). More details about these datasets and the data pre-processing can be found in Appendix E.
We train CoMM on the same data modalities as FactorCL and use the same backbone networks.
We consider two different experimental settings. In the self-supervised setting, we perform pre￾training and evaluate with linear probing. We consider FactorCL, “Cross” and “Cross+self” methods
for comparison. In the fine-tuning setting, CoMM is fully fine-tuned in a supervised way after pre￾training. We compare it against SupCon and FactorCL-Sup as supervised methods.
Results for these experiments are in Table 1. In the self-supervised experiments with linear probing
evaluation, CoMM surpasses FactorCL (second best) by large margins (16.3%, 2.6% and 8.1% of
top-1 accuracy on MOSI, UR-FUNNY and MUsTARD, respectively) on three out of four classifi￾cation datasets. On MIMIC, margins are considerably narrower, with FactorCL performing slightly
better, and CoMM showing comparable results with “Cross” and “Cross+Self”. In the regression
task of V&T, CoMM is considerably better than competitors (3×10−4
lower MSE than second best).
In the fine-tuning scenario, we observe the same pattern, with CoMM outperforming competitors on
four datasets and FactorCL taking the lead on MIMIC.
These experiments not only show CoMM’s efficiency at learning multimodal representations, but
also exhibit CoMM’s versatility to process different data domains (time series, audio, images, text,
etc.) and to adapt to diverse backbones.
7
Published as a conference paper at ICLR 2025
Model Modalities weighted-f1 macro-f1
SimCLR†
(Chen et al., 2020a) V 40.35±0.23 27.99±0.33
CLIP (Radford et al., 2021)
V 51.5 40.8
L 51.0 43.0
V+L 58.9 50.9
BLIP-2 (Li et al., 2023) V+L 57.4 49.9
SLIP†
(Mu et al., 2022) V+L 56.54±0.19 47.35±0.27
CLIP†
(Radford et al., 2021) V+L 54.49±0.19 44.94±0.30
CoMM (ours, CLIP backbone) V+L 61.48±0.18 54.63±0.22
CoMM (ours, BLIP-2 backbone) V+L 64.75±0.17 58.44±0.43
MFAS (Perez-R ´ ua et al. ´ , 2019) V+L 62.50 55.6
ReFNet (Sankaran et al., 2022) V+L - 56.7
CoMM‡
(ours, CLIP backbone) V+L 64.90±0.21 58.97±0.19
CoMM‡
(ours, BLIP-2 backbone) V+L 67.39±0.07 62.0±0.25
LLaVA-NeXT (Li et al., 2024) V+L 64.28 56.51
Table 2: Linear evaluation F1-score (weighted and macro)
(in %) on MM-IMDb after 70 epochs. † indicates further
training on unlabeled data. ‡ means supervised fine-tuning.
Rows in color are supervised.
Multimodal IMDb (MM-IMDb)
(Arevalo et al., 2017) is a dataset for
movie genre prediction. We consider
two modalities: images and text (the
movie poster and its plot’s description,
respectively). Since each movie can
be classified into one or more genres,
it is a multi-label classification task,
with 23 categories. MM-IMDb pro￾vides a suitable example of life-like
multimodal task as the genre predic￾tion cannot be performed accurately
from the movie poster or the movie
plot alone, while results significantly
improve by considering both (Arevalo
et al., 2017), suggesting that syn￾ergistic interactions are needed to
fulfill this task. We compare CoMM’s
performance on MM-IMDb against
important baselines under two different settings. First, in the self-supervised setting, we consider
CLIP (Radford et al., 2021), representing “Cross” methods, SLIP (Mu et al., 2022) representing
“Cross+Self” methods, SimCLR (Chen et al., 2020a) for unimodal self-supervised methods,
BLIP-2 (Li et al., 2023) as a recent powerful vision and language model, as baselines. All models
were trained on unlabeled MM-IMDb. For CLIP, we also report results with the publicly released
weights without further training. CoMM is initialized with pre-trained weights from CLIP and
BLIP-2. Second, in the fine-tuning setting, we compare CoMM fine-tuned with state-of-the-art
fully supervised baselines. We also include results for LLaVA-NeXT (Li et al., 2024) representing
new vision-and-language generative models, however these scores are not fully comparable to
ours, since the model might have seen the IMDb database during training. See Appendix B for
implementation details.
Table 2 shows that CoMM outperforms all models in both settings. In the self-supervised setting,
CoMM has a margin of 7.5% and 5.8% of macro and weighted F1-scores, respectively, with the
second-best method. It is interesting to observe that CLIP performs better with the publicly released
weights (probably because of the large-scale and diversity of the data it was trained on), than with
further training on MM-IMDb. This result suggests that the genre movie prediction does not benefit
from learning redundant information. Including uniqueness from the image modality allows for
some improvement (SLIP). In the fine-tuning setting, CoMM fine-tuned outperforms existing fully
supervised baselines, even if MFAS has been designed to search for the best fusion strategy, and
vision-language generative models (LLaVA-NeXT).
4.3 EXPERIMENTS WITH 3 MODALITIES ON REAL-WORLD DATASETS
Model #Mod. V&T CP UR-FUNNY
Cross 2 84.4 50.1
Cross+Self 2 86.8 59.9
CoMM (ours) 2 88.1 63.1
CMC (Tian et al., 2020a) 3 94.1 59.2
CoMM (ours) 3 94.2 64.6
Table 3: Linear evaluation top-1 accuracy (%) on
Vision&Touch and UR-FUNNY.
We test CoMM’s abilities to learn multimodal
interactions beyond 2 modalities. We perform
experiments on two large datasets including tri￾modal data from MultiBench: Vision&Touch
(contact prediction task) and UR-FUNNY.
In Table 3, we compare CoMM trained on
the three modalities in a self-supervised way
against CMC (Tian et al., 2020a). We also com￾pare with bi-modal models: CoMM, Cross and
Cross+Self trained on image and propriocep￾tion data for Vision&Touch and image and text data for UR-FUNNY.
First, we observe a consistent improvement (+6.1% on V&T, +1.5% on UR-FUNNY) when adding
a third modality with CoMM (compared to only using two), which demonstrates its versatility.
Second, we improve the state-of-the-art for SSL methods on datasets with more than two modalities
(+0.1% and +5.4% on V&T and UR-FUNNY, respectively).
8
Published as a conference paper at ICLR 2025
Figure 5: Linear probing accuracy of redundancy R, uniqueness U = n
1 P n
i=1 Ui and synergy S on
bimodal Trifeature when optimizing each term separately in LCoMM. Minimizing Li allows to learn
Ui and R, approximating I(Xi
; Y ) for i ∈ {1, ..., n}. Optimizing L = −I
ˆ(Z
′
, Z′′) allows to slowly
learn R, Ui and S. CoMM quickly captures all information.
5 ABLATION STUDIES
We test three main components of our framework (the loss, fusion module and augmentation strat￾egy) against important control baselines on bimodal Trifeature dataset (see Section 4.1).
Loss function. First, we check our claim that optimizing both L and P n
i=1 Li –as in Eq. (7)–
is required to accurately capture uniqueness, synergy, and redundancy. In Fig. 5, we show that
minimizing P n
i=1 Li
improves redundancy and uniqueness, as guaranteed by our Lemma 3, but
fails for synergy. Conversely, minimizing L allows one to learn all information terms but very
slowly. In particular, for synergy, we argue this is because the model has to learn modality-specific
features first (phase 1) before learning their interactions (phase 2). The former is learned through
I(Zi
, Z′
) + I(Zi
, Z′′) while the latter is captured with I(Z
′
, Z′′). Hence, LCoMM speeds up phase
1 and can learn synergy more efficiently in phase 2.
Fusion Redundancy Uniqueness Synergy Average
Concat + Linear 99.71±0.06 81.49±2.88 50.0±0.0 77.07
CoMM 99.92±0.03 87.83±1.55 71.87±2.06 86.83
Table 4: Linear fusion module versus attention-based fu￾sion with latent converters (CoMM). Non-linearities are re￾quired to learn synergistic interactions between modalities.
Fusion module. We compare our
attention-based latent fusion module
with shallow linear fusion. We project
modality-specific representations to
the common latent space using only
linear layers and remove latent con￾verters. Table 4 shows that synergy is
not captured with linear fusion. This is
expected as the XOR gate (typical example of synergistic interactions) cannot be approximated by
a linear function. Additionally, uniqueness accuracy is also degraded compared to CoMM (−6%),
suggesting that only modeling linear interactions limits the model’s representation power.
Augmentations R U1 U2 S Average
Modality 1 Modality 2
{All} ∅ 99.72±0.04 79.92±1.05 46.44±3.74 50.0±0.0 69.02
∅ {All} 99.58±0.13 53.89±6.69 86.04±2.01 50.0±0.0 72.37
{All}\{crop} {All} 88.79±0.96 25.65±0.10 84.00±4.43 50.0±0.0 62.11
{All} {All}\{crop} 90.50±2.07 83.22±2.86 21.74±1.36 50.0±0.0 61.36
CoMM 99.92±0.03 84.35±2.37 91.19±0.97 71.87±2.06 86.83
Table 5: Effect of data augmentation on linear probing accuracy
(%) of multimodal interactions. All refers to SimCLR augmenta￾tions (Chen et al., 2020a). CoMM uses All for both modalities.
Data augmentation. Finally,
we show in Table 5 that strong
data augmentation is crucial
for learning multi-modal inter￾actions, in line with the liter￾ature on unimodal contrastive
methods. Contrary to Fac￾torCL (Liang et al., 2023b), ap￾plying strong augmentation on
both modalities is beneficial for
CoMM and we do not require
task-dependent augmentations, highlighting the versatility of our framework.
6 RELATED WORK
Multimodal learning refers to methods that connect and integrate information from multiple
sources of data (Baltrusaitis et al. ˇ , 2018; Akkus et al., 2023). Early works focused on training
9
Published as a conference paper at ICLR 2025
separate encoders and studied different fusion mechanisms to blend the information from different
inputs (Zeng et al., 2007; Perez-R ´ ua et al. ´ , 2019). With the development of transformers and ViTs,
the focus has shifted towards training a unique architecture and designing specific tasks and loss
functions to integrate multimodal interactions (Xu et al., 2023; Lu et al., 2019; Sun et al., 2019;
Chen et al., 2020b; Lu et al., 2020). Today multimodal learning includes different research lines,
from generative multimodal learning (Suzuki & Matsuo, 2022; Ramesh et al., 2021; Saharia et al.,
2022; Wang et al., 2022b; Alayrac et al., 2022) to multimodal representation learning (Zong et al.,
2024; Tian et al., 2020a). CoMM belongs to the latter category.
Self-supervised multimodal representation learning. Self-supervised learning aims to learn
general representations through supervisory signals from the data itself (known as the pretext
task) (Balestriero et al., 2023). In the multimodal context (Zong et al., 2024), self-supervised meth￾ods can be grouped according to their pretext task. Clustering-based methods (Alwassel et al.,
2020; Asano et al., 2020), where cluster assignments are used as pseudo-labels to train the model
and different modalities can be used as supervisory signals to each other; masking modeling meth￾ods (Mizrahi et al., 2023; Bachmann et al., 2022; Lu et al., 2023) that reconstruct pieces of informa￾tion that have been masked from input data. In the multimodal case, masked modelling is used in
a cross-modal way, by predicting missing information conditioned in other modalities. Contrastive
methods (Radford et al., 2021; Jia et al., 2021) in the multimodal context have been mostly used on
matched data from different modalities to obtain aligned–yet distinct–representations. In this work,
instead of representing each modality separately, we take a different perspective for contrastive
methods by learning a single multimodal representation of the inputs.
Multimodal contrastive learning. Current methods in contrastive learning are inspired by the idea
of multiview learning (Li et al., 2018) and have shown remarkable results in representation learning
in unimodal tasks (Chen et al., 2020a; He et al., 2020; Tian et al., 2020a; Oord et al., 2018). The
natural extension of these approaches to multimodal settings is to optimize a cross-modal contrastive
loss (Radford et al., 2021; Alayrac et al., 2020; Jia et al., 2021). Other works have gone even
further by introducing cross-modal and intra-modal contrastive objectives (Mu et al., 2022; Jain
et al., 2021; Mi et al., 2024), or by adding other intra-modality regularization terms (Wang et al.,
2022a; Kim et al., 2022). However, these approaches are designed to learn redundant information,
neglecting the contributions of uniqueness or synergy. Recently, FactorCL (Liang et al., 2023b)
has proposed a solution to model shared and unique task-relevant information explicitly. Yet, the
method relies heavily on assumptions that are hard to meet in practice; it proposes a factorized
approximation of multimodal interactions that is prone to cumulative errors and does not model
synergistic information. Alternatively, we propose CoMM, a contrastive multimodal approach that
leverages multimodal augmentations with a modular architecture optimized through information
theory-grounded losses to capture unique, redundant and synergistic interactions.
7 CONCLUSIONS
Multisensory integration is at the core of human perception, allowing us to build coherent represen￾tations of our environment. In this paper, we introduce CoMM, a contrastive multimodal method
that enables the integration of multiple modalities in a single multimodal representation space. Un￾like existing multimodal contrastive models, CoMM is designed to learn multimodal interactions
beyond redundancy, through Partial Information Decomposition theory. Our controlled experiments
on the bimodal Trifeature dataset demonstrate that CoMM successfully learns redundant, unique and
synergistic information. In real-life multimodal datasets from Multibench with two and three input
modalities, CoMM outperforms existing methods by large margins in almost every case, showing
the efficiency and versatility of CoMM to handle data across diverse domains (robotics, healthcare,
affective computing, multimedia) and data structures (time series, image, text, audio, tabular).
This work opens large avenues for future research on multimodal representation learning, in partic￾ular for crafting label-preserving multimodal augmentations not limited to unimodal augmentations.
Limitations and perspectives for future research are further discussed in Appendix A. Overall, the
simplicity and versatility of CoMM’s design make it a good candidate to learn deep representations
of several modalities across domains. It offers promises in better solving real-world problems rang￾ing from neuroscience (Preti & Van De Ville, 2019) and medical imaging (Boehm et al., 2022) to
remote sensing (Gomez-Chova et al. ´ , 2015).
10
Published as a conference paper at ICLR 2025
ACKNOWLEDGMENTS
We thank Jonathan Sauder, Valentin Gabeff and Valerie Zermatten for providing helpful feedback ´
on earlier versions of this work. JCN acknowledges the support from EPFL Science Seed Fund. BD
acknowledges the support from the PHRT project number 643.
ETHICS STATEMENT
We acknowledge that there exist potential privacy risks when using human behavior or medical
data. However, we minimize these potential risks by using publicly available datasets that have been
carefully collected, respecting participants’ consent, de-identifying medical data and anonymizing
video data. In this work, we have followed best practices in maintaining the privacy and safety of
these datasets.
REPRODUCIBILITY STATEMENT
To ensure the reproducibility of our work, we have used publicly available datasets from Multi￾Bench (Liang et al., 2021) (MIMIC, MOSI, UR-FUNNY, MUsTARD and Vision&Touch); MM￾IMDb (Arevalo et al., 2017), and the synthetic Trifeatures dataset (Hermann & Lampinen, 2020).
Details about these datasets and pre-processing steps can be found in Appendix E. Appendix B
thoroughly describes neural network architectures, data augmentation strategies, and experimental
settings used in all our experiments. We include in Appendix D a pseudo-code for implementing
CoMM’s training, and we released our code in this Github repository. We also provide details
on processing times and model complexity compared to other multimodal models in Appendix F.
Finally, all proofs for our theoretical developments can be found in Appendix G.
REFERENCES
Cem Akkus, Luyang Chu, Vladana Djakovic, Steffen Jauch-Walser, Philipp Koch, Giacomo Loss,
Christopher Marquardt, Marco Moldovan, Nadja Sauter, Maximilian Schneider, et al. Multimodal
deep learning. arXiv preprint arXiv:2301.04856, 2023.
Jean-Baptiste Alayrac, Adria Recasens, Rosalia Schneider, Relja Arandjelovic, Jason Ramapuram, ´
Jeffrey De Fauw, Lucas Smaira, Sander Dieleman, and Andrew Zisserman. Self-supervised mul￾timodal versatile networks. In Advances in Neural Information Processing Systems (NeurIPS),
volume 33, pp. 25–37, 2020.
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel
Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language
model for few-shot learning. In Advances in Neural Information Processing Systems (NeurIPS),
volume 35, pp. 23716–23736, 2022.
Humam Alwassel, Dhruv Mahajan, Bruno Korbar, Lorenzo Torresani, Bernard Ghanem, and
Du Tran. Self-supervised learning by cross-modal audio-video clustering. Advances in Neural
Information Processing Systems (NeurIPS), 33:9758–9770, 2020.
John Arevalo, Thamar Solorio, Manuel Montes-y Gomez, and Fabio A Gonz ´ alez. Gated multi- ´
modal units for information fusion. In International Conference on Learning Representations -
Workshops (ICLR-W), 2017.
Yuki Asano, Mandela Patrick, Christian Rupprecht, and Andrea Vedaldi. Labelling unlabelled
videos from scratch with multi-modal self-supervision. Advances in Neural Information Pro￾cessing Systems (NeurIPS), 33:4660–4671, 2020.
Pradeep K Atrey, M Anwar Hossain, Abdulmotaleb El Saddik, and Mohan S Kankanhalli. Multi￾modal fusion for multimedia analysis: a survey. Multimedia systems, 16:345–379, 2010.
Bernard J Baars. A cognitive theory of consciousness. Cambridge University Press, 1988.
Roman Bachmann, David Mizrahi, Andrei Atanov, and Amir Zamir. MultiMAE: Multi-modal multi￾task masked autoencoders. In European Conference on Computer Vision (ECCV), pp. 348–367,
2022.
11
Published as a conference paper at ICLR 2025
Randall Balestriero, Mark Ibrahim, Vlad Sobal, Ari Morcos, Shashank Shekhar, Tom Goldstein,
Florian Bordes, Adrien Bardes, Gregoire Mialon, Yuandong Tian, et al. A cookbook of self￾supervised learning. arXiv preprint arXiv:2304.12210, 2023.
Tadas Baltrusaitis, Chaitanya Ahuja, and Louis-Philippe Morency. Multimodal machine learning: ˇ
A survey and taxonomy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(2):
423–443, 2018.
Nils Bertschinger, Johannes Rauh, Eckehard Olbrich, Jurgen Jost, and Nihat Ay. Quantifying unique ¨
information. Entropy, 16(4):2161–2183, 2014.
Kevin M Boehm, Emily A Aherne, Lora Ellenson, Ines Nikolovski, Mohammed Alghamdi, Ignacio
Vazquez-Garc ´ ´ıa, Dmitriy Zamarin, Kara Long Roche, Ying Liu, Druv Patel, et al. Multimodal
data integration using machine learning improves risk stratification of high-grade serous ovarian
cancer. Nature Cancer, 3(6):723–733, 2022.
Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, Julien Mairal, Piotr Bojanowski, and ´
Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of
the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 9650–9660, 2021.
Santiago Castro, Devamanyu Hazarika, Veronica P ´ erez-Rosas, Roger Zimmermann, Rada Mihalcea, ´
and Soujanya Poria. Towards multimodal sarcasm detection (an obviously perfect paper). In
Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL),
pp. 4619–4629, 2019.
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for
contrastive learning of visual representations. In International Conference on Machine Learning
(ICML), pp. 1597–1607, 2020a.
Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and
Jingjing Liu. UNITER: Universal image-text representation learning. In European Conference
on Computer Vision (ECCV), pp. 104–120, 2020b.
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep
bidirectional transformers for language understanding. In Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies, Vol￾ume 1 (Long and Short Papers), pp. 4171–4186, 2019.
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas
Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An im￾age is worth 16x16 words: Transformers for image recognition at scale. In International Confer￾ence on Learning Representations (ICLR), 2021.
Mohamed El Banani, Karan Desai, and Justin Johnson. Learning visual representations via
language-guided sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pp. 19208–19220, 2023.
Luis Gomez-Chova, Devis Tuia, Gabriele Moser, and Gustau Camps-Valls. Multimodal classifica- ´
tion of remote sensing images: A review and future directions. Proceedings of the IEEE, 103(9):
1560–1584, 2015.
Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting deep learning
models for tabular data. Advances in Neural Information Processing Systems (NeurIPS), 34:
18932–18943, 2021.
Anirudh Goyal and Yoshua Bengio. Inductive biases for deep learning of higher-level cognition.
Proceedings of the Royal Society A, 478(2266):20210068, 2022.
GroupLens research. MovieLens dataset, 2015. URL https://grouplens.org/datasets/
movielens/.
12
Published as a conference paper at ICLR 2025
Md Kamrul Hasan, Wasifur Rahman, Amir Zadeh, Jianyuan Zhong, Md Iftekhar Tanveer, Louis￾Philippe Morency, et al. UR-FUNNY: A multimodal language dataset for understanding humor.
In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP),
pp. 2046–2056, 2019.
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recog￾nition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 770–778, 2016.
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for
unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 9729–9738, 2020.
Katherine Hermann and Andrew Lampinen. What shapes feature representations? exploring
datasets, architectures, and training. Advances in Neural Information Processing Systems
(NeurIPS), 33:9995–10006, 2020.
Pengbo Hu, Xingyu Li, and Yi Zhou. SHAPE: A unified approach to evaluate the contribution and
cooperation of individual modalities. In Proceedings of the International Joint Conference on
Artificial Intelligence (IJCAI), pp. 3064–3070, 2022.
Aashi Jain, Mandy Guo, Krishna Srinivasan, Ting Chen, Sneha Kudugunta, Chao Jia, Yinfei Yang,
and Jason Baldridge. MURAL: Multimodal, multitask representations across languages. In Find￾ings of the Association for computational Linguistics: Empirical Methods in Natural Language
Processing (EMNLP), pp. 3449–3463, 2021.
Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan
Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning
with noisy text supervision. In International Conference on Machine Learning (ICML), pp. 4904–
4916, 2021.
Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad
Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. MIMIC-III,
a freely accessible critical care database. Scientific data, 3(1):1–9, 2016.
Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron
Maschinot, Ce Liu, and Dilip Krishnan. Supervised contrastive learning. Advances in Neural
Information Processing Systems (NeurIPS), 33:18661–18673, 2020.
Byoungjip Kim, Sungik Choi, Dasol Hwang, Moontae Lee, and Honglak Lee. Transferring pre￾trained multimodal representations with cross-modal similarity matching. Advances in Neural
Information Processing Systems (NeurIPS), 35:30826–30839, 2022.
Wessel Kraaij, Thomas Hain, Mike Lincoln, and Wilfried Post. The AMI meeting corpus. In
Proceedings of the International Conference on Methods and Techniques in Behavioral Research,
2005.
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. ImageNet classification with deep con￾volutional neural networks. Advances in Neural Information Processing Systems (NeurIPS), 25:
1097–1105, 2012.
Michelle A Lee, Yuke Zhu, Peter Zachares, Matthew Tan, Krishnan Srinivasan, Silvio Savarese,
Li Fei-Fei, Animesh Garg, and Jeannette Bohg. Making sense of vision and touch: Learning
multimodal representations for contact-rich tasks. IEEE Transactions on Robotics, 36(3):582–
596, 2020.
Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang,
Ziwei Liu, and Chunyuan Li. LLaVA-NeXT: Stronger LLMs supercharge multimodal
capabilities in the wild, May 2024. URL https://llava-vl.github.io/blog/
2024-05-10-llava-next-stronger-llms/.
13
Published as a conference paper at ICLR 2025
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image
pre-training with frozen image encoders and large language models. In International Conference
on Machine Learning (ICML), pp. 19730–19742, 2023.
Yingming Li, Ming Yang, and Zhongfei Zhang. A survey of multi-view representation learning.
IEEE Transactions on Knowledge and Data Engineering, 31(10):1863–1883, 2018.
Paul Pu Liang, Yiwei Lyu, Xiang Fan, Zetian Wu, Yun Cheng, Jason Wu, Leslie Chen, Peter Wu,
Michelle A Lee, Yuke Zhu, et al. MultiBench: Multiscale benchmarks for multimodal represen￾tation learning. In Neural Information Processing Systems (NeurIPS) – Track on Datasets and
Benchmarks, volume 1, 2021.
Paul Pu Liang, Yun Cheng, Xiang Fan, Chun Kai Ling, Suzanne Nie, Richard Chen, Zihao Deng,
Faisal Mahmood, Ruslan Salakhutdinov, and Louis-Philippe Morency. Quantifying & modeling
feature interactions: An information decomposition framework. Advances in Neural Information
Processing Systems (NeurIPS), 36:27351–27393, 2023a.
Paul Pu Liang, Zihao Deng, Martin Ma, James Zou, Louis-Philippe Morency, and Ruslan Salakhut￾dinov. Factorized contrastive learning: Going beyond multi-view redundancy. Advances in Neural
Information Processing Systems (NeurIPS), 36:32971–32998, 2023b.
Paul Pu Liang, Chun Kai Ling, Yun Cheng, Alex Obolenskiy, Yudong Liu, Rohan Pandey, Alex
Wilf, Louis-Philippe Morency, and Ruslan Salakhutdinov. Multimodal learning without labeled
multimodal data: Guarantees and applications. In International Conference on Learning Repre￾sentations (ICLR), 2024.
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Confer￾ence on Learning Representations (ICLR), 2019.
Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. ViLBERT: Pretraining task-agnostic visiolin￾guistic representations for vision-and-language tasks. Advances in Neural Information Processing
systems (NeurIPS), 32:13–23, 2019.
Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee. 12-in-1: Multi-task
vision and language representation learning. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 10437–10446, 2020.
Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi.
Unified-IO: A unified model for vision, language, and multi-modal tasks. In International Con￾ference on Learning Representations (ICLR), 2023.
Harry McGurk and John MacDonald. Hearing lips and seeing voices. Nature, 264(5588):746–748,
1976.
Li Mi, Chang Xu, Javiera Castillo-Navarro, Syrielle Montariol, Wen Yang, Antoine Bosselut, and
Devis Tuia. ConGeo: Robust cross-view geo-localization across ground view variations. arXiv
preprint arXiv:2403.13965, 2024.
David Mizrahi, Roman Bachmann, Oguzhan Fatih Kar, Teresa Yeo, Mingfei Gao, Afshin Dehghan, ˘
and Amir Zamir. 4M: Massively multimodal masked modeling. Advances in Neural Information
Processing Systems (NeurIPS), 36:58363–58408, 2023.
Gil Morrot, Fred´ eric Brochet, and Denis Dubourdieu. The color of odors. ´ Brain and language, 79
(2):309–320, 2001.
Norman Mu, Alexander Kirillov, David Wagner, and Saining Xie. SLIP: Self-supervision meets
language-image pre-training. In European Conference on Computer Vision (ECCV), pp. 529–
544, 2022.
Jiquan Ngiam, Aditya Khosla, Mingyu Kim, Juhan Nam, Honglak Lee, and Andrew Y Ng. Mul￾timodal deep learning. In International Conference on Machine Learning (ICML), pp. 689–696,
2011.
14
Published as a conference paper at ICLR 2025
Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. BYOL for
audio: Self-supervised learning for general-purpose audio representation. In International Joint
Conference on Neural Networks (IJCNN), pp. 1–8, 2021.
Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predic￾tive coding. arXiv preprint arXiv:1807.03748, 2018.
Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, ´
Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas
Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael
Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut,
Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without super￾vision. Transactions on Machine Learning Research, 2024.
Juan-Manuel Perez-R ´ ua, Valentin Vielzeuf, St ´ ephane Pateux, Moez Baccouche, and Fr ´ ed´ eric Jurie. ´
MFAS: Multimodal fusion architecture search. In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), pp. 6966–6975, 2019.
Maria Giulia Preti and Dimitri Van De Ville. Decoupling of brain function from structure reveals
regional behavioral specialization in humans. Nature Communications, 10(1):4747, 2019.
Sanjay Purushotham, Chuizheng Meng, Zhengping Che, and Yan Liu. Benchmarking deep learning
models on large healthcare datasets. Journal of biomedical informatics, 83:112–134, 2018.
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language under￾standing by generative pre-training. https://openai.com/research/language-unsupervised, 2018.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agar￾wal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya
Sutskever. Learning transferable visual models from natural language supervision. In Interna￾tional Conference on Machine Learning (ICML), pp. 8748–8763, 2021.
Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen,
and Ilya Sutskever. Zero-shot text-to-image generation. In International Conference on Machine
Learning (ICML), pp. 8821–8831, 2021.
Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT￾networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP￾IJCNLP), pp. 3982–3992, 2019.
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kam￾yar Seyed Ghasemipour, Raphael Gontijo-Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan
Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with
deep language understanding. Advances in Neural Information Processing Systems (NeurIPS),
35:36479–36494, 2022.
Sethuraman Sankaran, David Yang, and Ser-Nam Lim. Refining multimodal representations using a
modality-centric self-supervised module. https://openreview.net/forum?id=hB2HIO39r8G, 2022.
Cees GM Snoek and Marcel Worring. Multimodal video indexing: A review of the state-of-the-art.
Multimedia Tools and Applications, 25:5–35, 2005.
Chen Sun, Austin Myers, Carl Vondrick, Kevin Murphy, and Cordelia Schmid. VideoBERT: A
joint model for video and language representation learning. In Proceedings of the IEEE/CVF
International Conference on Computer Vision (ICCV), pp. 7464–7473, 2019.
Masahiro Suzuki and Yutaka Matsuo. A survey of multimodal deep generative models. Advanced
Robotics, 36(5-6):261–278, 2022.
Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In European
Conference on Computer Vision (ECCV), pp. 776–794, 2020a.
15
Published as a conference paper at ICLR 2025
Yonglong Tian, Chen Sun, Ben Poole, Dilip Krishnan, Cordelia Schmid, and Phillip Isola. What
makes for good views for contrastive learning? Advances in neural information processing sys￾tems (NeurIPS), 33:6827–6839, 2020b.
Haoqing Wang, Xun Guo, Zhi-Hong Deng, and Yan Lu. Rethinking minimal sufficient represen￾tation in contrastive learning. In Proceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR), pp. 16041–16050, 2022a.
Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu,
and Lijuan Wang. GIT: A generative image-to-text transformer for vision and language. Trans￾actions on Machine Learning Research, 2022b.
Paul L Williams and Randall D Beer. Nonnegative decomposition of multivariate information. arXiv
preprint arXiv:1004.2515, 2010.
Andrew T Woods, E Poliakoff, DM Lloyd, J Kuenzel, R Hodson, H Gonda, J Batchelor, Garmt B
Dijksterhuis, and A Thomas. Effect of background noise on food perception. Food Quality and
Preference, 22(1):42–47, 2011.
Peng Xu, Xiatian Zhu, and David A Clifton. Multimodal learning with transformers: A survey.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12113–12132, 2023.
Xin Yuan, Zhe Lin, Jason Kuen, Jianming Zhang, Yilin Wang, Michael Maire, Ajinkya Kale, and
Baldo Faieta. Multimodal contrastive training for visual representation learning. In Proceedings
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6995–
7004, 2021.
Ben P Yuhas, Moise H Goldstein, and Terrence J Sejnowski. Integration of acoustic and visual
speech signals using neural networks. IEEE Communications Magazine, 27(11):65–71, 1989.
Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. Multimodal sentiment in￾tensity analysis in videos: Facial gestures and verbal messages. IEEE Intelligent Systems, 31(6):
82–88, 2016.
Zhihong Zeng, Maja Pantic, Glenn I Roisman, and Thomas S Huang. A survey of affect recog￾nition methods: audio, visual and spontaneous expressions. In Proceedings of the International
Conference on Multimodal Interfaces (ICMI), pp. 126–133, 2007.
Yongshuo Zong, Oisin Mac Aodha, and Timothy Hospedales. Self-supervised multimodal learning:
A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–20, 2024.
16
Published as a conference paper at ICLR 2025
A LIMITATIONS AND FUTURE RESEARCH
• CoMM’s theoretical analysis for more than two modalities is still unclear since PID theory
is currently limited to n = 2 modalities (see Section 2 in the main paper). While uniqueness
can be easily defined for any n ≥ 2, the number of interactions between modalities (e.g. re￾dundancy and synergy) grows exponentially with n, making the analysis harder. Nonetheless,
we should recall that CoMM performs empirically very well even for n > 2 (see Section 4.3),
and the number of loss terms increases only linearly with n (see Eq. (7)). Another limitation is
the additional computational cost associated with adding modality-specific encoders. A simple
workaround is to use large pretrained (frozen) encoders (e.g. from CLIP (Radford et al., 2021)
for vision and text or DINOv2 (Oquab et al., 2024) for vision) and to only tune a lightweight
fusion transformer in CoMM, allowing a much faster training.
• CoMM’s computational cost for data augmentation is higher than in cross-modalities frame￾works (such as CLIP). This is because for each data tuple X = (X1, X2, . . . , Xn), CoMM
needs to compute X′ = t(X), while cross-modality methods feed X directly into the neu￾ral network. A possible solution would be the implementation of a momentum encoder and a
queue method, as in MoCo (He et al., 2020).
• Interpretability. Our experiments on the Trifeature dataset (see Section 4.1 in the main paper)
show that CoMM can efficiently learn unique, redundant and synergistic information. However,
it seems difficult to disentangle the contributions of these three interactions in the representation
space. Disentanglement might be one direction for future work. However, other approaches
to measure such quantities (given a dataset and a task) are emerging (Hu et al., 2022; Liang
et al., 2023a; 2024). Another interesting approach would be to use modality masking (already
implemented and handled by CoMM) to analyze the contribution of each modality individually
versus collectively.
We believe the above limitations are directions that were outside the scope of the current manuscript;
however, they are exciting avenues for future research in multimodal representation learning.
B IMPLEMENTATION DETAILS
For all experiments with CoMM, we use an attention-based fusion module that takes as input a
sequence of embeddings. We use a 1-layer Transformer with 8 heads that applies self-attention
to all inputs and we add a [CLS] learnable embedding at the beginning of the sequence. The
embedding size depends on each dataset, detailed below.
B.1 ENCODER ARCHITECTURES BY DATASET
• Trifeature is composed of two visual modalities, so we use an AlexNet encoder for both modal￾ities with a 512-d embedding space. For CoMM, we remove the last average pooling layer and
we apply a linear patch embedding layer (Dosovitskiy et al., 2021) to the 6 × 6 feature maps as
latent converter. We then add fixed 2D sine-cosine positional embeddings (Dosovitskiy et al.,
2021). For Cross, we use a linear projector for each encoder (512-d output space) and we opti￾mize the CLIP loss (Radford et al., 2021). For Cross+Self, we apply a 3-layers MLP projection
head to each encoder to optimize the SSL objectives as in SLIP (Mu et al., 2022) (1024-d hid￾den layers and 256-d output space) and we use the same projectors and CLIP objective as for
Cross.
• MIMIC contains tabular and time-series data, seen as two modalities. We use a 2-layers MLP
(10-d hidden dimension, 10-d output) as tabular encoder and a GRU (512-d hidden dimension)
as time-series encoder (similarly to FactorCL (Liang et al., 2023b)). For tabular data, we use a
feature tokenizer (Gorishniy et al., 2021) as latent converter with 512-d embedding space and
no converter for time-series.
• MOSI, UR-FUNNY and MUsTARD contain visual and textual modalities extracted from
videos. Similarly to FactorCL (Liang et al., 2023b), we use a 5-head Transformer with 5 layers
for each modality with a 40-d embedding space. We do not use latent converters in this case.
17
Published as a conference paper at ICLR 2025
• MM-IMDb also contains visual and textual modalities but in their raw format. We use a ViT￾B/32 (Dosovitskiy et al., 2021) image encoder pre-trained with CLIP (Radford et al., 2021) and
a Sentence-BERT multilingual text encoder2 pre-trained with CLIP and distilled with Sentence￾BERT (Reimers & Gurevych, 2019). For CoMM, we consider the token embeddings given by
the image and text encoders (frozen) and we do not use latent converters. For CLIP (Rad￾ford et al., 2021), we fine-tune the pre-trained encoders with their original architecture. For
SLIP (Mu et al., 2022), we use the same pre-trained encoders as CLIP and we add a 3-layers
visual projection MLP (4096-d hidden layers and 256-d output space) to compute the SSL
objectives.
• Vision&Touch has visual, force-torque and robot proprioception modalities available. For the
binary contact prediction task, we only use visual and proprioception data for the experiments
with two modalities and we encode images with a ResNet18 (He et al., 2016) (512-d output
space) and proprioception data with a 5-layer MLP (512-d output space), as in the original
paper (Lee et al., 2020). In the experiments with 3 modalities, force-torque data are encoded
with a 5-layer causal convolutions network (512-d output space). For CoMM, we remove the
last average pooling layer and apply a patch embedding layer (Dosovitskiy et al., 2021) to the
4 × 4 feature maps as latent converter. We consider the 1D feature vector of proprioception
data as a 1-length sequence in the fusion module. For the end-effector regression task, we use
visual and force-torque modalities. We use ResNet18 as image encoder (128-d output space)
and a 5-layer causal convolutions network (128-d output space) as force encoder (Lee et al.,
2020). For CoMM, we add the same latent converter for images as in the previous task and we
add a feature tokenizer (Gorishniy et al., 2021) for force-torque embeddings.
B.2 DATA AUGMENTATION BY MODALITY
For raw images (in Trifeature, MM-IMDb and Vision&Touch), we use the default SimCLR augmen￾tations (Chen et al., 2020a), which include RandomResizedCrop, ColorJitter, RandomGrayscale,
GaussianBlur and RandomHorizontalFlip (from the PyTorch library).
For tabular data (in MIMIC and Vision&Touch), we add a random Gaussian noise to each component
(assuming they are all continuous).
For time-series data (either extracted from videos as in MOSI, UR-FUNNY, MusTARD, from health
recordings as in MIMIC or from force-torque readings as in Vision&Touch), we apply a random
composition of Gaussian noise and random dropping between 0 and 80% of the sequence. We have
compared several other strategies for this modality and we present the results in Appendix C.
For raw text (in MM-IMDb), we randomly mask 15% of input tokens by using a special [MASK]
token as in BERT (Devlin et al., 2019).
B.3 LATENT CONVERTERS BY ENCODER ARCHITECTURE
• Transformer and GRU: the latent converter is the identity since the Transformer and GRU already
output a sequence of embeddings;
• CNN: the latent converter is a patch embedding projection module originally defined in
ViT (Dosovitskiy et al., 2021) that we apply to the features maps of the CNN;
• MLP: the latent converter is a feature tokenizer originally defined in (Gorishniy et al., 2021)
for tabular data. The feature vector hi
is transformed into sequential embeddings by applying
feature-wise multiplication with a learnable matrix and we add a bias term. For proprioception
and force-torque data, we simply consider them as a 1-length sequence in the fusion module.
B.4 EXPERIMENTAL SETTINGS
We use AdamW optimizer (Loshchilov & Hutter, 2019) in all experiments and a learning rate α =
3 × 10−4
for Trifeature (with weight decay 10−4
), α = 10−3
for MIMIC, MOSI, UR-FUNNY
and MusTARD (with weight decay 10−2
) and α = 10−4
for MM-IMDb and Vision&Touch (with
weight decay 10−2
). For MM-IMDb, we also use a cosine scheduler with final value 10−6
and a
warmup over 10 epochs. All models were optimized during 100 epochs. The critic in the InfoNCE
2https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1
18
Published as a conference paper at ICLR 2025
losses in LCoMM Eq. (7) is implemented as a 3-layer MLP (512-d hidden layers and 256-d output
space), similarly to the projection head in SimCLR (Chen et al., 2020a). All experiments ran on a
single V100 GPU with 32GB of memory.
Fine-tuning of CoMM. For all downstream classification tasks, we use the SupCon loss (Khosla
et al., 2020) to fine-tune CoMM (with no additional parameters). In the case of multi-label classifi￾cation with MM-IMDb, we use a linear head on top of CoMM and we optimize a cross-entropy loss
for each label. For the regression task on Vision&Touch, we also use a linear head and we optimize
the MSE loss. We systematically use early-stopping according to the validation accuracy in order to
prevent over-fitting on the downstream tasks.
B.5 EXPERIMENTAL SETTINGS ON THE BIMODAL TRIFEATURE DATASET
To generate our trifeature dataset, we considered the 1 000 combinations of the three features ex￾isting in the original dataset (see Appendix E) and split them into 800 combinations for training
and 200 for evaluation. To have more variety in the training set for training, each combination was
generated 3 times (the shape and the texture were randomly rotated), obtaining a training split of
2 400 images.
The bimodal Trifeature dataset used in our experiments was built by considering the trifeature dataset
twice (as two separate modalities) and building pairs from these two dataset copies. In total, we get
5 760 000 pairs (2 400 × 2 400) available for training, and 40 000 (200×200) available for evalua￾tion.
To create a controlled environment for evaluation of multimodal interaction learning, we needed to
carefully design tasks where the dominant interaction was clearly defined.
1. To measure uniqueness U1 (resp. U2), given a pair of trifeature images, the task is to predict
the texture of the first (resp. the second) image. The task is then a 10-class classification
problem and chance level is at 10%.
2. To measure redundancy R, given a pair of trifeature images with the same shape (but differ￾ent color and texture), the task is to predict the shape of the pair (therefore, the redundant
information that can be extracted either from the first or the second image). The task is
then a 10-class classification problem and chance level is at 10%.
3. To measure synergy S, the definition of the task was more subtle as it should require infor￾mation from both modalities simultaneously and should not be possible to perform it from
one of the images alone. To achieve this, we defined a mapping M between the ten tex￾tures and the ten colors (e.g. stripes=red, dots=green, etc.). Then, given a pair of trifeature
images, the task is to predict whether the pair satisfies the mapping or not. The task is then
a binary classification problem and chance level is at 50%.
To evaluate these tasks, we built two versions of the bimodal trifeature dataset:
(i) For uniqueness and redundancy, we considered 10 000 image pairs (out of the 5 760 000
pairs) for training and 4 096 for testing, that have the same shape (to measure redundancy)
and different texture (to measure uniqueness).
(ii) For synergy, we considered 10 000 image pairs that respect the mapping M and used the
same test set as before (4 096 image pairs).
B.6 ADDITIONAL DETAILS FOR LLAVA-NEXT EVALUATION
We evaluate LLaVA-NeXT (Li et al., 2024) on MM-IMDb based on its answer to the following
prompt: ”From the following plot: {...} and this poster image, give me all the movie genres it
belongs to among the following list: {...}. Give me the answer as a list.” We formulate it as a close
question with limited number of answers to be closer to the linear probing setting for representation
learning models.
19
Published as a conference paper at ICLR 2025
Figure 6: Linear evaluation accuracy (%) on four datasets from MultiBench (Liang et al., 2021)
when applying individual or composing data augmentations on the modalities. Diagonal entries cor￾respond to one augmentation applied to all modalities while off-diagonal entries are a composition
of two augmentations. Average is the average of all four matrices. Results are averaged over 5 inde￾pendent runs (with different seeds) in each cell.
C ADDITIONAL EXPERIMENTS
In this section we present additional analysis of CoMM, including: a benchmark of several aug￾mentation strategies for time-series data, ablation studies on our loss function on a real dataset, and
supplementary results on the trifeatures dataset.
C.1 DATA AUGMENTATION STRATEGY FOR TIME-SERIES
For time-series data like in MIMIC, MOSI, UR-FUNNY and MuSTARD, there is no consensus with
respect to the best data augmentation strategy to apply for self-supervised contrastive learning. We
benchmark several common augmentations for time-series data including random crop between 8%
and 100% of the signal, random drop between 0% and 80% of the signal and adding Gaussian noise.
We also designed two new multimodal augmentations: multi-crop (resp. multi-drop) consisting of
cropping (resp. dropping) a signal across multiple modalities for time-aligned data (introducing
consistency in the preserved multimodal signal).
We tested these augmentations along with their composition on four datasets from Multi￾Bench (Liang et al., 2021) and we plot the results in Fig. 6. Overall, we find that composing
Gaussian noise and random drop results in the best performances across datasets and tasks. This
is our default augmentations strategy in our main experiments. Our proposed multi-drop and multi￾crop augmentations can provide better results in some cases (for MIMIC and MuSTARD), but we
select the same default augmentations for consistency across all datasets.
C.2 ABLATION STUDIES ON A REAL DATASET
We performed the loss ablation study on MM-IMDb. Results follow the same tendency as in the
Tri-features dataset (Fig. 5 main paper).
20
Published as a conference paper at ICLR 2025
Loss weighted-f1 macro-f1
P
i Li 60.71±0.17 53.35±0.37
L 54.94±0.50 47.13±0.56
LCoMM = L +
P i Li 61.48±0.18 54.63±0.22
Table 6: Ablation study of loss function contribu￾tions on MM-IMDb. LCoMM allows to better capture
multimodal interactions than each term separately.
C.3 DESIGN CHOICES ON TRIFEATURES EXPERIMENTS
Hermann & Lampinen (2020) experimented with Trifeatures using AlexNet and ResNet-50 back￾bones. Both architectures showed comparable results. Therefore, we chose to use AlexNet in our ex￾periments. For completeness, we ran the same experiments on ResNet-50, which show that CoMM
is the only model to learn all interactions, regardless of the architecture.
Model Redundancy Uniqueness Synergy
Cross C+S CoMM Cross C+S CoMM Cross C+S CoMM
AlexNet 100.0±0.02 99.7±0.2 99.9±0.03 11.6±0.9 86.9±0.8 87.8±1.6 50.0±0.0 50.0±0.03 71.9±2.0
ResNet 100.0±0.04 99.9±0.04 99.9±0.03 6.5±0.7 96.2±0.8 96.3±1.3 50.0±0.0 50.0±0.0 75.0±1.7
Table 7: Linear probing accuracy of redundancy (shape), uniqueness (texture) and synergy (color
and texture) on bimodal Trifeature dataset, with different backbone encoders. These results are
complementary to Fig. 4.
C.4 ON THE FEASIBILITY OF ASSUMPTION 1
(a) Trifeature dataset (synergy) (b) MM-IMDb dataset
Figure 7: Downstream task performance as the cropping augmentation strength decreases (for
Trifeatures) or when both cropping and masking augmentation strength for image and text re￾spectively decreases. INCE(X, X′
) is measured using the trained encoder fθ in Eq. (6). As
INCE(X, X′
) increases, the downstream performance first increases and then decreases, revealing
a “sweet spot”. It suggests the existence of an optimal augmentation policy that preserves task￾relevant information while removing nuisance features from the multimodal input data, giving more
credit to our Assumption 1.
Assumption 1 states the existence of an optimal augmentation policy T
∗
such that I(X, t(X)) =
I(X, Y ) for any t ∈ T ∗
, given a task Y . Hence, the set of optimal multimodal transformations
21
Published as a conference paper at ICLR 2025
is task-dependent. However, in practice, we have seen that the augmentations applied in our ex￾periments are general enough to obtain good results in a variety of datasets (7 datasets, 10 tasks,
diverse data modalities). These augmentations were chosen according to the findings in unimodal
self-supervised learning [2, 3], and by a comprehensive ablation study on the best augmentation
strategies applicable to time-series in Appendix C.1.
Theoretically, the set of multimodal augmentations need to be large enough such that no information
about the task Y is lost, but small enough to extract this information only. Tian et al. (2020b) have
referred to this observation as the InfoMin principle. In Section 5, our data augmentation ablation
study shows that if the set of augmentations is not large enough, then synergy cannot be learnt (the
performance is always random chance). In order to further explore the feasibility of Assumption 1
and inspired by the strategy developed by Tian et al. (2020b), we evaluate CoMM by progressively
increasing the strength of the data augmentations applied. We use random crop as the augmentation
to control in the vision domain, mainly for two reasons: first, it is intuitive that by decreasing the
level of cropping (keeping less information about the images), we are destroying the task-relevant
information; and second, because it has been empirically demonstrated that cropping is a critical
transformation in self-supervised learning for vision Chen et al. (2020a). For the text domain, we
use masking as the augmentation to control. More specifically, on Trifeatures we randomly crop
the original image in the first modality from 0% up to x% (x = 20% is the strongest augmentation
while x = 60% is the lightest); and from x% to 100% (x = 0.05% is the strongest, x = 15% the
lightest). For MM-IMDb, we also use random crop of the image modality from x% up to 100% and
masking of text with a decreasing probability x% from 90% (the strongest) to 20% (the lightest).
Our results are shown in Fig. 7. They demonstrate, both in the controlled environment of the bi￾modal Trifeatures dataset and in the real-world application of MM-IMDb, that the sweet spot of the
InfoMin principle can be reached. By gradually increasing the strength of the applied transforma￾tions, we enhance model performance by reducing noisy information, up to an optimal point (the
sweet spot) where noise is minimized while task-relevant information is preserved. However, ap￾plying overly strong augmentations destroys task-relevant information, leading to a degradation in
model performance.
D PSEUDO-CODE
Algorithm 1 presents the pseudo-code for CoMM’s training. It is written in the general case when
we have n modalities. It is complementary to Fig. 3 (main paper), which depicts the case for n = 2.
CoMM’s official implementation will be released upon acceptance of this work.
E DATASETS DETAILS
E.1 TRIFEATURE
The Trifeature dataset (Hermann & Lampinen, 2020) was introduced as a controlled environment to
study the properties of vision neural networks and how they learn different features (shape, texture
and color). Images contain one of ten shapes (triangle, square, circle, etc.), rendered in one of ten
textures (solid, stripes, grid, etc.), in one of ten colors (red, green, blue, etc.). Shapes are rendered
within a 128 × 128 square, rotated at an angle drawn between [−45◦
, 45◦
] and placed at a random
position within a larger image (224×224), such that the shape is fully contained in the image. Then,
an independently rotated texture and a color are applied.
In our experiments, we consider the bimodal version of this dataset as explained in Section 4.1 (main
paper).
E.2 MULTIBENCH
All the following datasets are pre-processed as described in (Liang et al., 2021).
• MIMIC (Medical Information Mart for Intensive Care) (Johnson et al., 2016) is a dataset com￾prising de-indentified clinical data related to patients admitted to critical care units at a large
22
Published as a conference paper at ICLR 2025
Algorithm 1 CoMM training algorithm
Require: Multi-modal dataset {X1, X2, ..., Xn}, label-preserving transformations T
⋆
, set of pro￾jection transformations T = {t1, . . . , tn}, batch size N, uni-modal encoders (fi)i∈[1..n]
, fusion
transformer g
for sampled mini-batch {xk}k∈[1..N] = (x
1
k
, ..., x
n
k
)k∈[1..N] do
for k ∈ [1..N] do
draw t
′
, t′′ ∼ T ⋆
x
′
k, x
′′
k ← t
′
(xk), t′′(xk)
z
′
k ← g(f1(x
′1
k
), ..., fn(x
′n
k
))
z
′′
k ← g(f1(x
′′1
k
), ..., fn(x
′′n
k
))
for i ∈ [1..n] do
x
i
k ← ti(xk)
z
i
k ← g(fi(x
i
k
))
end for
end for
for i ∈ [1..n] do
Li ← −2
1
N
 
P
N
k=1
log exp sim(z
i
k
,z
′
k)
P
l=k
exp sim(z
i
k
,z′
l)
+
P
N
k=1
log exp sim(z
i
k
,z
′′
k)
P
l=k
exp sim(z
i
k
,z′′l)
!
end for
L ← − N
1
N
P
k=1
log exp sim(z
′
k,z
′′
k)
P
l=k
exp sim(z′
k,z′′l)
LCoMM ← L +
P
n
i=1 Li
update (fi)i∈[1..n]
, g to minimize LCoMM
end for
return (fi)i∈[1..n]
, g
Boston-area hospital between 2001 and 2012. It contains information about 53 423 hospital ad￾missions, including 38 597 distinct patients. We use the data as provided by MultiBench (Liang
et al., 2021), organized as in (Purushotham et al., 2018). There are two data modalities: first, a
time series modality, composed by a set of medical measurements of a given patient taken ev￾ery hour during 24 hours. Each measurement is a vector of size 12 (i.e., including 12 different
measured numerical values). Second, a static modality, including medical information about
the patient (age, gender, etc.), represented as a vector of size 5 (tabular data). As in (Liang et al.,
2023b), in our experiments we address the binary classification task of predicting whether a pa￾tient fits any disease in the ICD-9 code in group 7 (460-519). ICD-9 (International Statistical
Classification of Diseases and Related Health Problems) codes are used to classify diseases
and a variety of symptoms. Almost every health condition can be assigned a unique ICD-9
code group, where each group includes a set of similar diseases.
• MOSI (Zadeh et al., 2016) is a sentiment analysis dataset obtained from 2 199 YouTube video
segments. Each sample consists of a video (visual frames), the corresponding audio and tran￾scription (text). The original dataset evaluates sentiment intensities with continuous labels
ranging from -3 to 3. We follow previous works (Liang et al., 2023b) and consider the binary
version of the labels (positive and negative), and the same data modalities for training: text and
videos.
• UR-FUNNY (Hasan et al., 2019) is a dataset for humor detection in human speech. It was
created from 1 866 TED talk videos, obtaining more than 16 000 samples (parts of the videos).
Each sample in the dataset consists of videos (visual frames), audio and their transcrips (text).
The task is binary classification (humor or non-humor sequence).
• MUsTARD (Castro et al., 2019) is a multimodal video dataset for automated sarcasm detection.
It contains videos from popular television shows including Friends, The golden girls, The big
bang theory and Sarcasmaholics anonymous. Each sample in the dataset correspond to an
utterance composed of a video (visual frames), its corresponding audio and the transcription
(text), labeled as sarcastic or non-sarcastic. As previous works (Castro et al., 2019; Liang et al.,
23
Published as a conference paper at ICLR 2025
2023b), we use the balanced partition consisting of 690 utterances. In our experiments with
two modalities, we considered only text and vision.
• Vision&Touch (Lee et al., 2020) is a robot manipulation multimodal dataset including visual,
force and proprioception data for a peg insertion task. The data is collected from a 7-DoF,
torque-controlled Franka Panda robot, with a triangle peg attached to its end-effector. Its goal
is to insert the peg into a triangle hole situated in a box attached to a table in front of the
robot. By running a random policy (the robot takes random actions) and a heuristic policy (the
robot attempts peg insertion), 150 trajectories are recorded, each of them consisting of 1 000
timesteps. These trajectories contain RGB images, depth maps, force, end-effector position
and velocity. Following MultiBench (Liang et al., 2021), we consider two tasks on this dataset:
(i) the binary task of predicting contact or no contact in the next timestep, and (ii) predicting
the end-effector position (measured in MSE).
E.3 MM-IMDB
Multimodal IMDb (MM-IMDb) (Arevalo et al., 2017) is a multimodal dataset for movie genre pre￾diction. It has been built from the Movielens 20M dataset (GroupLens research, 2015) by filter￾ing out movies without poster image. Therefore, MM-IMDb comprises 25 959 movies along with
their plot, poster, genres and additional metadata (e.g. year, language, writer, etc.). As in previous
works (Arevalo et al., 2017; Liang et al., 2021; Perez-R ´ ua et al. ´ , 2019), we consider posters (image)
and plots (text) as input data modalities to perform the multilabel classification genre prediction task
(23 categories, each movie can be assigned to several categories). Technically, MM-IMDb is part
of the Multibench benchmark (Liang et al., 2021), however, since we used raw data instead of the
proposed pre-processed features, we treat it as a separate dataset.
F DETAILS ON PROCESSING TIMES AND MODEL COMPLEXITY
We include in Table 8 an analysis of the complexity of CoMM against CLIP and BLIP-2. As we
can observe, the fusion module in CoMM adds a marginal computational cost to existing backbones
without compromising speed.
Model FLOPs MACs #Params Fwd-latency
CLIP 251G 126G 222M 488ms
CoMM (w/ CLIP) 281G 140G 229M 493ms
BLIP-2 9.22T 4.61T 1.17B 14s
CoMM (w/ BLIP-2) 9.48T 4.74T 1.17B 15s
Table 8: Comparison of model complexity and processing times of
different multimodal architectures.
G PROOFS
Proof 1 (Lemma 1) Indeed, combining Assumption 1 (multiview redundancy) and equations for
I(X1; Y |X2) and I(X2; Y |X1) from Eq. (2) (main paper) we obtain:
0 < I(X1; Y |X2) + I(X2; Y |X1) = U1 + U2 + 2S < 2εinfo (8)
Since εinfo is supposed to be small and all the terms U1, U2, S ≥ 0, their contributions are negligible.
Thus, under the multiview redundancy assumption I(X1, X2; Y ) ≈ R.
■
Proof 2 (Lemma 2) Given data processing inequalities for the Markov chains X → X′ → Zθ
′ and
Zθ
′ → X → Zθ, we have:
I(Zθ;Zθ
′
) ≤ I(X, Zθ
′
) ≤ I(X, X′
) (9)
The equality can be achieved, for example, by selecting fθ(·) = Id(·), the identity function. ■
24
Published as a conference paper at ICLR 2025
Proof 3 (Lemma 3) First, we prove that I(Zθ
′
⋆ ; Y ) = I(X′
, Y ).
Indeed, we have:
I(X′
; Y ) = I(X′
; Y ; X) + I(X′
; Y |X)
= I(Zθ
′
⋆ ; Y ; X) (by lemma 1 in (Wang et al., 2022a))
= I(Zθ
′
⋆ ; Y ) − I(Zθ
′
⋆ ; Y |X)
= I(Zθ
′
⋆ ; Y ) because Zθ
′
⋆ = fθ
⋆ (t(X)) (10)
Second, let T = {ti} such that X′ = ti(X) = Xi and Zθ
′
⋆ = fθ
⋆ (Xi) = Zi
for i ∈ {1, 2}
(with a slight abuse of notation). Thanks to the previous result (in Eq. (10)) and by the consistency
equations for I(Xi
; Y ) in Eq. (2) (main paper), the final result follows:
I(Zi
; Y ) = I(Zθ
′
⋆ ; Y )
= I(X′
; Y ) because of Eq. (10)
= I(Xi
; Y )
= R + Ui because of consistency equations. (11)
■
25

以上是第二篇文章的内容。

---

好的，我已经仔细阅读了第二篇文章《WHAT TO ALIGN IN MULTIMODAL CONTRASTIVE LEARNING?》。

这篇文章提出了一个名为 CoMM (Contrastive Multimodal learning) 的新框架，同样旨在解决多模态对比学习中超越“多视图冗余”的局限性。

现在，我将为您详细比较 **FactorCL** (第一篇文章) 和 **CoMM** (第二篇文章) 的异同点。

### **核心思想与目标的异同**

#### **相似之处 (共同目标)**

两篇文章的出发点高度一致：

1.  **批判传统方法**：都明确指出，传统的跨模态对比学习（如 CLIP）依赖于“多视图冗余假设”，只关注学习模态间的**共享信息 (Redundancy)**，而在现实世界中，**独有信息 (Uniqueness)** 甚至 **协同信息 (Synergy)** 对下游任务至关重要。
2.  **超越冗余**：核心目标都是设计一个能够同时捕获**共享、独有**以及（在CoMM中明确提出的）**协同**信息的多模态自监督学习框架。
3.  **理论基础**：都基于信息论对多模态信息进行分解，作为其方法设计的理论依据。
4.  **解决方案**：都采用了**对模态信息进行增强**，并通过**对比学习**最大化不同“视图”间的互信息来驱动表示学习。

#### **不同之处 (核心分歧)**

尽管目标相似，但两篇文章在**理论框架、实现路径**和**核心假设**上存在根本性的区别。

1.  **信息分解理论**:
    *   **FactorCL**: 使用基于**条件互信息 (Conditional Mutual Information)** 的分解：`I(X1, X2; Y) = S + U1 + U2`。这里的 `S` 是 `I(X1; X2; Y)`，代表任务相关的共享信息。这个框架**没有显式地定义协同信息 (Synergy)**。
    *   **CoMM**: 使用**部分信息分解 (Partial Information Decomposition, PID)** 理论：`I(X1, X2; Y) = R + U1 + U2 + S`。这里的 `R` 是纯粹的冗余信息，`S` 是纯粹的协同信息（即信息只有在两个模态同时出现时才涌现）。CoMM 的理论框架比 FactorCL **更完整**，因为它明确地将**协同信息**作为一个独立的、需要学习的目标。

2.  **核心学习范式：“先分解再对齐” vs “先融合再对比”**
    *   **FactorCL (Factorize-then-Align)**：这是一个“**分解式**”的范式。它试图为共享信息和独有信息学习出**独立的、可分解的表示** (`ZS1`, `ZS2`, `ZU1`, `ZU2`)。其学习目标是精确地分离并优化这些不同的信息成分。
    *   **CoMM (Fuse-then-Contrast)**：这是一个“**整体式**”或“**融合式**”的范式。它**不试图显式地分离**不同的信息成分。相反，它首先将所有模态信息**融合**成一个统一的多模态表示 `Z`，然后通过对比这个统一表示的不同增强版本来学习。它认为各种信息成分（R, U, S）会从这个统一的对比目标中**自然涌现**。

### **方法和技术细节的详细比较**

为了更清晰地展示差异，我将使用一个表格进行对比：

| 特性 | **FactorCL (第一篇文章)** | **CoMM (第二篇文章)** |
| :--- | :--- | :--- |
| **理论框架** | 基于条件互信息分解，关注**共享(S)**和**独有(U)**信息。 | 基于部分信息分解(PID)，关注**冗余(R)**、**独有(U)**和**协同(S)**信息。 |
| **核心架构** | 学习**多个分离的表示**：`ZS1`, `ZS2` (共享) 和 `ZU1`, `ZU2` (独有)。 | 学习**一个统一的多模态表示** `Z`。架构核心是一个融合模块（如Transformer）。 |
| **对比目标** | 对齐**不同模态/信息成分**的表示。例如，通过`INCE(X1; X2)`学习共享信息。 | 对齐**同一个多模态表示的不同增强版本**。核心目标是最大化 `I(Z', Z'')`。 |
| **对模态增强的应用** | **核心目的**：用增强视图 `(X1', X2')` **近似任务标签 `Y`**，从而在自监督下计算**条件互信息** `I(X1; X2 \| Y)`，以分离任务相关和无关信息。 | **核心目的**：为统一的多模态表示 `Z` **创造不同的“视图”** (`Z'` 和 `Z''`) 以进行对比学习。 |
| **增强策略** | 提出“**最优多模态增强**”和“**独有信息增强**”。增强是**有条件的、非对称的**（对`X2`的增强依赖于`X1`），并且可能是任务相关的。 | 认为**标准的、强力的、独立的**单模态增强组合即可。增强是**无条件的**，不需要针对特定任务或模态间关系进行特殊设计。CoMM 甚至批评 FactorCL 的条件增强假设“不切实际”。 |
| **优化函数** | **显式地移除无关信息**。通过最大化互信息**下界**(`INCE`)捕获信息，同时最小化互信息**上界**(`INCE-CLUB`)来移除无关信息。 | **隐式学习**。只使用互信息下界估计器 (`InfoNCE`) 进行最大化。它没有显式地移除无关信息的机制，而是相信通过对比学习能自动关注重要信息。 |
| **如何捕获独有信息** | 通过最大化 `I(Xi; Y \| X-i)` 的下界（近似为 `INCE(Xi; Xi')` 减去共享信息项）来**直接优化**独有信息表示 `ZUi`。 | 通过一个辅助的对比目标 `I(Zi, Z')`。其中 `Zi` 是通过**掩码 (masking)** 其他所有模态只保留模态 `i` 得到的。这个目标用于学习 `R + Ui`。 |
| **对协同信息的处理** | **理论上未覆盖**。其框架无法显式地建模或捕获协同信息。 | **明确宣称可以捕获**。认为协同信息 `S` 会从最大化 `I(Z', Z'')` 的主要目标中自然涌现，因为这是唯一一个需要所有模态信息融合后才能优化的项。 |

### **总结与评述**

**FactorCL** 的方法更加“精细”和“严谨”。它试图从信息论的第一性原理出发，为每一种信息类型（共享、独有）设计精确的优化目标，甚至考虑了如何主动移除无关信息。它的核心创新在于利用**条件增强**来解决自监督下的信息分解问题。然而，这种方法的复杂性更高，并且其“独有信息增强”策略在实践中可能难以设计和实现。

**CoMM** 的方法则更加“简洁”、“实用”和“统一”。它回避了显式分解信息的复杂性，而是提出了一个非常简单的“先融合再对比”的框架。它相信一个强大的融合模块加上标准的对比学习目标，就足以让模型自己学会如何利用各种多模态交互（包括冗余、独有和协同）。它的核心创新在于**将对比学习的应用层面从跨模态提升到了多模态整体**，并首次将**协同信息**纳入了多模态对比学习的理论框架。实验结果（尤其是在需要协同信息的任务上）也显示了其方法的有效性。

**总而言之，两篇文章都认识到了同样的问题，并都巧妙地利用数据增强来使得基于互信息的自监督学习变得有意义。但它们的实现路径代表了两种不同的哲学：**

*   **FactorCL** 像是**分析化学家**，试图精确地**分离和提纯**混合物中的每一种成分。
*   **CoMM** 像是**冶金学家**，将所有原材料**熔于一炉**，相信在高温高压（对比学习）下能自然炼成性能优越的合金。

CoMM 的方法在理论上更完备（包含了协同信息），在实现上可能更简单和通用，这使其在实际应用中可能更具优势。而 FactorCL 提出的利用互信息上下界来提纯表示的思想，也极具启发性，可能会被未来的工作所借鉴。

---

在部分信息分解的研究中，我们需要独特/冗余/协同各个部分之间的互信息为0吗？在条件互信息的研究中呢？

---

这是一个非常深刻且关键的问题，它触及了这两种信息分解方法的核心差异。

简短的回答是：

*   在**部分信息分解 (PID)** 的研究中，**是的，我们追求**独特(U)、冗余(R)和协同(S)这几个部分在信息内容上是完全互斥的，它们之间的信息量交集为零。它们被定义为对总信息的一个**互斥划分 (Partition)**。
*   在**条件互信息 (CMI)** 的研究中，**是的，其分解出的各个项**在定义上也是信息内容互斥、不重叠的。这是由信息论的链式法则和条件作用的数学性质所保证的。

下面是详细的解释。

---

### 1. 部分信息分解 (Partial Information Decomposition, PID)

在 PID 的理论框架 `I(X1, X2; Y) = R + U1 + U2 + S` 中，其**核心目标**就是将整体信息 `I(X1, X2; Y)` 分解成四个**在功能上完全独立、不重叠**的“信息原子”。

*   **冗余信息 (R)**：这部分信息既可以从 `X1` 中获得，**也**可以从 `X2` 中获得。
*   **独有信息 (U1)**：这部分信息**只能**从 `X1` 中获得，而无法从 `X2` 中获得。
*   **独有信息 (U2)**：这部分信息**只能**从 `X2` 中获得，而无法从 `X1` 中获得。
*   **协同信息 (S)**：这部分信息**只能**在 `X1` 和 `X2` **同时**出现时才能获得，单独从任何一个模态中都无法获得。

**为什么它们必须互斥？**

因为这是 PID 框架的**定义性要求**。一个“信息比特”不能同时既是“`X1`独有的”又是“`X1`和`X2`冗余的”。这在逻辑上是矛盾的。PID 追求的是对信息进行一种类似集合划分的操作。

**一个直观的比喻：**

想象一下，要完成一项任务（预测 `Y`），需要收集四个不同的线索（R, U1, U2, S）。
*   **R** 是一个公开的秘密，Alice (`X1`) 和 Bob (`X2`) 都知道。
*   **U1** 是只写在 Alice 私人日记里的线索。
*   **U2** 是只存在于 Bob 个人记忆里的线索。
*   **S** 是一个密码，需要 Alice 的钥匙和 Bob 的保险箱同时出现才能解开。

很明显，这四个线索是完全不同的、不重叠的。PID 的挑战不在于是否应该让它们互斥（答案是肯定的），而在于**如何找到一个通用的数学定义来精确地衡量 R, U1, U2, S 的大小**，同时还能满足一系列直觉上的公理。这也是为什么 PID 领域至今仍有多种不同的度量方法在被研究和争论。

---

### 2. 条件互信息 (Conditional Mutual Information, CMI) 分解

在 FactorCL 使用的 CMI 分解 `I(X1, X2; Y) = I(X1; X2; Y) + I(X1; Y | X2) + I(X2; Y | X1)` 中，各个部分同样是**信息内容互斥的**。

这种互斥性是由**条件作用 (Conditioning)** 的数学本质所决定的。

*   **`I(X1; Y | X2)` (FactorCL 的 U1)**：这个项衡量的是“在**已经知道了 `X2` 所有信息**的前提下，`X1` **还能**为 `Y` 提供多少**新**信息”。
*   **`I(X2; Y | X1)` (FactorCL 的 U2)**：同理，这是在已知 `X1` 后，`X2` 提供的**新**信息。
*   **`I(X1; X2; Y)` (FactorCL 的 S)**：这是 `X1` 和 `X2` 关于 `Y` 的**共同**信息。

**为什么它们是互斥的？**

让我们以 `I(X1; Y | X2)` 为例。其定义本身就排除了所有可以从 `X2` 中获得的信息。因此，它所包含的信息内容，根据定义，不可能与 `I(X2; Y)` 或者 `I(X1; X2; Y)`（`X1`和`X2`的共享信息）重叠。条件作用就像一个过滤器，它把所有与条件变量 (`X2`) 相关的信息都“滤掉”了，只留下剩余的、独特的、增量的信息。

这实际上是信息论链式法则的一个直接推论，它本身就是一个无损且无重叠的分解。

**一个直观的比喻：**

假设 Alice (`X1`) 和 Bob (`X2`) 合作写一份报告（预测 `Y`）。
1.  首先，我们评估 Bob 独立完成了多少工作量，这就是 `I(X2; Y)`。
2.  然后，我们让 Alice 在 Bob 的工作成果**之上**继续工作。Alice 做的所有**新增的、非重复的**贡献，就是 `I(X1; Y | X2)`。
3.  这个过程保证了我们计算的 Alice 的贡献 (`I(X1; Y | X2)`) 和 Bob 的贡献 (`I(X2; Y)`) 是没有重叠的。

### 总结与关键区别

| 特性 | **部分信息分解 (PID)** | **条件互信息 (CMI) 分解** |
| :--- | :--- | :--- |
| **组件互斥性** | **是**，这是其核心**追求的目标**。 | **是**，这是其**数学定义**的直接结果。 |
| **分解方式** | 一种**概念性**的、**同时发生**的划分，试图回答“信息来自哪里？”。 | 一种**代数性**的、**序贯发生**的分解，回答“知道了A之后，B还能提供多少信息？”。 |
| **对协同信息的处理** | **显式定义**了协同项 `S`，这是其主要优势。 | **无法分离协同信息**。它将冗余和协同信息混在了一起。实际上，`I(X1; X2; Y) = R_pid - S_pid`。因此 CMI 分解无法处理纯粹的协同任务（如XOR问题）。 |
| **理论完备性** | 概念清晰，但**缺乏公认的度量衡**。 | 基于信息论公理，**定义明确且唯一**。 |

所以，两篇文章都依赖于一个信息内容互斥的分解框架。**CoMM** 选择了理论上更完备但度量上更具挑战性的 **PID** 框架作为其理论基础，因此它能明确地讨论并声称捕获了协同信息 `S`。而 **FactorCL** 选择了数学上定义清晰但无法分离协同信息的 **CMI** 框架，因此它只能关注于共享信息和独有信息。

---

我希望你进一步的详细列出两篇文章中分别使用了怎样的方法去捕获源变量中的各个成分。对于原文的证明部分我希望也一并列出以使得说明更加详尽。

---

好的，这是一个非常好的问题，因为它直击两篇论文方法论的核心。下面我将详细列出 FactorCL 和 CoMM 是如何捕获各自信息成分的，并附上原文中的关键证明，以使解释更加详尽和严谨。

---

### **第一部分: FactorCL 的捕获方法 (分解式)**

FactorCL 的核心思想是“**先分解，再优化**”。它为不同的信息成分设计了不同的、显式的优化目标。其理论基础是**条件互信息 (CMI)** 分解：

$I(X_1, X_2; Y) = \underbrace{I(X_1; X_2; Y)}_{S_{\text{cmi}}} + \underbrace{I(X_1; Y | X_2)}_{U_{1, \text{cmi}}} + \underbrace{I(X_2; Y | X_1)}_{U_{2, \text{cmi}}}$

#### **1. 捕获任务相关的共享信息 (S)**

*   **目标信息**: $S = I(X_1; X_2; Y) = I(X_1; X_2) - I(X_1; X_2|Y)$
    *   这个公式的直观含义是：共享信息 = 总共享信息 - 任务无关的共享信息。

*   **捕获方法**: FactorCL 设计了一个目标函数，该函数**最大化**总共享信息的**下界**，同时**最小化**任务无关共享信息的**上界**。在自监督设置下，标签 `Y` 被多模态增强视图 `(X_1', X_2')` 替代。

    $S \ge \underbrace{I_{NCE}(X_1; X_2)}_{\text{最大化总共享}} - \underbrace{I_{NCE-CLUB}(X_1; X_2|X_1', X_2')}_{\text{最小化无关共享}}$

    *   **`I_NCE(X_1; X_2)`**: 这是标准的对比学习损失 (如 InfoNCE)，通过将配对的 `(x_1, x_2)` 视为正样本，不配对的视为负样本，来拉近 `X1` 和 `X2` 的表示，从而捕获它们之间的全部共享信息。
    *   **`I_NCE-CLUB(X_1; X_2|X_1', X_2')`**: 这是一个创新的项。它估计了在已知增强视图 `(X_1', X_2')`（即近似于已知任务信息）后，`X1` 和 `X2` 之间**仍然存在**的共享信息。这部分信息被认为是与任务无关的，因此通过最小化其上界来将其从表示中**移除**。

#### **2. 捕获任务相关的独有信息 (Ui)**

*   **目标信息**: $U_1 = I(X_1; Y | X_2) = I(X_1; Y) - I(X_1; X_2; Y) = I(X_1; Y) - (I(X_1; X_2) - I(X_1; X_2|Y))$
    *   直观含义是：`X1` 的独有信息 = `X1` 的总任务信息 - `X1` 和 `X2` 的共享任务信息。

*   **捕获方法**: 类似地，FactorCL 通过最大化/最小化各项的下界/上界来构造目标函数。

    $U_i \ge \underbrace{I_{NCE}(X_i; X_i')}_{\text{捕获总任务信息}} - \underbrace{(I_{NCE-CLUB}(X_1; X_2) - I_{NCE}(X_1; X_2|X_1', X_2'))}_{\text{减去共享信息}}$

    *   **`I_NCE(X_i; X_i')`**: 这一项通过对比一个模态 `Xi` 和其自身的增强版本 `Xi'` 来学习。根据“最优增强”假设，`I(X_i; X_i') ≈ I(X_i; Y)`，因此这一项捕获了 `Xi` 中包含的**全部**任务相关信息（包括独有的和共享的）。
    *   **`-(...)`**：后面减去的项是上面计算出的共享信息的近似值。通过从总任务信息中减去共享信息，剩下的就是独有信息。

#### **原文证明**

FactorCL 在其附录 C.3 中的 **Theorem 6** 对上述目标函数的合理性进行了形式化证明。

> **Theorem 6. (Contrastive estimators for shared and unique information)**. Under assumptions on single-view augmentations $I(X_1; Y ) = I(X_1, X_1')$ (Definition 8) and optimal multi-view augmentation $X_2'$ such that $I(X_1, X_2; X_1', X_2') = I(X_1, X_2; Y )$ (Definition 9), we can define contrastive objectives for task-relevant shared and unique information with:
>
> $S = I(X_1; X_2; Y ) \ge I_{NCE}(X_1; X_2) - I_{NCE-CLUB}(X_1; X_2|X_1', X_2')$
>
> $U_i = I(X_i; Y |X_{-i}) \ge I_{NCE}(X_i; X_i') - I_{NCE-CLUB}(X_1; X_2) + I_{NCE}(X_1; X_2|X_1', X_2')$
>
> **Proof.** The objectives follow from the fact that $I_{NCE}(X_1; X_2)$ and $I_{NCE}(X_1; X_2|X_1', X_2')$ are lower bounds of $I(X_1; X_2)$ and $I(X_1; X_2|Y)$ respectively, and $I_{NCE-CLUB}(X_1; X_2)$ and $I_{NCE-CLUB}(X_1; X_2|X_1', X_2')$ are upper bounds of $I(X_1; X_2)$ and $I(X_1; X_2|Y)$ respectively:
>
> $S = I(X_1; X_2; Y ) = I(X_1; X_2) - I(X_1; X_2|Y)$
> $\ge I_{NCE}(X_1; X_2) - I_{NCE-CLUB}(X_1; X_2|X_1', X_2')$
>
> $U_i = I(X_i; Y |X_{-i}) = I(X_i; Y ) - (I(X_1; X_2) - I(X_1; X_2|Y))$
> $\ge I_{NCE}(X_i; X_i') - (I_{NCE-CLUB}(X_1; X_2) - I_{NCE}(X_1; X_2|X_1', X_2'))$
>
> and symmetrically for U2.

---

### **第二部分: CoMM 的捕获方法 (融合式)**

CoMM 的核心思想是“**先融合，再对比**”。它不为每个信息成分设计单独的目标，而是设计了两个目标，并声称 R, U, S 会从这些目标中**自然涌现**。其理论基础是**部分信息分解 (PID)**：

$I(X_1, X_2; Y) = \underbrace{R_{\text{pid}}}_{\text{冗余}} + \underbrace{U_{1, \text{pid}}}_{\text{独有}} + \underbrace{U_{2, \text{pid}}}_{\text{独有}} + \underbrace{S_{\text{pid}}}_{\text{协同}}$

#### **1. 捕获总信息 (R + U1 + U2 + S)**

*   **目标信息**: 捕获与任务相关的全部信息 $I(X, Y) = R+U_1+U_2+S$。
*   **捕获方法**: CoMM 的主要学习目标是最大化两个不同增强版本 `X'` 和 `X''` 经过**融合模块**后得到的**统一多模态表示** `Z'` 和 `Z''` 之间的互信息。

    $\mathcal{L} = - \hat{I}_{NCE}(Z', Z'')$

    *   **核心机制**: 论文假设存在“最小标签保持多模态增强” (Assumption 1)，使得 $I(X, X') = I(X, Y)$。通过最大化 `I(Z', Z'')`，模型被驱动去捕获 `I(X, X')` 中的所有信息，也就间接捕获了 `I(X, Y)` 中的所有信息，即 R, U1, U2, S 的总和。

#### **2. 捕获冗余和独有信息 (R + Ui)**

*   **目标信息**: 捕获单模态 `Xi` 所能提供的全部任务信息，即 $I(X_i; Y) = R + U_i$。
*   **捕获方法**: CoMM 引入了一个辅助的对比目标。它通过**掩码 (masking)** 的方式，只让模态 `i` 的信息通过模型，得到一个近似的单模态表示 `Zi`。然后最大化这个 `Zi` 与之前那个完整的、增强的多模态表示 `Z'` (或 `Z''`) 之间的互信息。

    $\mathcal{L}_i = - \frac{1}{2} (\hat{I}_{NCE}(Z_i, Z') + \hat{I}_{NCE}(Z_i, Z''))$

    *   **核心机制**: 这个目标强制模型学习到的表示 `Z` 具有这样的特性：即使只给出其中一个模态的信息（通过 `Zi`），也应该能与完整的表示 `Z'` 对齐。这会驱动模型保留该单模态 `Xi` 中包含的所有任务信息。根据 PID 的一致性方程，这部分信息正是 `R + Ui`。

#### **3. 捕获协同信息 (S)**

*   **目标信息**: 协同信息 `S`。
*   **捕获方法**: CoMM **没有为 `S` 设计直接的优化目标**。它认为 `S` 是**涌现**出来的。
    *   **核心机制**: 协同信息的定义是“必须同时看到 `X1` 和 `X2` 才能获得的信息”。在 CoMM 的两个损失项中，只有主要目标 $\mathcal{L} = - \hat{I}_{NCE}(Z', Z'')$ 需要模型**同时处理所有模态**来计算 `Z'` 和 `Z''`。而辅助目标 $\mathcal{L}_i$ 实际上只关注单模态信息。因此，模型为了优化主要目标 $\mathcal{L}$，**必须**学习如何利用多模态融合带来的 synergistic 信息。在 ablation study (Fig. 5) 中，论文也展示了单独优化 $\mathcal{L}_i$ 无法学习协同信息，而单独优化 $\mathcal{L}$ 虽然可以，但速度很慢，两者结合效果最好。

#### **原文证明**

CoMM 的理论 justification 来自其 **Lemma 2 和 Lemma 3** (附录 G)。

> **Proof 2 (Lemma 2)** Given data processing inequalities for the Markov chains $X \rightarrow X' \rightarrow Z'_{\theta}$ and $Z'_{\theta} \rightarrow X \rightarrow Z_{\theta}$, we have:
>
> $I(Z_{\theta};Z'_{\theta}) \le I(X, Z'_{\theta}) \le I(X, X')$
>
> The equality can be achieved, for example, by selecting $f_{\theta}(·) = Id(·)$, the identity function.

这个证明表明，最大化 $I(Z;Z')$ 的上限是 $I(X, X')$。结合 Assumption 1 ($I(X, X')=I(X,Y)$)，这就为**捕获总信息 (R+U+S)** 提供了理论依据。

> **Proof 3 (Lemma 3)** First, we prove that $I(Z'_{\theta \star} ; Y ) = I(X', Y )$.
> Indeed, we have:
> $I(X'; Y ) = I(X'; Y ; X) + I(X'; Y |X) = I(Z'_{\theta \star} ; Y ; X) \text{ (by lemma 1 in (Wang et al., 2022a))}$
> $= I(Z'_{\theta \star} ; Y ) - I(Z'_{\theta \star} ; Y |X) = I(Z'_{\theta \star} ; Y ) \text{ because } Z'_{\theta \star} = f_{\theta \star} (t(X))$
>
> Second, let $T = \{t_i\}$ such that $X' = t_i(X) = X_i$ and $Z'_{\theta \star} = f_{\theta \star} (X_i) = Z_i$ for $i \in \{1, 2\}$ (with a slight abuse of notation). Thanks to the previous result and by the consistency equations for $I(X_i; Y )$ in Eq. (2) (main paper), the final result follows:
>
> $I(Z_i; Y ) = I(Z'_{\theta \star} ; Y ) = I(X'; Y ) = I(X_i; Y ) = R + U_i$

这个证明是**捕获 R+Ui** 的关键。它表明，如果我们将增强函数 `t` 特殊化为仅选择第 `i` 个模态的“投影”操作，那么优化得到的表示 `Zi` 将保留 `Xi` 中关于 `Y` 的所有信息，根据 PID 理论，这部分信息就是 `R+Ui`。

---

两篇文章都使用了同一种模态增强方法，意图使得$I(X, X') = I(X, Y)$。这种方法在实际应用中如何实现？

---

当然。这是一个非常关键的问题，因为它揭示了这两篇论文从理论理想走向实际应用的核心桥梁。

首先，需要澄清一个重要的点：**两篇文章并非使用了完全相同的增强方法，而是遵循了同一个理论原则，但在实际应用哲学上有所不同。** 这个原则就是您提到的，通过数据增强 `X'` 来近似一个“保留任务相关信息，改变任务无关信息”的理想视图，从而使得 $I(X, X') \approx I(X, Y)$。

这个等式是一个**理论目标**，在实践中无法被精确实现或验证，因为在自监督学习中我们并不知道 `Y`，也不知道究竟哪些信息是“任务相关”的。因此，研究人员通过一系列**启发式 (heuristic)** 的方法来**近似**这个目标。

下面，我将详细列出这些启发式方法是如何在实际中实现的，并对比两篇论文在应用这些方法时的不同侧重点。

---

### **一、 如何在实践中实现“标签保持”增强**

实现的核心思想是：**一个好的增强应该在低级特征（如位置、颜色、音调、措辞）上产生变化，但在高级语义（如物体类别、核心情感、文本含义）上保持不变。** 因为下游任务通常与高级语义相关。

以下是针对不同模态的主流实现方法：

#### **1. 图像 (Image)**

这是研究最成熟的领域，通常使用“SimCLR-style”的一系列增强组合：
*   **RandomResizedCrop (随机裁剪并缩放)**：这是最关键的增强之一。它改变了物体的位置、大小和可见部分，但通常会保留核心物体本身。这迫使模型学习物体的内在特征，而不是其在图像中的特定位置。
*   **ColorJitter (色彩抖动)**：随机改变亮度、对比度、饱和度和色调。这使得模型对颜色不敏感，更关注形状和纹理。
*   **RandomGrayscale (随机灰度化)**：以一定概率将图像变为灰度图，进一步降低对颜色的依赖。
*   **RandomHorizontalFlip (随机水平翻转)**：对于大多数物体（如猫、汽车），翻转不会改变其类别。
*   **GaussianBlur (高斯模糊)**：轻微的模糊可以去除一些高频噪声，让模型关注更宏观的结构。

#### **2. 文本 (Text)**

文本增强比图像更具挑战性，因为微小的改动就可能改变语义。
*   **Masking (掩码)**：随机遮盖一部分词元 (token)，如 BERT 的做法。模型需要根据上下文恢复信息，这能学习到丰富的语境表示。
*   **Token Deletion/Shuffling (词元删除/打乱)**：小幅度地删除或打乱词的顺序，如果幅度不大，通常能保留句子主干含义。
*   **Back-translation (回译)**：将句子翻译成另一种语言，再翻译回来。例如，"The car is fast" -> (法语) "La voiture est rapide" -> (英语) "The vehicle is quick"。措辞（低级特征）变了，但核心语义（高级特征）得以保留。这是一种效果很好但计算成本较高的方法。

#### **3. 音频/时间序列 (Audio/Time-Series)**

*   **Adding Noise (添加噪声)**：在波形或频谱图上添加高斯噪声。
*   **Time Masking (时间掩码)**：在频谱图上遮盖掉一小段时间。
*   **Frequency Masking (频率掩码)**：在频谱图上遮盖掉一小段频率范围。（这两种掩码是 SpecAugment 的核心）
*   **Pitch Shifting (音调偏移)**：改变音频的音高，但不改变内容（如说话的词语或音乐的旋律）。
*   **Time Stretching (时间拉伸)**：在不改变音高的情况下，加快或减慢音频播放速度。

---

### **二、 FactorCL 和 CoMM 在应用这些方法时的哲学差异**

尽管它们可能使用上述工具箱中的相同工具（如都对图像使用随机裁剪），但它们**选择和组合这些工具的意图和方式**是不同的。

#### **CoMM 的方法：强力、通用、无条件 (Strong, General, Unconditional)**

CoMM 的哲学非常直接：它需要为其**统一的多模态表示 `Z`** 创造出两个尽可能不同的、但语义一致的“视图” `Z'` 和 `Z''`，以构造一个困难的对比任务。

*   **实现方式**：对于每个模态，**独立地**应用一套**强力**的、通用的增强。例如，对图像 `X1` 应用一套完整的 SimCLR 增强得到 `X1'`，同时独立地对文本 `X2` 应用 Masking 得到 `X2'`，组合成 `X' = (X1', X2')`。
*   **核心思想**：CoMM 认为，只要增强足够强，能够迫使模型在融合表示时忽略掉各模态内的低级噪声，就自然能学会关注跨模态的高级语义，包括冗余、独有和协同信息。它**不要求增强之间有任何依赖关系**。
*   **原文佐证**：在 CoMM 的 Ablation Study (Table 5) 中，作者明确指出，对两个模态都应用强力增强（{All}）时效果最好，并且特意反驳了 FactorCL 的观点，强调 CoMM **不需要任务相关的、有条件的增强**，这突显了其框架的通用性。

#### **FactorCL 的方法：精细、小心、有条件 (Careful, Conditional)**

FactorCL 的需求更为复杂。它不仅仅是为了创造对比视图，更是为了用增强 `(X1', X2')` 来**近似任务标签 `Y`**，以便能有意义地估计**条件互信息** $I(X_1; X_2 | Y)$。这就对其增强策略提出了更高的要求。

*   **实现方式**：FactorCL 提出了“**独有信息增强 (Unique Augmentation)**”的概念。这意味着对一个模态（如 `X2`）的增强，**需要考虑另一个模态（`X1`）的内容**，目的是**避免破坏它们之间的共享信息**。
*   **核心思想**：如果一个标准的增强（如随机裁剪）破坏了 `X1` 和 `X2` 之间的共享信息，那么增强后的视图 `(X1', X2')` 就不能很好地近似 `Y`，从而导致对任务无关共享信息 $I(X_1; X_2 | Y)$ 的估计出现偏差。
*   **一个具体的例子（来自原文）**：
    *   **输入**: 文本 `X1` = "这辆车很快"，图片 `X2` = 一辆在高速公路上飞驰的汽车。
    *   **共享信息**: “快”的概念，在图片中由“高速公路”这个背景元素体现。
    *   **CoMM 式的增强**: 可能会对图片 `X2` 进行随机裁剪，恰好把“高速公路”裁掉了。对于 CoMM 来说问题不大，模型可以从其他特征学习。
    *   **FactorCL 理想的增强**: 应该**避免**使用随机裁剪，而是使用**不影响背景**的 ColorJitter 或 HorizontalFlip。因为随机裁剪破坏了 `X1` 和 `X2` 都指向的“快”这个共享概念，使得 `(X1', X2')` 不再是 `Y` 的一个好代理。

### **总结**

| 对比维度 | **CoMM** | **FactorCL** |
| :--- | :--- | :--- |
| **增强目标** | 为**统一的多模态表示**创造强对比视图。 | 用增强视图**近似任务标签Y**，以计算条件互信息。 |
| **应用哲学** | **无条件的、独立的、强力的**。认为“越强越好”。 | **有条件的、依赖的、小心的**。需要避免破坏模态间的共享信息。 |
| **实现难度** | **更低**。可以直接套用各模态成熟的增强库。 | **更高**。需要根据具体的数据和任务，思考哪些增强是“安全”的，可能需要人工设计。 |
| **通用性** | **更强**。其方法不依赖于对模态间关系的先验知识。 | **可能更弱**。其“独有信息增强”策略可能需要为不同任务定制。 |

总而言之，两篇论文都将 $I(X, X') = I(X, Y)$ 作为理论基石，但在实践中，CoMM 采用了一种更简单、更鲁棒的近似方法，而 FactorCL 为了其更精细的分解目标，提出了一种更具挑战性、需要小心设计的条件增强策略。这也是 CoMM 在论文中批评 FactorCL 的假设“不切实际 (unrealistic)”的原因所在。

---

那么这两篇文章中的实验部分，作者在实际数据集上检验提出的方法时，有对实际使用的增强方法做介绍吗？如果有，详细的描述作者使用的方法。

---

是的，两篇文章都在其实验部分详细介绍了实际使用的数据增强方法。这是理解它们如何将复杂的理论思想转化为可执行代码的关键。

详细分析后可以发现，尽管它们的理论目标有细微差别，但在实践中，**CoMM 的介绍更为详尽和系统，而 FactorCL 的实现则更像是一个针对其理论的启发式简化**。

---

### **FactorCL (第一篇文章) 的增强方法**

FactorCL 的核心理论是“**最优多模态增强**”，特别是“**独有信息增强 (Unique Augmentation)**”，即在增强一个模态时，需要有条件地避免破坏与另一模态的共享信息。

作者在论文中主要通过 **IRFL (图像与比喻性语言)** 数据集来具体阐述他们是如何实践这一点的。

#### **1. 文本模态 (Text, `X1`)**

*   **方法**: 作者提到对文本使用了**词语掩码 (Word Masking)**。这是一种非常标准的文本增强方法，与BERT等模型的预训练方式类似。
*   **实现细节**: 在论文中，这部分没有进一步的详细参数描述，但通常这意味着随机选择一定比例的词元并用特殊标记 `[MASK]` 替换它们。

#### **2. 图像模态 (Image, `X2`) - 实践与理论的关键差异**

这是 FactorCL 增强方法的核心展示。作者对比了两种策略：

*   **策略一：独立的、标准的增强 (FACTORCL-IndAug)**
    *   **方法**: 对图像使用了一套标准的增强组合，包括**随机裁剪 (Cropping)**、**翻转 (Flipping)** 和 **色彩抖动 (Color Jittering)**。
    *   **意图**: 这代表了不考虑“独有信息增强”理论的基线方法。

*   **策略二：“独有信息增强”的实践 (FACTORCL-SSL)**
    *   **方法**: 作者在附录D.3中明确指出：“**独有信息增强简单地移除了裁剪操作 (The unique augmentation simply removes the cropping operation)**”。这意味着增强只包括**翻转 (Flipping)** 和 **色彩抖动 (Color Jittering)**。
    *   **意图与简化**: 这是对复杂理论的一个非常重要的**实践简化**。理论上，模型需要智能地识别并保留共享信息（比如不裁剪掉文本中提到的物体）。实践中，作者采用了一个简单的启发式规则：**在所有增强中，随机裁剪最有可能意外地移除一个完整的、具有语义信息的物体或场景元素**。因此，最直接、最“安全”地近似“不破坏共享信息”这一目标的方法，就是完全禁用裁剪。

**总结 (FactorCL)**:
FactorCL 在实践中并没有实现一个复杂的、能够感知另一模态内容的条件增强系统。而是通过一个简单的**操作禁用 (disabling an operation)** —— 即移除随机裁剪 —— 来近似其理论目标。这是一种务实但简化的解决方案。对于其他数据集（如视频、医疗数据），论文没有详细说明其“独有信息增强”的具体实现。

---

### **CoMM (第二篇文章) 的增强方法**

CoMM 的哲学是使用**强力的、通用的、无条件的**增强来为统一的多模态表示创造对比视图。作者在其附录B.2中，对不同数据类型使用的增强方法给出了一个**非常全面和详细的清单**。

#### **1. 图像 (Images)**
*   **数据集**: Trifeature, MM-IMDb, Vision&Touch
*   **方法**: 明确说明使用了“**默认的 SimCLR 增强** (default SimCLR augmentations)”。
*   **实现细节**:
    *   RandomResizedCrop (随机裁剪并缩放)
    *   ColorJitter (色彩抖动)
    *   RandomGrayscale (随机灰度化)
    *   GaussianBlur (高斯模糊)
    *   RandomHorizontalFlip (随机水平翻转)

#### **2. 表格数据 (Tabular Data)**
*   **数据集**: MIMIC, Vision&Touch
*   **方法**: 对每个特征分量**添加随机高斯噪声** (add a random Gaussian noise to each component)。

#### **3. 时间序列 (Time-Series)**
*   **数据集**: MOSI, UR-FUNNY, MUsTARD, MIMIC, Vision&Touch
*   **方法**: 作者特别指出，他们对时间序列的增强策略进行了基准测试（见附录C.1），最终选择了效果最好的组合。
*   **实现细节**: **高斯噪声 (Gaussian noise)** 和 **随机丢弃 (Random Dropping)** 的随机组合，丢弃比例在序列的 0% 到 80% 之间。

#### **4. 原始文本 (Raw Text)**
*   **数据集**: MM-IMDb
*   **方法**: **随机掩码** (randomly mask)，类似于 BERT。
*   **实现细节**: 随机掩码 **15%** 的输入词元。

**总结 (CoMM)**:
CoMM 的增强方法是其“简单通用”哲学的高度体现。它没有引入新的、复杂的增强理论，而是系统性地为每种数据类型选择了在单模态自监督学习中被证明**最有效、最强大的标准增强方法**。其方法描述清晰、可复现性强，并且与它的核心理论（强对比能涌现复杂交互）完全一致。

---

### **关键对比与总结**

| 特性 | FactorCL (实践中) | CoMM (实践中) |
| :--- | :--- | :--- |
| **整体哲学** | 将复杂理论**简化**为一个启发式规则。 | **系统性地应用**强大的标准方法。 |
| **图像增强** | 为了近似“独有信息增强”，**移除了随机裁剪**，只保留翻转和色彩抖动。 | 使用**全套强大的SimCLR增强**，包括随机裁剪。 |
| **文本增强** | 使用标准的词语掩码。 | 使用标准的词语掩码 (15%)。 |
| **其他模态** | 描述较少。 | 对表格、时间序列等多种模态给出了明确、具体的增强方法。 |
| **条件性** | **实践了“有条件”的思想**（通过移除一个“危险”操作）。 | **完全无条件**，对每个模态独立应用最强增强。 |

**最终结论**: 两篇论文都成功地将理论落地，但方式截然不同。FactorCL 的实践是其理论的一个巧妙但简化的代理，显示了理论与实践之间的差距。而 CoMM 的实践则是其理论的直接、有力支撑，展示了如何通过组合现有强大工具来实现一个概念上新颖的目标。CoMM 的方法在多样化数据集上的适用性和明确性上表现得更为出色。

---

那么我在思考，如何利用这种思想，将其应用到生理信号，EEG与fNIRS的研究中。我希望能够对两个模态信号做出处理，并设计网络框架，能够显式的分解出EEG与fNIRS两种模态之间冗余，独特与协同的部分，并且将每个部分与对应的生理解释联系起来。

| 信息类别 | 生理意义 | EEG | fNIRS |
| :--- | :--- | :--- | :--- |
| **冗余信息** | **大脑局部激活事件** | 事件相关电位的出现；特定频段能量显著增加。 | 几秒延迟后，含氧血红蛋白浓度上升，脱氧血红蛋白浓度下降。 |
| **独特信息** | **激活模式** | 通过区分Alpha, Beta, Gamma等频段，识别激活的模式：放松、专注、高级认知过程等不同状态。 | N/A |
| **独特信息** | **大脑代谢成本** | N/A | HbO/HbR的变化幅度直接反映了神经活动的能量消耗 |
| **协同信息** | **神经血管耦合延迟** | 提供神经电活动的时序信息 | 提供由神经活动引发的血流动力学响应过程 |
| **协同信息** | **大脑神经效率** | 提供神经活动的强度信息（如Gamma波能量） | 提供神经活动的成本（如HbO增幅）。 |

---

这是一个非常出色和前沿的研究构想。将信息分解的理论应用于 EEG 和 fNIRS 这两种互补的生理信号上，不仅在技术上极具挑战性，而且在神经科学上具有重大意义。您提供的表格已经为这项研究奠定了坚实的生理学基础。

基于 FactorCL 和 CoMM 的思想，我们可以设计一个混合框架，**该框架采用 FactorCL 的显式分解架构，并借鉴 CoMM 更完备的 PID 理论（包含协同信息）**，专门用于解决您提出的问题。

以下是一个详细的、分步骤的研究框架设计。

---

### **第一步：概念框架——混合分解模型 (Hybrid Decomposition Model)**

我们的目标是建立一个端到端的神经网络，输入是原始或预处理的 EEG 和 fNIRS 时间序列，输出是四个**在表示空间中相互正交（信息互斥）**的特征向量：$Z_R$ (冗余), $Z_{U_{eeg}}$ (EEG独有), $Z_{U_{fnirs}}$ (fNIRS独有), 和 $Z_S$ (协同)。



*   **设计哲学**:
    *   **显式分解 (Explicit Factorization)**: 我们不依赖信息“自然涌现”，而是像 FactorCL 一样，为每个信息成分设计专门的“分解头” (Decomposition Head)。
    *   **完整理论 (Complete Theory)**: 我们将 CoMM 的协同信息 `S` 纳入框架，使其比 FactorCL 更完备。
    *   **生理驱动 (Physiology-Driven)**: 整个框架的设计，从预处理到损失函数，都将围绕您提供的生理学表格进行，以确保每个分解出的成分都具有可解释性。

---

### **第二步：数据预处理与特征工程 (注入生理先验)**

为了帮助网络更容易地学习到目标信息，我们需要在输入端就提供包含强烈生理意义的特征。

*   **输入**: 对齐的 EEG 和 fNIRS 时间序列片段 (epochs)。
*   **模态增强**: 对每个 epoch 进行数据增强以用于对比学习。
    *   **通用增强**: 添加高斯噪声、时间片段掩码 (Time-series Masking)、随机丢弃 (Dropout)。
    *   **生理学启发式增强**:
        *   **EEG**: 随机改变特定频段的信噪比，模拟不同程度的认知状态。
        *   **fNIRS**: 轻微地改变血流动力学响应函数 (HRF) 的形状（如延迟或幅度），模拟个体差异。

#### **EEG 通道 (`X_eeg`)**

1.  **时域特征**: 提取**事件相关电位 (ERPs)**。对 epochs 进行平均，得到诱发电位的波形。这直接对应于冗余信息中的“事件相关电位”。
2.  **频域特征**: 使用短时傅里叶变换 (STFT) 或小波变换，将信号分解为不同频段的能量时间序列：
    *   **Delta (δ, 1-4 Hz)**: 深度睡眠。
    *   **Theta (θ, 4-8 Hz)**: 困倦、记忆。
    *   **Alpha (α, 8-12 Hz)**: 放松、闭眼静息。
    *   **Beta (β, 12-30 Hz)**: 专注、活跃思考。
    *   **Gamma (γ, >30 Hz)**: 高级认知、信息整合。
    *   这些特征是捕获 EEG **独有信息（激活模式）** 和 **协同信息（神经效率）** 的关键。

#### **fNIRS 通道 (`X_fnirs`)**

1.  **血氧浓度计算**: 使用修正的朗伯-比尔定律 (modified Beer-Lambert Law) 将原始光强数据转换为含氧血红蛋白 (HbO) 和脱氧血红蛋白 (HbR) 的浓度变化。
2.  **信号提纯**: 应用带通滤波（例如 0.01-0.1 Hz）以去除生理噪声（如心跳、呼吸）。
3.  **激活特征**: 提取 HbO/HbR 曲线的关键特征，如峰值幅度、达到峰值的时间、曲线下面积。这些特征直接对应于**冗余信息（局部激活）**、**fNIRS 独有信息（代谢成本）**和**协同信息（神经血管耦合）**。

---

### **第三步：网络框架设计**

该框架包含三个主要部分：模态编码器、分解头和融合模块。

#### **1. 模态编码器 (Modality Encoders)**

*   **`Encoder_EEG`**: 输入预处理的 EEG 特征，输出 EEG 的高维表示 `H_eeg`。由于数据是时间序列，可以使用 **Transformer Encoder** 或 **LSTM**。
*   **`Encoder_fNIRS`**: 输入预处理的 fNIRS 特征，输出 fNIRS 的高维表示 `H_fnirs`。同样，使用 **Transformer Encoder** 或 **LSTM**。

#### **2. 分解头 (Decomposition Heads)**

这是实现显式分解的核心。每个头都是一个小型多层感知机 (MLP)，将编码器输出的高维表示投影到最终的特征空间。

*   **冗余头 (Redundancy Heads)**:
    *   `Head_R_eeg`: $Z_{R_{eeg}} = \text{Head_R_eeg}(H_{eeg})$
    *   `Head_R_fnirs`: $Z_{R_{fnirs}} = \text{Head_R_fnirs}(H_{fnirs})$
    *   目标是让 $Z_{R_{eeg}}$ 和 $Z_{R_{fnirs}}$ 尽可能相似。

*   **独有头 (Unique Heads)**:
    *   `Head_U_eeg`: $Z_{U_{eeg}} = \text{Head_U_eeg}(H_{eeg})$
    *   `Head_U_fnirs`: $Z_{U_{fnirs}} = \text{Head_U_fnirs}(H_{fnirs})$

*   **协同头 (Synergy Head)**:
    *   协同信息必须从两个模态的交互中产生。因此，协同头的输入必须是两个模态信息的**融合**。
    *   **融合模块 (Fusion Module)**: 可以使用简单的拼接 (Concatenation) `[H_eeg, H_fnirs]` 或者更复杂的交叉注意力 (Cross-Attention) 机制来融合信息，得到融合表示 `H_fused`。
    *   `Head_S`: $Z_S = \text{Head_S}(H_{fused})$

---

### **第四步：损失函数设计 (核心驱动力)**

总损失是多个子损失的加权和：$\mathcal{L}_{total} = \lambda_R\mathcal{L}_R + \lambda_U\mathcal{L}_U + \lambda_S\mathcal{L}_S + \lambda_{orth}\mathcal{L}_{orth}$

#### **1. 冗余损失 ($\mathcal{L}_R$)**

*   **目标**: 强制 `EEG冗余头` 和 `fNIRS冗余头` 的输出表示相同的信息。
*   **方法**: 使用**跨模态对比损失 (Cross-modal Contrastive Loss)**。将同一时间片段的 $(Z_{R_{eeg}}, Z_{R_{fnirs}})$ 作为正样本对，不同时间片段的作为负样本对。
    *   $\mathcal{L}_R = \text{InfoNCE}(Z_{R_{eeg}}, Z_{R_{fnirs}})$
*   **生理连接**: 这个损失会迫使网络找到 EEG 的 ERPs/频段能量增加 和 fNIRS 的血氧变化之间的**共同模式**，即您表格中定义的“大脑局部激活事件”。

#### **2. 独有损失 ($\mathcal{L}_U$)**

*   **目标**: 确保 $Z_{U_{eeg}}$ 只包含 EEG 的独有信息，反之亦然。
*   **方法**: 使用**信息最小化/重构**的组合。
    *   **重构损失**: 独有表示应该能重构其来源模态的重要特征。例如，`Decoder_eeg` 应该能从 $Z_{U_{eeg}}$ 中重构出 Alpha/Beta/Gamma 的相对比例。
        *   $\mathcal{L}_{U_{recon}} = \text{MSE}(\text{Decoder}_{eeg}(Z_{U_{eeg}}), \text{EEG_bands})$
    *   **跨模态信息最小化**: 独有表示不应包含另一模态的任何信息。可以通过一个对抗性分类器或梯度反转层 (Gradient Reversal Layer) 实现，或者简单地在对比学习中将其视为强负样本。
        *   $\mathcal{L}_{U_{neg}} = -\log(1 - \text{sim}(Z_{U_{eeg}}, H_{fnirs}))$
*   **生理连接**:
    *   EEG 独有损失将驱动 $Z_{U_{eeg}}$ 专注于频段信息，即“激活模式”。
    *   fNIRS 独有损失将驱动 $Z_{U_{fnirs}}$ 专注于 HbO/HbR 的幅度信息，即“大脑代谢成本”。

#### **3. 协同损失 ($\mathcal{L}_S$)**

*   **目标**: 捕获只有在 EEG 和 fNIRS 结合时才出现的信息。
*   **方法**: 借鉴 CoMM 的核心思想。对同一个输入样本进行**两次不同的增强**，得到两个不同的融合表示 $H'_{fused}$ 和 $H''_{fused}$，然后通过协同头得到 $Z'_S$ 和 $Z''_S$。
    *   $\mathcal{L}_S = \text{InfoNCE}(Z'_S, Z''_S)$
*   **生理连接**: 这个损失迫使网络学习 EEG 的高时间精度和 fNIRS 的血流动力学信息之间的**联合分布和时序关系**。这正是捕获“神经血管耦合延迟”和“大脑神经效率”等协同信息的理想方式。

#### **4. 正交损失 ($\mathcal{L}_{orth}$) (至关重要)**

*   **目标**: 确保 R, U, S 四个子空间在信息上是互斥的。
*   **方法**: 最小化不同信息成分表示之间的余弦相似度（或点积），强制它们在表示空间中**正交**。
    *   $\mathcal{L}_{orth} = |\text{sim}(Z_{R_{eeg}}, Z_{U_{eeg}})| + |\text{sim}(Z_{R_{eeg}}, Z_{S})| + |\text{sim}(Z_{U_{eeg}}, Z_{S})| + \dots$ （包含所有不同成分对）
*   **作用**: 这是实现**显式分解**的最强约束。它防止了信息泄露，确保例如冗余头不会学到独有信息。

---

### **第五步：验证与解释**

在模型训练完成后，最关键的一步是验证分解出的成分是否真的对应于预期的生理意义。

1.  **下游任务验证 (Downstream Task Validation)**:
    *   **冗余验证**: 使用 $Z_{R_{eeg}}$ 或 $Z_{R_{fnirs}}$ 单独进行事件检测任务（如刺激出现 vs. 基线），两者应表现相似且优异。
    *   **EEG 独有验证**: 使用 $Z_{U_{eeg}}$ 进行认知状态分类（如放松 vs. 专注），表现应优于其他任何单一成分。
    *   **fNIRS 独有验证**: 将 $Z_{U_{fnirs}}$ 的范数（magnitude）与任务难度或被试的主观疲劳度进行相关性分析，应存在显著相关。
    *   **协同验证**: 在需要高神经效率的复杂认知任务中（如工作记忆任务），使用 $Z_S$ 预测被试的反应时或正确率，其预测能力应超过其他成分的简单线性组合。

2.  **可视化与编码分析 (Visualization and Encoding Analysis)**:
    *   使用 t-SNE 或 UMAP 对大量的 $Z$ 向量进行降维可视化，并根据任务条件或认知状态进行着色，观察不同成分空间中是否存在有意义的聚类。
    *   训练简单的线性回归模型，看能否从 $Z_{U_{eeg}}$ 中解码出 Gamma 频段的能量，或者从 $Z_S$ 中解码出 EEG 信号峰值与 fNIRS 信号峰值之间的时间差。

通过这个完整的框架，您不仅能够实现对 EEG 和 fNIRS 信号的显式信息分解，还能系统性地验证每个分解成分的生理学意义，从而在神经科学和机器学习的交叉领域做出开创性的贡献。

