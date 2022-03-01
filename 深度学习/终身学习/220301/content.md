# 递归梯度优化的持续学习

在不忘记先前知识的情况下按顺序学习多个任务，称为持续学习 (Continual Learning, CL)，仍然是神经网络长期面临的挑战。大多数现有方法依赖于额外的网络容量或数据重放。相比之下，我们引入了一种称为递归梯度优化 (Recursive Gradient Optimization, RGO) 的新方法。 RGO 由一个迭代更新的优化器和一个虚拟特征编码层 (Feature Encoding Layer, FEL) 组成，该优化器修改梯度以在没有数据重放的情况下最大限度地减少遗忘，以及一个仅使用任务描述符表示不同网络结构的虚拟特征编码层 (FEL)。实验表明，与基线相比，RGO 在流行的连续分类基准上具有明显更好的性能，并在 20-split-CIFAR100 (82.22%) 和 20-split-miniImageNet (72.63%) 上实现了新的最先进的性能。该方法具有比单任务学习 (Single-Task Learning, STL) 更高的平均准确率，灵活可靠，可为依赖梯度下降的学习模型提供持续学习能力。

## 引言

在许多应用场景中，需要在不访问历史数据的情况下学习一系列任务，称为持续学习。 尽管随机梯度下降 (Stochastic Gradient Descent, SGD) 的变体对神经网络在许多领域取得的进展做出了重大贡献，但这些优化器需要小批量数据来满足独立同分布 (independent identically distributed, i.i.d.) 假设。 在持续学习中，违反此要求会导致先前任务的性能显着下降，称为灾难性遗忘。 最近的工作试图通过从各种角度修改训练过程来解决这个问题。

**Memory-based**: 基于内存的方法使用额外的内存来存储一些样本 (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2020)、梯度 (Chaudhry et al., 2019a; 2020; Saha et al., 2021) 或它们的生成模型 (Shin et al., 2017; Shen et al., 2020) 来修改未来的训练过程。重放的内存导致空间复杂度相对于任务数量线性增加。

**Expansion-based**: 基于扩展的方法动态选择网络参数(Yoon et al., 2018; Rosenbaum et al., 2018; Serra et al., 2018; Kaushik et al., 2021)，随着新任务的到来添加额外的组件 (Rusu et al., 2016; Fernando et al., 2017; Alet et al., 2018; Chang et al., 2019; Li et al., 2019)，或使用更大的网络来生成网络参数 (Aljundi et al., 2017; Yoon et al., 2019; von Oswald et al., 2019)。这些方法通过额外的任务特定参数减少任务之间的干扰。单任务学习  (Single-Task Learning, STL)  也可以看作是一种基于扩展的方法，它分别为每个任务训练一个网络。

**Regularization-based**: 基于正则化的方法通过将二次惩罚项引入损失函数 (Kirkpatrick et al., 2017; Zenke et al., 2017; Yin et al., 2020)  或约束参数更新的方向 (Farajtabar et al., 2019; Chaudhry et al., 2019a; Saha et al., 2021) 。**我们的方法也是基于正则化的，它结合了损失惩罚和梯度约束的优点。**

在这项工作中，我们专注于在没有数据重放的固定容量网络中持续学习。主要贡献如下：
- 为了在不降低当前任务性能的情况下最小化过去任务总损失的预期增量。为此提出了二次损失估计的上限，并设计了一个递归优化程序，将梯度的方向修改为该上限下的最优解。
- 根据当前任务优先 (current-task-first, CFT) 的原则，引入迹归一化过程来保证训练过程中的学习率。这种标准化过程使我们的方法与绝大多数为单任务解决方案精心设计的现有模型和学习策略兼容。由于梯度修改过程独立于数据样本和先前的参数，我们的优化器可以直接在大多数深度架构网络中用作典型的单任务优化器，如 SGD。
- 为了减少任务之间的干扰，开发了一种特征编码策略来表示网络的多模态结构，而无需额外的参数。在每个真实层之后附加一个虚拟特征编码层 (feature encoding layer, FEL)，它使用整数任务描述符作为种子随机排列输出特征图。因此，每个任务在相同的网络参数下获得特定的虚拟结构。由于网络的参数空间没有改变，这样的策略不会改变神经网络的拟合能力。
- 几个持续学习基准的实验验证表明，与现有的固定容量基线相比，所提出的方法具有显着更少的遗忘和更高的准确性。在 20-split-CIFAR100 (82.22%) 和 20-split-miniImageNet (72.62%) 上实现了最先进的性能。除了最小化遗忘之外，这种方法与单独处理所有任务的单任务学习相比具有相当或更好的性能。

## 预先知识

考虑 $K$ 个顺序到达的监督学习任务 $\{T_k|k\in [K]\}$，其中 $[N]:=\{1,2,...,N\}$ 代表任何小于等于 $N$ 的正整数。在每个任务 $T_k$ 中，有 $n_k$ 个数据点 $\{(x_{k,i}, y_{k,i})|i \in [n_k]\}$ 从未知分布 $D_k$ 中采样。令 $X$、$Y$ 和 $W$ 分别是输入、目标和模型的参数空间。通过将预测变量表示为 $f(x,k): X \times [K] \rightarrow Y$，与数据点 $(x, y)$ 和任务标识符 $k$ 相关的 $\theta \in W$ 的损失函数可以表示为 $l(f(\theta;x,k),y):W \rightarrow \mathbb{R}$，任务 $T_k$ 的经验损失函数定义为：

$$
\tag{1}
L_k(\theta) = \frac{1}{n_k}\sum_{i=1}^{n_k} l(f(\theta;x_{k,i},k),y_{k,i})
$$

在这项工作研究的持续学习场景中，参数空间 $W$ 保持固定大小，并且在训练和测试时都提供整数任务描述符。在不访问过去样本的情况下，我们使用二阶泰勒展开来估计先前任务的损失函数。令 $\theta ^{*}_{j}$ 为根据 $\nabla _{\theta ^{*}_{j}} L_j$ 梯度下降过程生成的 $L_j(\theta)$ 的最优参数。对于 $\theta ^{*}_{j}$ 邻域的新模型参数 $\theta$，先前任务 $T_j (j < k)$ 的损失可以估计为：

$$
\tag{2}
L_j(\theta) = L_j(\theta ^{*}_{j}) + \frac{1}{2}(\theta - \theta ^{*}_{j})^{\intercal}H_j(\theta - \theta ^{*}_{j})
$$

其中，$H_j:=\nabla ^{2} L_j(\theta ^{*}_{j})$ 是 Hessian 矩阵。

## 问题表述和解决方案

在我们的固定容量持续学习环境中，核心目标是找到一个在任务序列中运行良好的适当联合解决方案。 为此，我们引入了一种新颖的持续学习优化问题和相应的迭代优化策略。

### 优化问题

在本文中，我们将遗忘形式化为旧任务损失的增量。 如第 2 章所述，在 $T_k$ 之前的任务的总损失，表示为 $F_k$，可以通过以下方式估计：

$$
\tag{3}
F_k(\theta) = \sum_{j=1}^{k-1} L_j(\theta) \approx \sum_{j=1}^{k-1}[L_j(\theta ^{*}_{j}) + \frac{1}{2}(\theta - \theta ^{*}_{j})^{\intercal}H_j(\theta - \theta ^{*}_{j})]
$$

由于公式 (3) 需要先前模型参数的显式值 (explicit value)，因此计算 $Fk$ 太昂贵而不能成为持续学习的优化目标。 我们转向一种更简洁的形式，我们称之为递归最小损失 (recursive least loss, RLL)：

$$
\tag{4}
F_k^{RLL}(\theta) :=\frac{1}{2}(\theta - \theta ^{*}_{j})^{\intercal}(\sum_{j=1}^{k-1}H_j)(\theta - \theta ^{*}_{j})
$$

在附录 A.2 中，我们证明如果所有先前的任务都经过充分训练，则 $F_k^{RLL}$ 和 $F_k$ 在优化方面是等价的。 基于以上结论，任务 $T_k$ 期间的优化问题形式化为：

$$
\tag{5}
\theta ^{*}_{k}: \; \underset{\theta}{\min}F_k^{RLL},\;\;\;\mathrm{subject\;to\;}\nabla L_k(\theta)=0
$$

$F_k^{RLL}$ 与许多基于正则化的方法中的正则化项具有相同的形式。 这些方法的优化目标是 $L_k(\theta) + \lambda F_k^{RLL}(\theta)$ 的变体，源自使用高斯先验的贝叶斯后验逼近 (Kirkpatrick et al., 2017; Nguyen et al., 2018) 或使用自然梯度下降的 KL 散度逼近 (Amari, 1998; Ritter et al., 2018; Tseran et al., 2018)。贝叶斯方法试图估计和最小化整体损失函数，而我们的方法通过 $\nabla L_k(\theta)=0$ 优先考虑当前任务的性能，并最小化过去任务的预期遗忘 $F_k^{RLL}(\theta)$。

### 梯度修改

对于最新的任务 $T_k$，$\nabla L_k(\theta)=0$ 的最优解应该是从任务 $T_{k-1}$ 结束时的前一个最优模型参数 $\theta_{k-1}^{*}$ 开始的随机梯度下降得到的。 用下标 $i$ 表示第 $i$ 步的参数，初始状态 $\theta_{0} = \theta_{k-1}^{*}$，单步更新可以表示为：

$$
\theta_{i} = \theta_{i-1} - \eta_i \nabla L_k(\theta_{i-1})
$$

假设预先设定的学习率 $\eta_i$ 小到可以忽略高阶项，则一步更新后的损失函数可以表示为：

$$
L_k(\theta_{i}) = L_k(\theta_{i-1}) - \eta_i (\nabla L_k(\theta_{i-1}))^{\intercal} \nabla L_k(\theta_{i-1})
$$

如果我们只希望解决任务 $T_k$，根据上面的梯度更新参数 $\theta$ 就足够了。 然而，如上所述，这样的方法会鼓励神经网络逐渐忘记旧的任务。因此，我们修改更新方向以最小化遗忘的期望。 为此，我们引入了一个新的具有适当维度的正定对称矩阵 $P$ 来修改梯度 ($g \rightarrow P_g$)。 修改后的一步更新为：

$$
\tag{6}
\begin{aligned}
\left\{\begin{matrix}
&\theta_{i} = \theta_{i-1} - \eta_i P\nabla L_k(\theta_{i-1})\\ 
&L_k(\theta_{i}) = L_k(\theta_{i-1}) - \eta_i (\nabla L_k(\theta_{i-1}))^{\intercal} P\nabla L_k(\theta_{i-1})
\end{matrix}\right.
\end{aligned}
$$

为了在持续学习问题中保持预先设定的学习率，避免重复选择超参数，我们对投影矩阵的迹施加了额外的约束，并证明了相应的收敛速度一致性定理 1。

**定理 1**:  (收敛率一致性) 在 $trace(P) = dim(P)$ 的约束下，未知各向同性分布的学习率期望与原优化器相同。

如上所述，我们方法的唯一额外内存是投影矩阵 $P$，其中包含先前任务的信息。 这使得我们的方法成为一种空间不变的方法，与典型的基于记忆或基于扩展的持续学习方法不同。 由于我们方法的性能是通过 P 的选择来确定的，因此以下问题是：如何找到一个好的投影矩阵？ 我们将在以下部分回答这个问题。

### 近似解决方案

下一步是找到公式 (5) 的解决方案。 当 $T_k$ 的训练过程结束时，最终状态 θ∗k 和残差损失可以通过一步更新的累加得到：

$$
\tag{7}
\begin{aligned}
\left\{\begin{matrix}
&\theta_{k}^{*} = \theta_{k-1}^{*} - \sum_{i=1}^{n_k}\eta_i P\nabla L_k(\theta_{i-1})\\ 
&L_k(\theta_{k}^{*}) = L_k(\theta_{k-1}^{*}) - \sum_{i=1}^{n_k}\eta_i (\nabla L_k(\theta_{i-1}))^{\intercal} P\nabla L_k(\theta_{i-1})
\end{matrix}\right.
\end{aligned}
$$

这样，对于给定的样本序列和初始值 $\theta_{k-1}^{*}$，$\theta_{k}^{*}$ 的结果取决于 $P$。$\theta_{k}^{*}$ 上的优化公式 (5) 转化为 $P$ 上的优化问题。

然而，$F_k^{RLL}(\theta_{k}^{*})$ 和 $P$ 之间的关系过于复杂，无法用于优化过程。 为了解决这个问题，我们提出了 $F_k^{RLL}(\theta_{k}^{*})$ 的上限作为实际的优化目标。

**定理 2**: (上限) 将 $\hat{\sigma}_{m}(\cdot)$ 表示为最大特征值的符号，将 $\eta_m$ 表示为最大单步学习率，递归最小损失有一个上限：

$$
\tag{8}
F_k^{RLL}(\theta_{k}^{*}) \leq \frac{1}{2}n_k\eta_m \hat{\sigma}_{m}(P\bar{H})L_k(\theta_{k-1}^{*})
$$

其中 $\bar{H} = \sum_{j=1}^{k-1}H_j$ 定义为所有旧任务的 Hessian 矩阵之和。

丢弃常数项，我们得到投影矩阵 $P$ 的替代优化问题：

$$
\tag{9}
\begin{aligned}
P: \left\{\begin{matrix}
&\underset{p}{\min}\;\hat{\sigma}_{m}(P\bar{H})\\ 
&\mathrm{subject\;to\;} trace(P) = dim(P) 
\end{matrix}\right.
\end{aligned}
$$

标准化的解决方案是：

$$
\tag{10}
P=\frac{dim(\bar{H})}{trace(\bar{H}^{-1})}\bar{H}^{-1}
$$

归一化投影过程可以描述为：找到一个对当前任务有相似影响的新梯度，以最小化旧任务损失的上限。 我们的优化器只修改梯度的方向而不减少搜索区域，这将保证网络的拟合能力在整个任务序列中保持一致。

## 实现

### 虚拟特征编码层 Virtual FEL

在多层网络中，前一层的输出可以看作是由特征提取器生成的一组特征 (例如，权重矩阵、偏置向量、卷积核等)。 为了使反向传播过程中生成的梯度符合我们的各向同性分布假设 (定理 1)，我们提出了一个虚拟特征编码层 (FEL)，将任务特定的连接应用于前一层的输出和输入的下一层。

**定义 4.1** (特征编码层) 特征编码层对输入特征图应用特定于任务的重新排列，其顺序是使用任务标识符作为种子随机生成的。

请注意，FEL 只是现有特征图的排列，其顺序在训练过程中不会改变。 虽然这个特征编码层不需要额外的空间，但为了方便理论分析，我们将其写成矩阵形式。 层 $l$ ($l = 1, 2,..., L$) 的置换矩阵为：

$$
S_l(k):=\mathrm{random\;permutation\;of\;} I_l \mathrm{\;with\;} seed(k)
$$

其中 $I_l$ 是一个单位矩阵，其维度等于特征图的数量。 考虑到特征的顺序对于下一层获得可解释信息至关重要，如果没有正确的特征顺序，网络的识别能力将大大降低。 FEL 提供了一种有效的方法来消除任务之间的干扰，其中由特定任务描述符编码的特征将在其他任务中随机排列。 因此，同一个特征提取器在不同的任务中扮演着不同的角色。 虽然不同层的梯度在当前任务中具有很强的相关性，但从旧任务的角度来看几乎没有相关性。 这意味着当前对过去任务的影响可以被视为来自不同局部特征提取器的影响的独立总和。

### 局部-全局对等

然后，在任务 $T_k$ 的反向传播过程中，$L_k$ 在中间层 $h_l$ 上的梯度可以通过链式法则计算：

$$
\tag{11}
g_l=\frac{\partial L_k}{\partial h_l}=\frac{\partial L_k}{\partial h_L}\prod_{j=l}^{L-1}\frac{\partial h_{j+1}}{\partial h_j} = g_L\prod_{j=l}^{L-1}S_{j+1}(k)D_{j+1}W_{j+1}
$$

其中 $D_{j+1}$ 是一个对角矩阵，表示非线性激活函数的导数。 由于无法访问先前的样本来重新计算梯度，因此常见的基于梯度的方法 (Li et al., 2019; Azizan et al., 2019; Farajtabar et al., 2019) 假设联合最优参数位于先前的最优参数并使用先前计算的梯度作为近似值，这导致 $\nabla f(\theta;x) \approx \nabla f(\theta^{*};x)$ 和 $\frac{\partial^2 h_L}{\partial h_l^2} \approx 0$。我们遵循这个假设并使用建议的优化器来确保这一点尽可能满足假设。 因此我们有：

$$
\tag{12}
\frac{\partial ^2 L_k}{\partial h_l\partial h_r} = (\frac{\partial h_L}{\partial h_l})^{\intercal}\frac{\partial^2 L_k}{(\partial h_L)^2}(\frac{\partial h_L}{\partial h_r})
$$

$$
\tag{13}
\bar{H}_l = \sum_{j=1}^{k-1} \frac{\partial ^2 L_j}{\partial h^2_l} = \sum_{j=1}^{k-1}\sum_{i=1}^{n_j}(\frac{\partial h_L}{\partial h_l})^{\intercal}{l}''(f(\theta;x);y)\frac{\partial h_L}{\partial h_l} +\alpha I_l
$$

其中 $\alpha$ 是一个惩罚参数，以确保 $\bar{H}_l$ 的正定性。 根据第 3 章节中提出的迹归一化，我们在所有后续部分中设置 $\alpha=1$。

**定理 3** (局部-全局等价) 在紧邻假设下，全局优化问题等价于独立的局部优化问题。 第 $l$ 层的局部最优投影矩阵为：

$$
P_l=\frac{dim(\bar{H}_l)}{trace(\bar{H}^{-1}_l)}\bar{H}^{-1}_l, \mathrm{\;where\;}\bar{H}_l = \sum_{j=1}^{k-1}H_{j,l}
$$

### 迭代更新

考虑到计算 $n$ 维逆矩阵的复杂度为 $\mathrm{O}(n^3)$，实际计算 Hessian 逆矩阵 $\bar{H}^{-1}_l$ 是比较耗时的。 相反，我们像递归最小二乘 (RLS) (Haykin, 2002) 算法一样在训练步骤迭代地更新投影矩阵 $P_l$（更多细节请参见附录 B.1）。 这使得 RGO 成为一种在线算法，在模型参数的数量上具有线性内存复杂度和单步时间复杂度。 我们将投影矩阵的梯度修改和迭代更新总结为算法 1。

请注意，同一层中的特征提取器共享一个投影矩阵，该投影矩阵是考虑到它们的线性相关性，由梯度的平均值计算得出。 这样，在同一层处理多个梯度不会增加更新投影的复杂度。 我们在附录 B.2 中列出了不同类型特征提取器的内存大小和单步时间复杂度。 值得一提的是，由于不同层的局部优化器是独立的，在得到反向传播梯度后，可以并行处理不同层的梯度修改过程，进一步减少所需时间。

![](https://raw.githubusercontent.com/Lan-ping/blog_content/main/blogImg/alg_1.jpg)

