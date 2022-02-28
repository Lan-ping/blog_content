# 批量在线持续学习的自适应正交投影

灾难性遗忘是持续学习的主要障碍。 最先进的方法之一是正交投影。 这种方法的想法是通过仅在与所有先前任务输入所跨越的子空间正交的方向上更新网络参数或权重来学习每个任务。 这确保不会干扰已学习的任务。 使用该想法的系统 OWM 与其他最先进的系统相比表现非常出色。 在本文中，作者首先讨论了在这种方法的数学推导中发现的一个问题，然后提出了一种称为 AOP（自适应正交投影）的新方法来解决它，这在批量和在线持续学习设置，无需像基于重放的方法那样保存任何以前的训练数据。

## 引言

许多技术已经提出用来解决持续学习 (Continual Learning, CL) 中的灾难性遗忘 (Catastrophic Forgetting, CF) 问题，其旨在增量学习一系列任务 (Chen and Liu 2018)。 每个任务 $i$ 由要学习的 $k_i (\geq  1)$ 个类组成。 一旦学习了一项任务，它的训练数据通常就无法再访问了。 CF 意味着在学习新任务时，需要修改为先前任务学习的参数，这可能会导致先前任务的准确率显着下降 (McCloskey 和 Cohen 1989)。

本文重点关注 CL 的一种特定设置，即类增量学习 (Class Incremental Learning, Class-IL)。 在 Class-IL 中，系统从一系列任务中逐步学习越来越多的类。 在测试时，学习模型可以将测试用例分类到任何类别，而无需提供任务 ID。 CL 的另一个主要范式是任务增量学习 (Task Incremental Learning, Task-IL)，其中为训练中的每个任务构建模型。 在测试中，为每个测试用例提供任务 ID，以便可以应用任务模型对测试用例进行分类。

本文进一步关注了最近在 OWM 系统中对 Class-IL 的正交投影方法 (Zeng et al. 2019)。 OWM 的工作原理如下：在学习每个任务时，网络参数仅在与所有先前任务的输入所跨越的子空间正交的方向上更新。 这确保不会干扰为先前任务学习的参数，因此不会导致先前任务的 CF。

与其他最先进的方法相比，OWM 的表现非常好。它在每个任务由一个类组成的场景中表现得特别好。在大多数 CL 论文中，每个任务都包含要学习的多个类。但是，增量学习一个类（相当于每个任务一个类）可能在应用程序中最常发生，因为在应用程序中，每当出现新对象/类时，人们都希望立即学习它以使系统保持最新而不是等待许多对象（或类）出现，然后一起学习它们。例如，在聊天机器人上下文中，当识别出新技能并准备好数据时，公司自然希望聊天机器人立即学习新技能，以便可以将新服务提供给用户，而无需等待和积累一些技能并一起学习它们。对于我们人类来说，每当我们遇到一个新物体时，我们都会立即学会识别它，我们从不等待看到许多新物体出现然后一起学习识别它们。每个任务一个类的学习任务也是最困难的 CL 场景，因为它具有最大数量的任务。众所周知，当任务数量增加时，CF 变得更加严重，从而导致分类精度降低。我们还应该注意，一次增量学习一个类（或每个任务一个类）是 CL 的最常见情况，因为将 n 个类作为一个任务一起增量学习可以简化为一个一个类的增量学习。不用说，OWM 和我们的方法都可以在每个任务中学习任意数量的类。

本文确定了 OWM 方法的数学推导中的一个问题，该方法通过存储基于旧任务的训练输入计算的正交投影算子来处理 CF，以确保在学习每个新任务时权重更新仅发生在与表示所有旧任务输入的子空间。为了处理矩阵可逆性问题，它在计算投影时引入了一个小的常数 $\alpha$。然而，$\alpha$ 与旧任务的输入无关，这对旧任务输入空间的估计不准确，导致性能较弱。 OWM 将 $\alpha$ 视为超参数，我们将证明它是不合适的。然后，我们提出了一种自适应正交投影（Adaptive Orthogonal Projection, AOP）方法来解决 OWM 的这个问题，该方法基于对所有旧任务数据和当前批次的一些整体考虑来计算每个训练批次的 $\alpha$ 值。修复此问题后，CL 准确度结果显着提高。实验评估表明，AOP 不仅在批处理和在线 CL 设置中显着优于原始 OWM，而且还显着优于其他现有的最先进的基线。据我们所知，它是在线 CL 的第一个梯度正交方法。

请注意，OWM 和 AOP 都不保存任何训练示例或构建数据生成器。 因此，AOP 更通用，适用于学习后无法再访问训练数据的环境。 旧数据的不可访问性可能是由于未记录的遗留数据、专有数据和数据隐私，例如在联邦学习中（Zhang et al. 2020）。

## 相关工作

由于已经讨论过 OWM，本节将重点介绍处理 CF 的其他 CL 方法。 一种流行的方法是使用正则化来确保从旧任务中学到的知识在学习新任务时受到的影响最小。 EWC 是该方法的代表 (Kirkpatrick et al. 2017)。 许多其他论文使用了相关方法 (Zenke, Poole, and Ganguli 2017; Fernando et al. 2017; Aljundi et al. 2017; Ritter, Botev, and Barber 2018; Xu and Zhu 2018; Kemker and Kanan 2018; Parisi et al. 2018; Ahn et al. 2019; Hu et al. 2021; Dhar et al. 2019; Adel, Zhao, and Turner 2020)。 作为一种正则化的知识蒸馏也被普遍使用 (Li and Hoiem 2017; Wu et al. 2019; Castro et al. 2018; Belouadah and Popescu 2019; Liu et al. 2020; Lee et al. 2019; Tao et al. 2020)。

另一种流行的方法是重放方法，它记住每个任务的少量训练示例，并将它们用于训练新任务，以保持先前任务的参数变化最小。 许多系统都采用这种方法，例如 iCaRL (Rebuffi et al. 2017)、GEM (Lopez-Paz and Ranzato 2017)、A-GEM (Chaudhry et al. 2019a)、RPSnet (Rajasegaran et al. 2019) 和其他 (Rusu et al. 2016; Wu et al. 2019; Rolnick et al. 2019; de Masson d'Autume et al. 2019; Hou et al. 2019)。 伪重放方法不是保存训练示例，而是学习数据生成器来生成先前任务的伪样本，以用于训练新任务，以确保先前学习的知识是最新的。 相关工作包括 (Shin et al. 2017; Wu et al. 2018; Kamra, Gupta, and Liu 2017; Seff et al. 2017; Hu et al. 2019; Lesort et al. 2018; Ostapenko et al. 2019; Hayes et al. 2019；von Oswald 等人 2020)。

除了上述流行的方法外，还有其他方法。 例如，Progressive Networks 通过构建独立模型并在它们之间建立联系来处理 CF (Rusu et al. 2016)。HAT (Serra et al. 2018) 和 CAT (Ke, Liu, and Huang 2020) 为每个任务学习硬注意力，以阻止每个先前任务的参数被更新。 BNS 使用强化学习 (Qin et al. 2021)。 OGD 保存先前任务的梯度 (Farajtabar et al. 2020)。

一些工作已经进行了跨任务的知识转移 (Ke, Liu, and Huang 2020; Ke et al. 2021; Schwarz et al. 2018; Fernando et al. 2017; Rusu et al. 2016 )。 终身学习下的早期技术主要执行知识转移，但不解决 CF (Chen and Liu 2014; Ruvolo and Eaton 2013; Benavides-Prado, Koh, and Riddle 2020)。

所提出的 AOP 不同于上述所有方法，因为它基于正交投影并且不存储以前的数据。 Chaudhry et al. (2020) 通过在不同的正交子空间中学习不同的任务，提出了一种正交方法。 Saha, Garg, and Roy (2021) 提出了一种正交方法，通过将新任务在正交方向上的梯度步骤引入对过去任务认为重要的梯度子空间。 但这些是 Task-IL 方法。 AOP 适用于 Class-IL。

在计算机视觉中，一些研究人员使用术语增量学习来表示 Class-IL，因为他们像任何其他 Class-IL 方法一样学习，每个任务有多个类 (Wu et al. 2019; Castro et al. 2018; Liu et al. 2020; Lee et al. 2019; Belouadah and Popescu 2019, 2020)。 一些传统的方法也可以一次学习一个类别。 他们通常会保存一些随机选择的样本或每个类的平均值。 在测试中，使用最接近样本/类均值的距离函数进行分类(Rebuffi et al. 2017; Lee et al. 2018; Javed and Shafait 2018; Bendale and Boult 2015)。 **AOP 不使用任何这些方法。**

最近，许多研究人员对在线持续学习 (online CL) 行了研究。 然而，在线 CL 方法主要使用重放策略。 例如，MIR 方法 (Aljundi et al. 2019a) 是一种基于重放的在线 CL 方法。 其主要思想是使模型能够专注于损失较大的回放缓冲区样本。 GSS (Aljundi et al. 2019b) 是另一种重放方法。 它使用梯度信息使存储在重放缓冲区中的数据多样化。 ASER (Shim et al. 2020) 也是一种基于回放的方法。 它具有受 Shapley 值理论启发的新重放缓冲区更新/检索策略。 同样，AOP 是不同的，因为它基于正交投影并且不保存任何以前的数据。

## OWM 问题

OWM 通过使用正交投影算子在学习每个新任务时处理 CF，使参数更新仅发生在与表示所有先前任务的输入的空间正交的方向上。 OWM 将之前所有任务的输入训练数据 $\mathbf{A}$ 视为之前的输入空间。 它计算正交投影 $P = \mathbf{I} - \mathbf{A}(\mathbf{A}^{\intercal}\mathbf{A})^{-1}\mathbf{A}^{\intercal}$，其中 $\mathbf{I}$ 是单位矩阵，$P \in \mathbb{R}^{d \times d}$，$d$ 是输入样本或示例的维度。$P$ 用于学习新任务，将参数更新投射到与先前任务的所有训练输入空间正交的方向上，即 $\Delta W = \mathfrak{k} P \Delta W^{BP}$ ，其中 $\mathfrak{k}$ 是学习率，$\Delta W^{BP}$ 是通过反向传播计算的梯度。 这确保了新的任务学习不会干扰之前的任务，因此不会导致 CF。 由于 CL 设置不允许记忆所有先前的数据 $\mathbf{A}$，在 OWM 中 $P$ 是增量计算的（见下文）。

## OWM 中输入空间的不准确估计

精确的正交投影算子是OWM克服CF的关键。 然而，为了避免矩阵可逆性问题，OWM 在原方程 $P = \mathbf{I} - \mathbf{A}(\mathbf{A}^{\intercal}\mathbf{A})^{-1}\mathbf{A}^{\intercal}$ 中添加了一个小的常数 $\alpha$。 所以投影算子变为 ${P}' = \mathbf{I} - \mathbf{A}(\alpha \mathbf{I} + \mathbf{A}^{\intercal}\mathbf{A})^{-1}\mathbf{A}^{\intercal}$。 OWM 将 $\alpha$ 视为超参数。 我们认为这对于精确构建投影算子 $P$ 是有问题的。

我们注意到 $\mathbf{A}^{\intercal}\mathbf{A}$ 是一个半正定矩阵，这意味着存在一个正交矩阵 $U \in \mathbb{R}^{d \times d}$ 使得：

$$
\tag{1}
U\mathbf{A}^{\intercal}\mathbf{A}U^{\intercal}=\begin{pmatrix}
 \delta_1& 0 & 0 & ... & 0 & 0 \\ 
 0 & \delta_2 & 0 &  ... & 0 & 0 \\ 
 0 & 0 & \delta_3 &  ... & 0 & 0 \\ 
 ... & ... & ... &  ... & ... & ... & \\ 
 0 & 0 & 0 &  ... & 0 & 0 \\ 
 0 & 0 & 0 &  ... & 0 & 0 
\end{pmatrix}
$$

由于缺少特征值，上述矩阵的最后几行可能全为零，从而导致可逆性问题。 如果添加一个小的常数 $\alpha$ 来解决这个问题，公式 (1) 成，

$$
\tag{2}
U(\alpha \mathbf{I} + \mathbf{A}^{\intercal}\mathbf{A})U^{\intercal}=\begin{pmatrix}
 \delta_1 + \alpha& 0 & 0 & ... & 0 & 0 \\ 
 0 & \delta_2 + \alpha& 0 &  ... & 0 & 0 \\ 
 0 & 0 & \delta_3 + \alpha&  ... & 0 & 0 \\ 
 ... & ... & ... &  ... & ... & ... & \\ 
 0 & 0 & 0 &  ... & \alpha & 0 \\ 
 0 & 0 & 0 &  ... & 0 & \alpha 
\end{pmatrix}
$$

这意味着我们得到了过去任务的输入空间 $X$ 的近似值，其中包含一个由具有正交基的常数 $\alpha$ 构成的额外空间 $\mathbf{S}$。 我们需要的是一个与先前任务的输入空间相关或包含有关信息的值，它可以提供更好的性能。添加一个与输入空间无关或仅表示局部的固定值（例如，特殊的特征值 ) 不能解决这个问题，反而会导致性能变差。 并且 OWM 使用 RLS（递归最小二乘）算法在训练期间增量计算或更新投影算子 ${P}'$，因为它无法记住 CL 设置中所有任务的输入数据 $\mathbf{A}$。

令 $W_l$ 为模型的第 $l$ 层权重/参数，其中 $l \in {0, 1, 2, ..., L}$，$L$ 是模型的总层数。 对于第 $j$ 个任务中的每个批次 $i$ $(= 1, 2, 3, ..., n_j)$，OWM 更新 $P_l \in \mathbb{R}^{d \times d}$ 的权重 $W_l$，记为 $P_l(i,j)$ 并迭代计算，

$$
\tag{3}
\begin{aligned}
P_l(i,j) &= P_l(i-1, j) - Q_l(i,j)x_{l-1}(i,j)^{\intercal}P_l(i-1, j) \\
Q_l(i,j) &= \frac{P_l(i-1, j)x_{l-1}(i,j)}{\alpha + x_{l-1}(i,j)^{\intercal}P_l(i-1, j)x_{l-1}(i,j)}\\
P_l(0,0) &= \mathbf{I} \\
P_l(0,j) &= P_l(n_{j-1},j-1)
\end{aligned}
$$

其中 $x_{l-1}(i,j)$ 是第 $l-1$ 层响应第 $j$ 个任务的第 $i$ 批次输入平均值的输出，$n_{j−1}$ 是第 $j-1$ 个任务。

由于 OWM 使用迭代的方法计算正交投影，它不同于传统的矩阵计算，它只添加一次 $\alpha$ 以避免反演问题。 在 OWM 中，由 $\alpha$ 引起的额外空间 $\mathbf{S}$ 在每次迭代中被添加到输入空间的历史中，这导致与正确方向的较大偏差。

总之，OWM 使用与输入空间无关的常数 $\alpha$ 来解决矩阵可逆性问题，但这会导致对先前任务输入空间的估计相当不准确，导致投影算子 ${P}'$ 较差，从而性能不佳。

## Proposed Solution

根据伍德伯里矩阵恒等式，可得：

$$
\tag{4}
\begin{aligned}
{P}' &=  \mathbf{I} - \mathbf{A}(\alpha \mathbf{I} + \mathbf{A}^{\intercal}\mathbf{A})^{-1}\mathbf{A}^{\intercal}\\
&= \mathbf{I} - \mathbf{A}( \mathbf{I} + \alpha ^{-1}\mathbf{A}^{\intercal}\mathbf{A})^{-1}\mathbf{A}^{\intercal}\alpha ^{-1} \\
&=\alpha (\sum _{i=1}^{n} x_ix_i^{\intercal} + \alpha \mathbf{I})^{-1}
\end{aligned}
$$

其中 $x_i \in \mathbb{R}^{d \times 1}$ 是 $\mathbf{A} = [x_1, x_2, ..., x_n]$ $\in \mathbb{R}^{n \times d}$ 的第 $i$ 个输入向量。$d$ 是输入向量的维数，$n$ 是所有先前输入向量的数量。 ${P}' $ 等价于近似相关矩阵 ${\Phi }'(n) = \sum _{i=1}^n x_ix_i^{\intercal} + \alpha\mathbf{I}$ 的求逆。 输入的原始相关矩阵 $\mathbf{A} = [x_1, x_2, ..., x_n]$ $\in \mathbb{R}^{n \times d}$ 是 $\Phi  (n)=\mathbf{A}\mathbf{A}^{\intercal} =\sum _{i=1}^n x_ix_i^{\intercal}$ 。

我们从统计学的角度提出了一个更全面的 $\Phi  (n)$ 近似值，以减少 $\alpha$ 的不良影响，这也受到了近似相关矩阵 $\Phi  (n)$ 用于跟踪 RLS 算法在非平稳环境 (Haykin 2008)。 根据 Eleftheriou 在 1986 年的证明(Eleftheriou and Falconer 1986)，我们可以给出一个近似方程 $\Phi (n)=\sum_{i=1}^n \lambda E(x_ix_i^{\intercal}) + \tilde{\Phi}(n)$，其中 $\tilde{\Phi}(n)$ 是一个 Hermitian 扰动矩阵，其各个条目由在统计上独立于输入向量 $x_i$ 的零均值随机变量表示，$\lambda$ 是加权因子。当 $n$ 很大并且 $\lambda$ 接近于单位 1 时，我们可以将 $\tilde{\Phi}(n)$ 视为一个准确定矩阵 (quasi-deterministic matrix)，因为对于较大的 $n$，可得：

$$
\tag{5}
E[||\tilde{\Phi}(n)||^2] \ll E[||\Phi(n)||^2]
$$

其中 $||\cdot||$ 表示矩阵范数。 在这种情况下，假设 $\lambda$ 为 1，可以进一步忽略扰动矩阵 $\tilde{\Phi}(n)$，从而将相关矩阵 $\Phi  (n)$ 近似为：

$$
\tag{6}
\Phi(n)\approx \sum_{i=1}^{n}E(x_ix_i^{\intercal}) \mathrm{\;for\;large\;} n \mathrm{\;and\;} \lambda = 1
$$

我们可以根据期望规则计算 $E(x_ix_i^{\intercal})$，

$$
\tag{7}
E(x_ix_i^{\intercal}) = E(x_i)E(x_i)^{\intercal} + Cov(x_i)
$$

其中 $Cov(x_i)$ 是 $x_i$ 的方差。 假设变量 $X^j \in \mathbb{R}^{d \times 1}$ 代表任务 $j$ 的训练数据，$d$ 是输入维度。 OWM 使用一个批次中所有输入的平均值作为变量 $X^j$ 的期望值的估计，然后将此估计值作为输入向量来更新投影算子。 最后得到投影算子 ${P}' = \alpha (\sum _{j=1}^{N} \sum _{i=1}^{n_j} x_i^j(x_i^j)^{\intercal} + \alpha \mathbf{I})^{-1}$，其中 $N$ 是到目前为止学习的任务数，$n_j$ 是在任务 $j$ 中的批次数 ，$x_i^j$ 是任务 $j$ 的第 $i$ 批次中所有输入的平均值。

我们建议根据任务的一些整体信息更新任务 $j$ 训练中每个批次的 $\alpha$ 值。 目标是将第 $j$ 个任务的输入空间的粗略估计，即 OWM 中的 $\sum_{i=1}^{n_j}x_i^j(x_i^j)^{\intercal} + \frac{\alpha}{N}\mathbf{I}$ 替换为更准确的矩阵。 我们可以将输入信号相关矩阵 $x_i^j(x_i^j)^{\intercal}$ 替换为其期望值，并使用期望规则计算其值。 但这也会有矩阵可逆性问题。作者提议将标量 $\alpha$ 的值替换为当前批次的整个期望估计的平均值，即

$$
\tag{8}
\begin{aligned}
\alpha _i^j &= \mathrm{average}(E(x_i^j(x_i^j)^{\intercal})) \\
                  &= \frac{||E(x_i^j(x_i^j)^{\intercal})||_{1,1}}{d^{2}}
\end{aligned}
$$

其中操作 $\mathrm{average}(Y)$ 是得到矩阵 $Y$ 中所有条目的平均值，$d$ 是矩阵 $E(x_i^j(x_i^j)^{\intercal})$ 的维数。

最终提出的计算正交投影算子的方法是依次计算 $P$ 和 $α$，得到 $P^{*}=(\sum_{j=1}^{N}\sum_{i=1}^{n_j}(x_i^j(x_i^j)^{\intercal} + \frac{\alpha_i^j}{n_j} \mathbf{I}))^{-1}$。 当训练的批次数量较少时，$x_i^j(x_i^j)^{\intercal} + \frac{\alpha_i^j}{n_j} \mathbf{I} \approx x_i^j(x_i^j)^{\intercal} + \alpha_i^j \mathbf{I}$，比原始 OWM 中的方法更准确地估计输入空间，因为我们估计了从相关矩阵 xj i (xj i )T 和输入向量的统计特性表示输入空间。对于提出的相关矩阵 $\Phi ^{*}(n) = \sum_{j=1}^{N}\sum_{i=1}^{n_j}(x_i^j(x_i^j)^{\intercal} + \frac{\alpha_i^j}{n_j} \mathbf{I})$ 和原始相关矩阵 $\Phi (n) = \sum_{j=1}^{N}\sum_{i=1}^{n_j}x_i^j(x_i^j)^{\intercal}$，我们有以下命题：

**命题 1 **：对于任何矩阵范数 $||\cdot||$，我们有：

$$
\tag{9}
||\Phi ^{*}(n) - \Phi(n)|| \leq ||T(\Phi(n))||
$$

其中 $T(\Phi(n))$ 是一个对角矩阵，其单个对角元素值是所有输入 $x_i^j(x_i^j)^{\intercal}$ 的奇异值之和的平均值。 证明在附录 1 中给出了。

作者的新方法可以与RLS算法和期望规则相结合，解决矩阵可逆问题，并在**经验上**取得更好的性能。 也就是说，我们改变第 $l$  层权重 $W_l$ 的更新规则 $P_l \in \mathbb{R}^{d \times d}$。 当我们在任务 $j$ 中获得第 $i$ 个批次的输入时，我们更新 $P_l(i, j)$ 如下 (请注意由于使用层号 $l$ 导致的符号变化)：

$$
\tag{10}
\begin{aligned}
P_l(i,j) &= P_l(i-1, j) - Q_l(i,j)x_{l-1}(i,j)^{\intercal}P_l(i-1, j) \\
Q_l(i,j) &= \frac{P_l(i-1, j)x_{l-1}(i,j)}{\alpha_{i}^{j} + x_{l-1}(i,j)^{\intercal}P_l(i-1, j)x_{l-1}(i,j)}\\
\alpha_{i}^{j} &= \mathrm{average}(E(x_{l-1}(i,j)x_{l-1}(i,j)^{\intercal})) \\
                      &= \mathrm{average}(E(x_{l-1}(i,j))E(x_{l-1}(i,j)^{\intercal}) + Cov(x_{l-1}(i,j)))) \\
                      &\approx \mathrm{average}(x_{l-1}(i,j)x_{l-1}(i,j)^{\intercal} + Cov(x_{l-1}(i,j)))) \\
P_l(0,0) &= \mathbf{I} \\
P_l(0,j) &= P_l(n_{j-1},j-1)
\end{aligned}
$$

其中 $x_{l−1}(i,j)$ 是任务 $j$ 第 $i$ 批次输入的平均值经过第 $l-1$ 层的输出，$\mathrm{average}(\cdot)$ 获取矩阵中所有条目的平均值，$n_{j-1}$ 是任务 $j-1$ 中的批次数。

任务 $j$ 中第 $i$ 批次输入通过网络后，我们计算投影算子，然后更新每一层的权重矩阵

$$
\tag{11}
\begin{aligned}
W_l(i,j) &= W_l(i-1,j) - \mathfrak{k}(i,j)  \Delta W_l^{BP}(i,j)  \mathrm{\;if\;} j=1\\
W_l(i,j) &= W_l(i-1,j) - \mathfrak{k}(i,j)  P_l(n_{j-1},j-1)\Delta W_l^{BP}(i,j) \mathrm{\;if\;} j=2,3,... \\
P_l(0,0) &= \mathbf{I}_l
\end{aligned}
$$

其中 $W_l(i-1,j)$ 是第 $l$ 层的权重，$\Delta W_l^{BP}(i,j)$ 是通过标准反向传播 (BP) 方法计算的权重 $W_l(i-1,j)$ 的梯度，$\mathfrak{k}(i,j)$ 是学习率，$\mathbf{I}_l$ 是一个单位矩阵，其维数与权重所在行的维数相同。

现在让我们检查在 BP 算法的后向传播中乘以投影矩阵的效果，看看为什么作者的方法有效。训练任务 $j$ 后，根据梯度下降算法，第 $l$ 个全连接（FC）层的权重矩阵 $W_l$ 变为 $W_l(n_j,j) = W_l(n_{j-1},j) - \mathfrak{k}  P_l(n_{j-1},j-1)\Delta W_l^{BP}(j)$，其中 $\mathfrak{k}$ 是学习率，$\Delta W_l^{BP}(j)$ 表示训练期间 BP 计算的梯度。 对于来自第 ${j}' ({j}' \leq  j)$ 个任务的测试集或分布的测试样本 {x}'，第 $l - 1$ 个 FC 层的输出为 ${x}'_{l-1}$。 根据 AOP 中的梯度更新规则，第 $l$ 个 FC 层 ${x}'_{l}$ 的输出可以分解为：

$$
\tag{12}
\begin{aligned}
{x}'_{l} &= (W_l(n_{j},j) - \mathfrak{k}  P_l(n_{j-1},j-1)\Delta W_l^{BP}(j)){x}'_{l-1} \\
           &= (W_l(n_{{j}'},{j}') - \sum_{i={j}'+1}^{j}\mathfrak{k}  P_l(n_{i-1},i-1)\Delta W_l^{BP}(i)){x}'_{l-1}\\
           &\approx W_l(n_{{j}'},{J}'){x}'_{l-1}
\end{aligned}
$$

其中 $n_{i−1}$ 是任务 $i−1$ 中的批次数，$\Delta W_l^{BP}(i)$ 是第 $i$ 个任务训练期间由 BP 计算的梯度。 因为在第 ${j}'$ 个任务训练后计算的任何投影算子都与第 ${j}'$ 个任务分布的输入近似正交，我们得到公式 (12) 的近似方程。 在这里，我们假设 $W_l(n_{{j}'},{j}')$ 是我们在训练第 ${j}'$ 个任务时获得的对第 ${j}'$ 个任务进行分类的最佳权重。 所以我们实际上近似的是在第 ${j}'$ 个任务的训练过程完成并且新任务还没有到来时对 ${x}'_l$ 进行分类的最优模型性能。 这样，在训练一个任务后，模型可以保持任务的性能，因为新任务的梯度更新与先前任务分布的样本的内积近似为零。

总之，根据公式 (10) 所示，每个训练批次都有一个不同的 $\alpha$ 值 (因此是自适应的)，该值是根据当前批次的均值和方差计算得出的。 这不仅解决了可逆矩阵问题，还提供了对整个空间的更准确估计，进而计算出更准确的投影仪。 根据公式 (11) 和 (12)，我们可以在优化中使用正交投影算子来避免 CF。 OWM 在整个训练过程中使用一个固定的 $\alpha$ 值。 实验结果表明，所提出的 AOP 取得了明显优于 OWM 和其他基线的结果。