# Temporal Teacher Pool for Self-Knowledge Distillation

## 研究背景

DTSKD 使用 Historical Teacher：

Teacher(t) = Prediction(t−1)

即当前模型仅学习上一轮训练产生的知识。

然而训练过程中会产生大量历史知识：

* Epoch 50
* Epoch 100
* Epoch 150
* Epoch 200
* Epoch 250

这些知识被直接丢弃。

已有工作 EEKD (2022) 证明：

多个历史教师融合优于单个最终教师。

因此提出：

Temporal Teacher Pool (TTP)

利用训练轨迹中的多个历史状态作为候选教师。

---

## Research Question 1

历史教师是否必须来自上一轮？

Baseline:

Teacher = P(t−1)

Proposed:

Teacher Pool

{P(50), P(100), P(150), P(200), P(250)}

实验：

* Last Teacher
* Mean Teacher Pool
* Weighted Teacher Pool

比较性能。

---

## Research Question 2

不同样本是否适合不同历史教师？

假设：

Easy Sample：

更适合后期教师。

Hard Sample：

更适合中期教师。

因此：

Teacher Selection Module

输入：

Current Logits

输出：

Teacher Index

选择：

P(50)
P(100)
P(150)
P(200)
P(250)

中的一个。

---

## Research Question 3

如何衡量历史教师质量？

候选指标：

1. Validation Accuracy
2. Prediction Entropy
3. Calibration Error
4. Diversity

构造：

Teacher Score

根据 Score 进行动态选择。

---

## 实验路线

Phase 1

复现 DTSKD。

Phase 2

构造 Teacher Pool。

Phase 3

Mean Ensemble。

Phase 4

Weighted Ensemble。

Phase 5

Teacher Selection Network。

---

## Potential Contribution

1. Temporal Teacher Pool
2. Historical Teacher Selection
3. Sample-wise Teacher Routing

目标：

突破 DTSKD 的固定历史教师设计。
