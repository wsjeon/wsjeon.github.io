---

layout: post

title: "Learning Fairness in Multi-Agent Systems"

category: summary

tags:
  - multi-agent
  - reinforcement learning
  - fairness

comments: true

---

## Authors
- Jiechuan Jiang
- Zongqing Lu

## Abstract
Fairness is essential for human society, contributing to stability and productivity. Similarly, fairness is also the key for many multi-agent systems. Taking fairness into multi-agent learning could help multi-agent systems become both efficient and stable. However, learning efficiency and fairness simultaneously is a complex, multi-objective, joint-policy optimization. To tackle these difficulties, we propose FEN, a novel hierarchical reinforcement learning model. We first decompose fairness for each agent and propose fair-efficient reward that each agent learns its own policy to optimize. To avoid multi-objective conflict, we design a hierarchy consisting of a controller and several sub-policies, where the controller maximizes the fair-efficient reward by switching among the sub-policies that provides diverse behaviors to interact with the environment. FEN can be trained in a fully decentralized way, making it easy to be deployed in real-world applications. Empirically, we show that FEN easily learns both fairness and efficiency and significantly outperforms baselines in a variety of multi-agent scenarios.

## Summary

### Methods

#### Fair-Efficient Reward

- Suppose there are $n$ agents and the environment's resources are limited.
  - non-excludable and rivalrous, e.g., CPU memory, network bandwidth.

- At the time step $t$, the reward $r$ from the environment is related to its occupied resources at $t$.

- The utility of agent $i$ at the time step $t$ is defined as
  $$
  u_t^i=\frac{1}{t}\sum_{t'=0}^tr_{t'}^i,
  $$
  which is the average reward over all elapsed time steps $t'=0, ..., t$.

- The **coefficient of variation (CV)** of agents' utilities $u_1, ..., u_N$ is defined as (why?)
  $$
  \sqrt{
    \frac{1}{n-1}
    \sum_{i=1}^n
    \frac{(u^i-\bar{u})^2}{\bar{u}^2}
  },
  $$
  where $\bar{u}:=\frac{1}{N}\sum_{i=1}^Nu_i$. Note that CV is used as a measure of fairness in this work.
A system is said to be **fairer** if and only if CV is **smaller**.

- The **fair-efficient reward** is defined as
  $$
  \hat{r}_t^i
  =
  \frac{\bar{u}_t/c}{\epsilon+|u_t^i/\bar{u}_t-1|},
  $$

  - $\bar{u}_t/c$: the resource utilization of the system, encouraging the agent to improve efficiency
    - $c$ is a normalization constant and is set to the maximum environmental reward the agent obtains at a time step.
  - $|u_t^i/\bar{u}_t-1|$: a measure for the agent's utility deviation from the average and the agent will be punished no matter it is above or below the average
  - $\epsilon$: a small positive number to avoid zero denominator

- Each agent $i$ tries to maximizes
  $$
    F_i = \mathbb{E}\left[
      \sum_{t=0}^\infty \gamma^t \hat{r}_t^i
    \right],
  $$

  - $\gamma$: a discount factor

**Proposition 1.** The optimal fair-efficient policy set $\Pi^*$ is Pareto efficient in infinite-horizon sequential decision-making.

**Proposition 2.** The optimal fair-efficient policy set $\Pi^*$ achieves equal allocation when the resources are fully occupied.



## Overall Score
- *NOTE*: Minimum score is 1.0. If there's been no assessment, score is 0.0.
- Review Assessment
  - Thoroughness In Paper Reading: 1.0 / 5.0
  - Level of Understanding: 1.0 / 5.0
  - Checking Correctness Of Derivations And Theory: 1.0 / 5.0
  - Checking Correctness Of Experiments: 1.0 / 5.0
- Novelty: 0.0 / 5.0
- Readability: 0.0 / 5.0
- Reproducibility: 0.0 / 5.0
