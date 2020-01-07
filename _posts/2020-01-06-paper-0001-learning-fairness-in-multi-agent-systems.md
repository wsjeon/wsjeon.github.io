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

- $n$ agents, limited **resources**
  - non-excludable and rivalrous resources (common resources), e.g., CPU memory, network bandwidth
    - For more information, see [this post](https://www.reviewecon.com/rival-excludable).

- At the time step $t$, the environment's reward $r_t^i$ is assumed to be its occupied resources at $t$.
  - e.g, bandwidth allocated to agent $i$ at time $t$

- The **utility** of agent $i$ at the time step $t$
  $$
  u_t^i=\frac{1}{t}\sum_{t'=0}^tr_{t'}^i,
  $$
  - Average resources of agent $i$ until time $t$

- The **coefficient of variation (CV)** for $n$ agents at time $t$
  $$
  CV_t
  =
  \sqrt{
    \frac{1}{n-1}
    \sum_{i=1}^n
    \frac{(u_t^i-\bar{u}_t)^2}{(\bar{u}_t)^2}
  }
  =
  \sqrt{
    \frac{1}{n-1}
    \sum_{i=1}^n
    \left(
      \frac{u_t^i}{\bar{u}_t} - 1
    \right)^2
  }
  ,
  $$
  - $\bar{u}_t:=\frac{1}{n}\sum_{i=1}^nu_t^i$
  - A measure of fairness.
    - A system is said to be **fairer** if and only if CV is **smaller**.
    - $u_t^i/\bar{u}_t\rightarrow 1,\forall i$, then, $CV_t\rightarrow0$
    - See [Rajendra K Jain, Dah-Ming W Chiu, and William R Hawe. A quantitative measure of fairness and discrimination. Technical report, 1984](https://arxiv.org/abs/cs/9809099) for more detail.

- The **fair-efficient reward**
  $$
  \hat{r}_t^i
  =
  \frac{\bar{u}_t/c}{\epsilon+|u_t^i/\bar{u}_t-1|},
  $$

  - $\bar{u}_t/c$ for a constant $c>0$: the resource utilization of the system, encouraging the agent to improve efficiency
  - $\epsilon+|u_t^i/\bar{u}_t-1|$: giving punishment if agents' own utility deviates from the average one.
    - $\epsilon>0$ is to avoid zero division.

  - Each agent $i$ tries to maximizes discounted sum of fair-efficient rewards.
    $$
      F_i = \mathbb{E}\left[
        \sum_{t=0}^\infty \gamma^t \hat{r}_t^i
      \right],
    $$


## Overall Score
- *NOTE*: Minimum score is 1.0. If there's been no assessment, score is 0.0.
- Review Assessment
  - Thoroughness In Paper Reading: 1.5 / 5.0
  - Level of Understanding: 1.5 / 5.0
  - Checking Correctness Of Derivations And Theory: 1.5 / 5.0
  - Checking Correctness Of Experiments: 1.0 / 5.0
- Novelty: 3.0 / 5.0
- Readability: 2.0 / 5.0
- Reproducibility: 1.0 / 5.0
