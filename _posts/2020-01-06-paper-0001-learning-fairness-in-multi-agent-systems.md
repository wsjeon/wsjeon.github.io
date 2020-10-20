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

  - A measure of fairness.

    - $\bar{u}_t=\frac{1}{n}\sum_{i=1}^{n}u_t^i$.

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
  - $\epsilon + \lvert u_t^i/\bar{u}_t-1 \rvert$: giving punishment if agents' own utility deviates from the average one.
    - $\epsilon>0$ is to avoid zero division.

  - Each agent $i$ tries to maximizes discounted sum of fair-efficient rewards.

    $$
    F_i = \mathbb{E}\left[
      \sum_{t=0}^\infty \gamma^t \hat{r}_t^i
    \right],
    $$

**Proposition 1.** The optimal fair-efficient policy set is Pareto efficient in infinite-horizon sequential decision-making.

- Optimum means we cannot increase one of $F_i$s without decreasing another $F_j$.

- The resources must be fully occupied.

    - Assume $\pi$ didn't fully use the resource. Then, we can always find out $\pi'$ that fully utilizes the resources in a way that it separates the remaining resources according to the ratio of $u^i/n\bar{u}$. Then, for the remaining resources $\delta$,

    $$
    v^i=u^i+\delta\frac{u^i}{n\bar{u}},
    $$

    $$
    \bar{v}=\bar{u}+\delta\frac{1}{n},
    $$

    $$
    \frac{v^i}{\bar{v}}-1
    =
    \frac{
      \bar{u}(\dfrac{u^i}{\bar{u}}-1)+\dfrac{\delta}{n}(\dfrac{u^i}{\bar{u}}-1)
    }{
      \bar{u}+\delta\dfrac{1}{n}
    }
    =
    \frac{u^i}{\bar{u}}-1.
    $$

    - It's natural since the resource allocation ratio is preserved.
    - However, $\bar{v}/c > \bar{u}/c$, which means $F_i'>F_i$ and $\pi$ is not optimal. That is, optimal policy should fully occupy resources.

  - Optimal policy is Pareto efficient.
    - Assume $\pi$ didn't achieve Pareto optimality, then, there must exist $\forall i, v^i\ge u^i \land \exists i, v^i > u^i$, so $\sum_{i=1}^n v^i > \sum_{i=1}^n u^i$, which contradicts optimum should fully occupy resources.


**Proposition 2.** The optimal fair-efficient policy set $\Pi^*$ achieves equal allocation when the resources are fully occupied.

- Honestly, I don't understand the proof 100%, but I guess their method is like below. Assume non-equal resource allocation, i.e., $\exists i, u_i>\bar{u}$. For those agents (single or multiple agents, let $\mathcal{I}$), let them give up their resources, i.e., $u_i=\bar{u}$. Since $\pi^*$ (optimal fair policy) should fully utilize resource, make those resourced used by the other agents (let $\mathcal{I}^C$). I haven't rigorously proved, but probably, by following the ratio-based resource allocation, it can be shown that $F_i'>F_i$ for all $i\in\mathcal{I}^C$. Additionally, since the mean $\bar{u}$ is maintained (due to the fully occupied resources), $F_i' > F_i$ for $i\in\mathcal{I}$ since $\hat{r}_t^i$ increases (see denominator). This process can be done recursively and since all procedures always increase (non-decrease) $F_i$, which contradicts the precondition that $\pi$ was optimal.

#### Hierarchy

Using hierarchical training (like option-critic architecture) is proposed to balance between efficiency (individual performance) and fairness. High-level policy (controller) chooses one of the low-level policies (called sub-policy in the paper) and uses fair-efficient reward. Low-level policies are separated into two types: a policy updated with environment's reward and other policies based on entropy-based reward ($\log p(z|o)$). Honestly, I think there's no theoretical link between implementation and suggested theory except the fact that fair-efficient reward is used. I also doubt if using hierarchical policy for each agent is necessary in this setting.

#### Decentralized Training

Authors considered decentralized training for each agent (agent-wise PPO) and hope to coordinate agents via $\bar{u}$. However, knowing exact $\bar{u}$ is problematic. They used gossip-based algorithm to solve this issue, which I think their novelty comes out.

### Experiments

I haven't read the experiments yet.



## Overall Score
- *NOTE*: Minimum score is 1.0. If there's been no assessment, score is 0.0.
- Review Assessment
  - Thoroughness In Paper Reading: 2.5 / 5.0
  - Level of Understanding: 2.5 / 5.0
  - Checking Correctness Of Derivations And Theory: 3.5 / 5.0
  - Checking Correctness Of Experiments: 0.0 / 5.0
- Novelty: 3.5 / 5.0
- Readability: 2.5 / 5.0
- Reproducibility: 2.0 / 5.0
