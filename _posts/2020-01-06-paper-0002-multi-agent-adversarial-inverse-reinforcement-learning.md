---

layout: post

title: "Multi-Agent Adversarial Inverse Reinforcement Learning"

category: summary

tags:
  - multi-agent
  - inverse reinforcement learning
  - imitation learning
  - reward recovery

comments: true

---


## Authors
- Lantao Yu
- Jiaming Song
- Stefano Ermon



## Abstract
Reinforcement learning agents are prone to undesired behaviors due to reward mis-specification.
Finding a set of reward functions to properly
guide agent behaviors is particularly challenging in multi-agent scenarios. Inverse reinforcement learning provides a framework to automatically acquire suitable reward functions from expert demonstrations. Its extension to multi-agent
settings, however, is difficult due to the more
complex notions of rational behaviors. In this
paper, we propose MA-AIRL, a new framework
for multi-agent inverse reinforcement learning,
which is effective and scalable for Markov games
with high-dimensional state-action space and unknown dynamics. We derive our algorithm based
on a new solution concept and maximum pseudolikelihood estimation within an adversarial reward learning framework. In the experiments, we
demonstrate that MA-AIRL can recover reward
functions that are highly correlated with ground
truth ones, and significantly outperforms prior
methods in terms of policy imitation.

## Summary


### Implementation detail
I read [authors' released code](https://github.com/ermongroup/MA-AIRL) and summarize their implementation details as follows:



## Overall Score
- *NOTE*: Minimum score is 1.0. If there's been no assessment, score is 0.0.
- Review Assessment
  - Thoroughness In Paper Reading: 3.0 / 5.0
  - Level of Understanding: 3.0 / 5.0
  - Checking Correctness Of Derivations And Theory: 3.0 / 5.0
  - Checking Correctness Of Experiments: 3.0 / 5.0
- Novelty: 4.0 / 5.0
- Readability: 3.0 / 5.0
- Reproducibility: 2.0 / 5.0
