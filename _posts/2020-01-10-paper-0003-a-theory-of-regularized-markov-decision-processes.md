---

layout: post

title: "A Theory of Regularized Markov Decision Processes"

category: summary

tags:
  - reinforcement learning
  - Markov decision process
  - regularization

comments: true

---

$$
\newcommand{\S}{\mathcal{S}}
\newcommand{\A}{\mathcal{A}}
\newcommand{\G}{\mathcal{G}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\one}{\mathbf{1}}
\newcommand{\grad}{\nabla}
\newcommand{\inner}[2]{\langle#1, #2\rangle}
\newcommand{\norm}[2]{\lVert#2\rVert_{#1}}
\newcommand{\argmax}[1]{\underset{#1}{\mathrm{argmax}}}
$$

## Summary of Contents

### Background

####  Notation
- $Y^X$
  - A set of applications from $X$ to $Y$
- $\Delta^X$
  - A set of probability distributions over a finite set $X$
- $\Delta_X^Y$
  - A set of conditional probability distributions over a finite set $Y$ conditioned on a finite set $X$
- $\inner{\cdot}{\cdot}$
  - The dot product
- $\norm{p}{\cdot}$
  - The $\ell_p$ norm
- For two functions, $f_1, f_2\in\R^X$,
  $$
    f_1\le f_2
    \Leftrightarrow
    \forall x\in X, f_1(x)\le f_2(x).
  $$

  - $<, \le, >, \ge$ among two functions are similarity defined.

#### Unregularized MDP
- $(\S, \A, P, r, \gamma)$
  - $\S$
    - A finite state space
  - $\A$
    - A finite action space
  - $P\in\Delta_{\S\times\A}^{\S}$
    - A Markovian transition probability
  - $r\in\R^{\S\times\A}$
    - A reward function
  - $\gamma$
    - A discount factor
- $\pi\in\Delta_\S^\A$
  - A policy

- An **associated state-action value function** $q(\cdot, \cdot;v)\in\R^{\S\times\A}$ for any $v\in\R^\S$ is defined as
  $$
    q(s, a;v):=r(s, a) + \gamma \E_{s'\sim P(\cdot|s, a)}v(s').
  $$
- A **Bellman evaluation operator** $T_\pi$
  $$
    \forall v\in\R^\S, \forall s\in\S,
    (T_\pi v)(s):=
    \E_{a\sim\pi(\cdot|s)}
    [
      r(s, a) + \gamma\E_{s'\sim P(\cdot|s, a)}v(s')
    ].
  $$

  - shown to be a $\gamma$-contraction in $\norm{\infty}{\cdot}$
  - has a unique fixed-point as $v_\pi$


- An **associated state-action value function** $q(\cdot, \cdot;v)\in\R^{\S\times\A}$ for any $v\in\R^\S$ is defined as
  $$
    q(s, a;v):=r(s, a) + \gamma \E_{s'\sim P(\cdot|s, a)}v(s').
  $$
  - We can represent the above Bellman operator as
    $$
      (T_\pi v)(s) = \inner{\pi(\cdot|s)}{q(s, \cdot;v)}_\A,
    $$
    where $\inner{\pi(\cdot|s)}{q(s, \cdot;v)}_\A:=\sum_{a\in\A}\pi(a|s)q(s, a;v)$.

- A **Bellman optimality operator** $T_*$
  $$
    \forall v\in\R^\S, \forall s\in\S,
    (T_*v)(s)= \max_\pi(T_\pi v)(s).
  $$
  - shown to be a $\gamma$-contraction in $\norm{\infty}{\cdot}$
  - has a unique fixed point as $v_*$

- A set $\G(v)$ of **greedy** policies w.r.t. $v$
  $$
    \pi'\in\G(v)
    \Leftrightarrow
    T_*v = T_{\pi'}v
  $$

#### Legendre-Fenchel transform
- For any strongly convex function $\Omega: \Delta^A\rightarrow\R$,
  the *Legendre-Fenchel tranform* (or *convex conjugate*) $\Omega^*:\R^\A\rightarrow\R$ of $\Omega$ is defined as
  $$
    \forall f\in\R^\A,
    \Omega^*(f):=\max_{p\in\Delta^\A}\inner{p}{f}_\A - \Omega(p)
  $$


- **Properties of Legendre-Fenchel transform.** For any strongly convex function $\Omega: \Delta^A\rightarrow\R$, the following properties hold:
  1.  **Unique maximizing argument.** $\nabla\Omega^*(f)$ is Lipschitz and satisfies
      $$
        \nabla\Omega^*(f)
        =
        \argmax{p\in\Delta^\A}
        \inner{p}{f}_\A
        -
        \Omega(p)
      $$

      - Note that $\nabla$ is used to indicate the *maximizer*.


  2.  **Boundedness.** If there are constants $L$ and $U$ such that $L\le\Omega(p)\le U, \forall p \in\Delta^\A$,
      $$
        \max_{a\in\A}f(a)-U
        \le
        \Omega^*(f)
        \le
        \max_{a\in\A}f(a)-L,
        \forall f \in\R^\A.
      $$

  3.  **Distributivity.** For any $c\in\R$ and all-one function $\one$,
      $$
        \Omega^*(f+c\one)
        =
        \Omega^*(f) + c
      $$

  4.  **Monotonicity.** For any $f_1, f_2\in\R^\A$ such that $f_1\le f_2$,
      $$
        \Omega^*(f_1) \le \Omega^*(f_2).
      $$

- Examples.
  1.  Negative entropy
      $$
        \Omega(p):=\sum_{a\in\A}p(a)\log p(a)
      $$

      - Unique maximizing argument
        $$
          \nabla\Omega^*(f)(a)=\frac{\exp f(a)}{\sum_{a\in\A}\exp f(a)}
        $$
      - Maximum
        $$
          \Omega^*(f) = \log\sum_{a\in\A}\exp f(a)
        $$

  2.  KL divergence between $p\in\Delta^\A$ and a uniform distribution $u\in\Delta^\A$
      $$
        \Omega(p)
        :=
        \sum_{a\in\A}
        p(a)\log\frac{p(a)}{u(a)}
        =
        \sum_{a\in\A}
        p(a)\log p(a)
        +\log\lvert \A\rvert
      $$

      - Unique maximizing argument
        $$
          \nabla\Omega^*(f)(a)=\frac{\exp f(a)}{\sum_{a\in\A}\exp f(a)}
        $$
      - Maximum
        $$
          \Omega^*(f) = \log\sum_{a\in\A}\frac{1}{\lvert \A\rvert}\exp f(a)
        $$

  3.  Tsallis entropy
      $$
        \Omega(p):=\frac{1}{2}(\inner{p}{p}_\A-1)
      $$

      - Unique maximizing argument: sparsemax
      - Maximum


### Regularized MDPs

#### Regularized Bellman operators

- A **regularized Bellman evaluation operator** $T_{\pi,\Omega}$
  $$
    \forall v\in\R^\S, \forall s\in\S, (T_{\pi,\Omega}v)(s):=\inner{\pi(\cdot|s)}{q(s, \cdot;v)}_\A - \Omega(\pi(\cdot|s)).
  $$

- A **regularized Bellman optimality operator** $T_{*,\Omega}$
  $$
    \forall v\in\R^\S, \forall s\in\S, (T_{*,\Omega}v)(s):=\max_{\pi\in\Delta_\S^\A}\inner{\pi(\cdot|s)}{q(s, \cdot;v)}_\A- \Omega(\pi(\cdot|s))=\Omega^*(q(s,\cdot)).
  $$

- A set $\G_\Omega(v)$ of **greedy** policies w.r.t. $v$
  $$
    \pi'\in\G_\Omega(v)
    \Leftrightarrow
    T_{*,\Omega}v = T_{\pi',\Omega}v
  $$


- Properties
  1. **Monotonicity.** For $v_1, v_2\in\R^\S$ such that $v_1\ge v_2$,
  $$
    T_{\pi,\Omega}v_1\ge T_{\pi,\Omega}v_2\\
    T_{*,\Omega}v_1\ge T_{*,\Omega}v_2\\
  $$

  2. **Distributivity.** For any $c\in\R$,
  $$
    T_{\pi, \Omega}(v+c\one)=T_{\pi,\Omega}v+\gamma c\one,\\
    T_{*, \Omega}(v+c\one)=T_{*,\Omega}v+\gamma c\one,\\
  $$
  3. **Contraction**. Both are $\gamma$-contractions in $\norm{\infty}{\cdot}$.

#### Regularized value functions

- A **regularized value function** $v_{\pi,\Omega}\in\R^\S$ of $\pi$ is the unique fixed point of the *regularized Bellman evaluation operator* $T_{\pi,\Omega}$.

- A **regularized optimal value function** $v_{*,\Omega}\in\R^\S$ of $\pi$ is the unique fixed point of the *regularized Bellman optimality operator* $T_{*,\Omega}$.

- **Theorem 1.** (uniqueness and optimality.) A greedy policy $\pi_{*,\Omega}\in\G_\Omega(v_{*,\Omega})$ is *unique*, and it is optimal, i.e.,
  $$
    v_{\pi_{*,\Omega},\Omega}=v_{*,\Omega}\ge v_{\pi,\Omega}, \forall \pi\in\Delta_\A^\S.
  $$

- **Proposition3.** (Relationship between unregularized and regularized value functions.) For constants $L$ and $U$ such that $L\le\Omega\le U$,
    $$
      v_\pi-\frac{U}{1-\gamma}\one
      \le
      v_{\pi,\Omega}
      \le
      v_\pi - \frac{L}{1-\gamma}\one,\\
      v_*-\frac{U}{1-\gamma}\one
      \le
      v_{*,\Omega}
      \le
      v_* - \frac{L}{1-\gamma}\one.
    $$


- **Theorem 2.** (Performance of optimal policy in regularized MDP in unregularized MDP.)
  $$
    v_* - \frac{U-L}{1-\gamma}
    \le
    v_{\pi_*,\Omega}
    \le
    v_*.
  $$

### Regularized Modified Policy Iteration (Reg-MPI)
- reg-MPI for short
  $$
    \pi_{k+1}=\G_\Omega(v_k),\\
    v_{k+1}=(T_{\pi_{k+1},\Omega})^mv_k
  $$

  - With $m=1$, it becomes a regularized value iteration algorithm.
  - With $m=\infty$, it becomes a regularized policy iteration algorithm.

- Related Algorithms
  -





## Overall Score
- *NOTE*: Minimum score is 1.0. If there's been no assessment, score is 0.0.
- Review Assessment
  - Thoroughness In Paper Reading: 1.5 / 5.0
  - Level of Understanding: 1.5 / 5.0
  - Checking Correctness Of Derivations And Theory: 1.5 / 5.0
  - Checking Correctness Of Experiments: 1.5 / 5.0
- Novelty: 5.0 / 5.0
- Readability: 2.5 / 5.0
- Reproducibility: 0.0 / 5.0
