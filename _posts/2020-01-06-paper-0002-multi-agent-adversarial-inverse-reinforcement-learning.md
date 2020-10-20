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

## Implementation detail
I read [authors' released code](https://github.com/ermongroup/MA-AIRL) and summarize their implementation details as follows:

### Command

  `README.md`
  ```bash
  python -m irl.mack.run_mack_airl
  ```

### Hyperparameters

  `irl/mack/run_mack_airl.py`
  ```python
  @click.option('--dis_lr', type=click.FLOAT, default=0.1)
  @click.option('--disc_type', type=click.Choice(['decentralized', 'decentralized-all']),
                default='decentralized')
  @click.option('--bc_iters', type=click.INT, default=500)
  @click.option('--l2', type=click.FLOAT, default=0.1)
  @click.option('--d_iters', type=click.INT, default=1)
  @click.option('--rew_scale', type=click.FLOAT, default=0)
  ```

  `def main` in `irl/mack/run_mack_airl.py`
  ```python
  lrs = [0.1]
  batch_sizes = [1000]
  ```

### Learning module

  `irl/mack/run_mack_airl.py`
  ```python
  from irl.mack.airl import learn
  ```


### Model: Policy / Value / Relevant Optimizers

  `irl/mack/airl.py`
  ```python
  make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps=nsteps,
                             nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=vf_fisher_coef,
                             lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                             lrschedule=lrschedule, identical=identical)
  ```

  ```python
  class Model(object):
      def __init__(self, policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=2, nsteps=200,
                   nstack=1, ent_coef=0.00, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
                   kfac_clip=0.001, lrschedule='linear', identical=None):
  ```


### Discriminator

  `irl/mack/airl.py`
  ```python
  from irl.mack.kfac_discriminator_airl import Discriminator
  ```

  ```python
  Discriminator(model.sess, ob_space, ac_space,
                state_only=True, discount=gamma, nstack=nstack, index=k, disc_type=disc_type,
                scope="Discriminator_%d" % k, # gp_coef=gp_coef,
                total_steps=total_timesteps // (nprocs * nsteps),
                lr_rate=dis_lr, l2_loss_ratio=l2) for k in range(num_agents)
  ```

  - Use *state-only* rewards.
  - Use decentralized discriminators for each agent.
    - I guess centralized discriminators might not work well.
  - `dis_lr=0.1` and `l2=0.1` and `nstack=1`

  `irl/mack/kfac_discriminator_airl.py`
  ```python
  class Discriminator(object):
    def __init__(self, sess, ob_spaces, ac_spaces, state_only, discount,
                 nstack, index, disc_type='decentralized', hidden_size=128,
                 lr_rate=0.01, total_steps=50000, scope="discriminator", kfac_clip=0.001, max_grad_norm=0.5,
                 l2_loss_ratio=0.01):
  ```

  - Use 128 hidden units.
  - `kfac_clip` and `max_grad_norm` hasn't been used in `Discriminator` class. The authors only use `AdamOptimizer`.
  - Learning rate of the discriminator was decayed.
    ```python
    self.index = index
    ob_space = ob_spaces[index]
    ac_space = ac_spaces[index]
    self.all_ob_shape = sum([obs.shape[0] for obs in ob_spaces]) * nstack
    try:
        self.all_ac_shape = sum([ac.n for ac in ac_spaces]) * nstack
    except:
        self.all_ac_shape = sum([ac.shape[0] for ac in ac_spaces]) * nstack

    if disc_type == 'decentralized':
         self.obs = tf.placeholder(tf.float32, (None, self.ob_shape))
         self.nobs = tf.placeholder(tf.float32, (None, self.ob_shape))
         self.act = tf.placeholder(tf.float32, (None, self.ac_shape))
         self.labels = tf.placeholder(tf.float32, (None, 1))
         self.lprobs = tf.placeholder(tf.float32, (None, 1))
     elif disc_type == 'decentralized-all':
         self.obs = tf.placeholder(tf.float32, (None, self.all_ob_shape))
         self.nobs = tf.placeholder(tf.float32, (None, self.all_ob_shape))
         self.act = tf.placeholder(tf.float32, (None, self.all_ac_shape))
         self.labels = tf.placeholder(tf.float32, (None, 1))
         self.lprobs = tf.placeholder(tf.float32, (None, 1))
     else:
         assert False
    ```

  - If `disc_type == 'decentralized'`, observation shape is `self.ob_shape` and action shape is `self.ac_shape`. If it becomes `decentralized-all`, concatenated shapes will be used.

  - `self.nobs` means the observation in the next time step.
  - [?] What is the roles of `self.labels` and `self.lprobs`?

####  Discriminator Network Architecture

```python
with tf.variable_scope(self.scope):
    rew_input = self.obs
    if not self.state_only:
        rew_input = tf.concat([self.obs, self.act], axis=1)

    with tf.variable_scope('reward'):
        self.reward = self.relu_net(rew_input, dout=1)
        # self.reward = self.tanh_net(rew_input, dout=1)

    with tf.variable_scope('vfn'):
        self.value_fn_n = self.relu_net(self.nobs, dout=1)
        # self.value_fn_n = self.tanh_net(self.nobs, dout=1)
    with tf.variable_scope('vfn', reuse=True):
        self.value_fn = self.relu_net(self.obs, dout=1)
        # self.value_fn = self.tanh_net(self.obs, dout=1)

    log_q_tau = self.lprobs
    log_p_tau = self.reward + self.gamma * self.value_fn_n - self.value_fn
    log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
    self.discrim_output = tf.exp(log_p_tau - log_pq)
```

- Maybe,
$$
\max_\omega
\mathbb{E}_{\pi_E}\left[
  \sum_{i=1}^N\log\frac{
    \exp(f_{\omega_i}(s, a, s'))
  }{
    \exp(f_{\omega_i}(s, a, s')) + q_{\theta_i}(a_i|s)
  }
\right]
\\
+\mathbb{E}_{q_\theta}\left[
  \sum_{i=1}^N\log\frac{
    q_{\theta_i}(a_i|s)
  }{
    \exp(f_{\omega_i}(s, a, s')) + q_{\theta_i}(a_i|s)
  }
\right]
$$

  - For $f_{\omega_i}$,
    $$
    f_{\omega_i, \phi_i}(s_t, a_t, s_{t+1}) :=
    g_{\omega_i}(s_t, a_t) + \gamma h_{\phi_i}(s_{t+1}) - h_{\phi_i}(s_t).
    $$
    - For state-only rewards, $g_{\omega_i}(s_t)$.
      - In the end, $g_{\omega_i}(s_t)$ is used as a reward function, ignoring $h_{\phi_i}(s_t)$ parts.
        ```python
        def get_reward(self, obs, acs, obs_next, path_probs, discrim_score=False):
            if len(obs.shape) == 1:
                obs = np.expand_dims(obs, 0)
            if len(acs.shape) == 1:
                acs = np.expand_dims(acs, 0)
            if discrim_score:
                feed_dict = {self.obs: obs,
                             self.act: acs,
                             self.nobs: obs_next,
                             self.lprobs: path_probs}
                scores = self.sess.run(self.discrim_output, feed_dict)
                score = np.log(scores + 1e-20) - np.log(1 - scores + 1e-20)
            else:
                feed_dict = {self.obs: obs,
                             self.act: acs}
                score = self.sess.run(self.reward, feed_dict)
            return score
        ```

      - Note: Writing it as `log_p_tau` is a bit weird though I understand their intention. I think `log_exp_f` is a correct way to describe it.
- `self.lprobs` is `log probability` meaning that log probability of the agent's policy.
  - Coming from policy (fake, agent's policy)

- `self.discrim_output` was expert-side output, but I need to check it really was in the code as well.
  - Expert-side output.

- Two Layer Relu Activated Network
  ```python
  def relu_net(self, x, layers=2, dout=1, hidden_size=128):
      out = x
      for i in range(layers):
          out = relu_layer(out, dout=hidden_size, name='l%d' % i)
      out = linear(out, dout=dout, name='lfinal')
      return out
  ```

- `l2_loss` for reward regularization.
  ```python
  self.total_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))
  self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params]) * self.l2_loss_ratio
  self.total_loss += self.l2_loss
  ```
-  `self.labels=1` means `log_p_tau - log_pq` is gonna be minimized. Expert or Agent? I could find the answer below:
      ```python
      def train(self, g_obs, g_acs, g_nobs, g_probs, e_obs, e_acs, e_nobs, e_probs):
          labels = np.concatenate((np.zeros([g_obs.shape[0], 1]), np.ones([e_obs.shape[0], 1])), axis=0)
          feed_dict = {self.obs: np.concatenate([g_obs, e_obs], axis=0),
                       self.act: np.concatenate([g_acs, e_acs], axis=0),
                       self.nobs: np.concatenate([g_nobs, e_nobs], axis=0),
                       self.lprobs: np.concatenate([g_probs, e_probs], axis=0),
                       self.labels: labels,
                       self.lr_rate: self.lr.value()}
          loss, _ = self.sess.run([self.total_loss, self.d_optim], feed_dict)
          return loss
      ```
      - That is, label 1 for expert, 0 for agent.

#### Keep Away (Competitive)
- Weird thing is that `simple_tag` is conditioned in the code, but not utilized in the paper. Probably, they didn't get a desired result from this environment.

  `airl.py`

  ```python
  # add reward regularization
  if env_id == 'simple_tag':
      reward_reg_loss = tf.reduce_mean(
          tf.square(discriminator[0].reward + discriminator[3].reward) +
          tf.square(discriminator[1].reward + discriminator[3].reward) +
          tf.square(discriminator[2].reward + discriminator[3].reward)
      ) + rew_scale * tf.reduce_mean(
          tf.maximum(0.0, 1 - discriminator[0].reward) +
          tf.maximum(0.0, 1 - discriminator[1].reward) +
          tf.maximum(0.0, 1 - discriminator[2].reward) +
          tf.maximum(0.0, discriminator[3].reward + 1)
      )
  reward_reg_lr = tf.placeholder(tf.float32, ())
  reward_reg_optim = tf.train.AdamOptimizer(learning_rate=reward_reg_lr)
  reward_reg_train_op = reward_reg_optim.minimize(reward_reg_loss)
  ```

#### Log action probability
- This is needed to get $q(a|s)$ for expert and agent state-action pairs.
  `airl.py`
  ```python
  g_log_prob = model.get_log_action_prob(g_obs, g_a)
  e_log_prob = model.get_log_action_prob(e_obs, e_a)
  ```

  In `class Model`,
  ```python
  def get_log_action_prob(obs, actions):
      action_prob = []
      for k in range(num_agents):
          if identical[k]:
              continue
          new_map = {
              train_model[k].X: np.concatenate([obs[j] for j in range(k, pointer[k])], axis=0),
              A[k]: np.concatenate([actions[j] for j in range(k, pointer[k])], axis=0)
          }
          log_pac = sess.run(self.log_pac[k], feed_dict=new_map)
          if scale[k] == 1:
              action_prob.append(log_pac)
          else:
              log_pac = np.split(log_pac, scale[k], axis=0)
              action_prob += log_pac
      return action_prob

  self.get_log_action_prob = get_log_action_prob
  ```

  ```python
  logpac = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=train_model[k].pi, labels=A[k])
  self.log_pac.append(-logpac)
  ```

#### Pearson's correlation coefficient (PCC) and Spearman's rank correlation coefficient (SCC)

- The authors use `scipy.stats`.
  ```python
  from scipy.stats import pearsonr, spearmanr
  ```

  ```python
  try:
      logger.record_tabular('pearson %d' % k, float(
          pearsonr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
      logger.record_tabular('spearman %d' % k, float(
          spearmanr(report_rewards[k].flatten(), mh_true_returns[k].flatten())[0]))
      logger.record_tabular('reward %d' % k, float(np.mean(rewards[k])))
  except:
      pass
  ```







## Overall Score
- *NOTE*: Minimum score is 1.0. If there's been no assessment, score is 0.0.
- Review Assessment
  - Thoroughness In Paper Reading: 4.0 / 5.0
  - Level of Understanding: 4.5 / 5.0
  - Checking Correctness Of Derivations And Theory: 3.5 / 5.0
  - Checking Correctness Of Experiments: 4.5 / 5.0
- Novelty: 4.0 / 5.0
- Readability: 2.5 / 5.0
- Reproducibility: 3.5 / 5.0
