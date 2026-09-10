# Heterogenous Agent Collaborative Reinforcement Learning

## 1.Paper of this work.

All experiments from the paper [*“Heterogenous Agent Collaborative Reinforcement Learning”*](https://arxiv.org/abs/2603.02604) can be reproduced with this repo.

Heterogenous Agent Collaborative Policy Optimization (HACPO) is an RLVR framework designed to facilitate the collaborative training of multiple heterogeneous agents on a common task.

> **verl V1 update:** A HACPO implementation for the verl V1 training engine is available on the [`verl-v1`](https://github.com/zzx-peter/HACRL-code/tree/verl-v1) branch. The results reported in the paper correspond to the original implementation on `main`.

## 2. Key Contributions

|  Feature | What it does |
| :----------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Agent-Capability-Aware Advantage Estimation** | Compute distinct intra-group advantage baselines for different agents based on their performance disparities under shared rollouts.|
| **Model Capabilities Discrepancy Coefficient** | Dynamically modulate the magnitude of mutual learning between agents, functioning similarly to a 'learning rate' based on their divergent capabilities.|
| **Exponential Importance Sampling** | Employ a gradient-detached exponential importance weight to encourage agents to prioritize learning from responses that exhibit minimal divergence from their own policy distributions.|
| **Stepwise Clipping** | Apply progressively tightening clip bounds to rollouts from other agents within a batch, ensuring that these external rollouts do not dominate the training process and thereby enhancing overall training stability.|

### 3. Quick Start
#### 3.1 Installation
This repo use the same environment as verl, so you can find detailed setting on https://verl.readthedocs.io/en/latest/start/install.html.
#### 3.2 Start
```bash
bash ./recipe/hacpo/run_qwen3-1.7b_qwen3-4b.sh
# key parameter
# aux_model.enable: wether to use multiple heterogenous agents
# aux_model.path: the path of other heterogenous agent
# algorithm.adv_estimator: use mapo
# actor_rollout_ref.actor.policy_loss.loss_mode: use mapo_clip
# actor_rollout_ref.actor.alpha: expontial importance sampling
# actor_rollout_ref.actor.accuracy_window_size: the size of batches used to estimate the capability of agent
# actor_rollout_ref.actor.aux_clip_ratio_low: the low bound for responses from other agents
# actor_rollout_ref.actor.aux_clip_ratio_step: the step size used in stepwise clipping
```

### 4. Detail of the repo
#### 4.1 Train two agents together
`verl/trainer/ppo/ray_trainer.py` 实现了同时训练 **Actor** 和 **Aux** 模型的功能。它通过独立的 Worker Group 管理 Aux 模型，支持为两者配置不同的 Tokenizer，并在数据流中处理 `aux_` 前缀的专用数据（如 `aux_input_ids`），从而在 Ray 框架下实现双模型的高效并行训练。

```python
# verl/trainer/ppo/ray_trainer.py

# Resource of two models
# If aux_model is enabled, spawn it as a dedicated WorkerGroup (separate process)
if self.use_aux_model and ("aux_model" in class_dict):
    main_class_dict = {k: v for k, v in class_dict.items() if k != "aux_model"}
    # ... (spawn main worker group) ...

    aux_only_dict = {"aux_model": class_dict["aux_model"]}
    worker_dict_cls_aux = create_colocated_worker_cls(class_dict=aux_only_dict)
    wg_dict_aux = self.ray_worker_group_cls(
        resource_pool=resource_pool,
        ray_cls_with_init=worker_dict_cls_aux,
        **wg_kwargs,
    )
    spawn_wg_aux = wg_dict_aux.spawn(prefix_set=aux_only_dict.keys())
    all_wg.update(spawn_wg_aux)

# Train of two models
# if auxiliary model is enabled, also generate rollouts for auxiliary model
if self.use_aux_model:
    with marked_timer("gen_aux", timing_raw, color="purple"):
        if not self.async_rollout_mode:
            aux_gen_batch_output = self.aux_model_wg.generate_sequences(aux_gen_batch)
            # mark this is from auxiliary model in the output
            gen_batch_output.batch["model_source"] = torch.zeros(
                gen_batch_output.batch.batch_size[0], dtype=torch.long
            )
            aux_gen_batch_output.batch["model_source"] = torch.ones(
                aux_gen_batch_output.batch.batch_size[0], dtype=torch.long
            )

    # ... (merging batches) ...
    batch = DataProto.concat([batch, aux_batch])

    # ... (compute old_log_prob for each model) ...
    old_log_prob_main = self.actor_rollout_wg.compute_log_prob(main_batch)
    old_log_prob_aux = self.aux_model_wg.compute_log_prob(aux_batch)

    # ... (update actor) ...
    # use_aux_model update
    print(f"Updating actor")
    '''
    all batch's input_ids, responses, response_mask, attention_mask, position_ids are from chat_template and tokenizer of actor
    '''
    aux_mask = batch.batch["model_source"] == 1
    swap(batch, mask=aux_mask)
    actor_output = self.actor_rollout_wg.update_actor(batch)

    # ... (update aux model) ...
    # reverse the model_source to get the auxiliary model data
    print(f"Updating aux model")
    '''
    all batch's input_ids, responses, response_mask, attention_mask, position_ids are from chat_template and tokenizer of aux
    '''
    swap(batch)
    batch.batch["model_source"] = 1 - batch.batch["model_source"]
    # recompute the advantage, the same method but different in group baseline, kl_penalty
    # ...
    aux_output = self.aux_model_wg.update_actor(batch)
```

#### 4.2 The Core algorithms
`verl/trainer/ppo/core_algos.py` 引入了支持双模型的损失计算逻辑（例如 `mapo_clip`）。它利用 `model_source` 标识区分模型来源，为 Actor 计算标准 PPO Loss，为 Aux 模型计算基于性能权重 (`performance`) 的加权 Loss，从而在单次更新中实现对两个模型的联合优化。

```python
# verl/trainer/ppo/core_algos.py

# advantage estimator
@register_adv_est(AdvantageEstimator.MAPO)
def compute_mapo_outcome_advantage(...):
    # ...
    # Update id2score with weighted aux model scores
    for i in range(bsz):
        if model_source[i] == 0:  # main model
            id2score[index[i]].append(scores[i])
            id2main_score[index[i]].append(scores[i])
        elif model_source[i] == 1:  # aux model
            id2score[index[i]].append(scores[i] * aux_model_performance_reciprocal)
    
    # Recompute mean and std using the weighted scores
    for idx in id2score:
        # ...
        id2mean[idx] = torch.sum(scores_tensor * weights_tensor) / torch.sum(weights_tensor)
        # ...

# loss function
@register_policy_loss("mapo_clip")
def compute_policy_loss_mapo_clip(...):
    # ...
    # Separate main and aux model data
    main_mask = model_source == 0
    aux_mask = model_source == 1
    
    # ... (compute main model loss) ...

    # Compute loss for aux model (with alpha and performance weighting)
    if aux_response_mask.numel() > 0:
        aux_advantages = advantages[aux_mask]
        aux_seq_importance_ratio = seq_importance_ratio[aux_mask]
        aux_performance_values = performance[aux_mask].clamp(min=0.1, max=10.0)
        
        aux_seq_importance_ratio_clip = torch.clamp(aux_seq_importance_ratio, min=aux_clip_ratio_low + aux_clip_ratio_step * batch_idx, max=1.0)
        # ...
        pg_losses[aux_mask] = -aux_advantages * aux_seq_importance_ratio_clip * aux_performance_values.unsqueeze(-1) *(aux_seq_importance_ratio_clip.detach() ** config.alpha)
```
