# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model, need_aux_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # Default: 1 (merge all roles into a single process). If aux_model is enabled,
            # allow 2 processes so actor and aux_model can each own a vLLM sleep-mode instance.
            max_colocate = 2 if Role.AuxModel in self.mapping.values() or Role.AuxModel in self.mapping else 1
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=max_colocate, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics

def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.MAPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        model_source = None
        performance = None
        old_seq_ratio = None
        if "model_source" in data.batch:  # optional - for multi-model weighting
            model_source = data.batch["model_source"]
        if "performance" in data.batch:
            performance = data.batch["performance"]
        if "old_seq_ratio" in data.batch:
            old_seq_ratio = data.batch["old_seq_ratio"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_mapo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            model_source = model_source,
            performance = performance,
            old_seq_ratio = old_seq_ratio
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        if "model_source" in data.batch:  # optional - for multi-model weighting
            adv_kwargs["model_source"] = data.batch["model_source"]
        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data

def compute_response_mask(data: DataProto):
        """Compute the attention mask for the response part of the sequence.

        This function extracts the portion of the attention mask that corresponds to the model's response,
        which is used for masking computations that should only apply to response tokens.

        Args:
            data (DataProto): The data containing batched model outputs and inputs.

        Returns:
            torch.Tensor: The attention mask for the response tokens.
        """
        responses = data.batch["responses"]
        response_length = responses.size(1)
        attention_mask = data.batch["attention_mask"]
        return attention_mask[:, -response_length:]

# swap the input_ids, responses, response_mask, attention_mask, position_ids, prompts
def swap(batch: DataProto, mask: Optional[torch.Tensor] = None) -> DataProto:
    """
    Swap the input_ids, responses, response_mask, attention_mask, position_ids.
    
    Args:
        batch: DataProto to swap
        mask: Optional boolean mask or indices. If None, swaps entire batch.
              If provided, only swaps the specified subset.
    
    Returns:
        The same batch object (modified in-place)
    """
    # Determine indices to swap
    if mask is None:
        indices = slice(None)  # Swap entire batch
    elif isinstance(mask, torch.Tensor) and mask.dtype == torch.bool:
        indices = mask  # Boolean mask
    else:
        indices = mask  # Integer indices
    
    # Perform swap using indices
    tmp_input_ids = batch.batch["input_ids"][indices]
    tmp_responses = batch.batch["responses"][indices]
    tmp_response_mask = batch.batch["response_mask"][indices]
    tmp_attention_mask = batch.batch["attention_mask"][indices]
    tmp_position_ids = batch.batch["position_ids"][indices]
    
    batch.batch["input_ids"][indices] = batch.batch["aux_input_ids"][indices].to(
        dtype=batch.batch["input_ids"].dtype, device=batch.batch["input_ids"].device
    )
    batch.batch["responses"][indices] = batch.batch["aux_responses"][indices].to(
        dtype=batch.batch["responses"].dtype, device=batch.batch["responses"].device
    )
    batch.batch["response_mask"][indices] = batch.batch["aux_response_mask"][indices].to(
        dtype=batch.batch["response_mask"].dtype, device=batch.batch["response_mask"].device
    )
    batch.batch["attention_mask"][indices] = batch.batch["aux_attention_mask"][indices].to(
        dtype=batch.batch["attention_mask"].dtype, device=batch.batch["attention_mask"].device
    )
    batch.batch["position_ids"][indices] = batch.batch["aux_position_ids"][indices].to(
        dtype=batch.batch["position_ids"].dtype, device=batch.batch["position_ids"].device
    )
    
    batch.batch["aux_input_ids"][indices] = tmp_input_ids.to(
        dtype=batch.batch["aux_input_ids"].dtype, device=batch.batch["aux_input_ids"].device
    )
    batch.batch["aux_responses"][indices] = tmp_responses.to(
        dtype=batch.batch["aux_responses"].dtype, device=batch.batch["aux_responses"].device
    )
    batch.batch["aux_response_mask"][indices] = tmp_response_mask.to(
        dtype=batch.batch["aux_response_mask"].dtype, device=batch.batch["aux_response_mask"].device
    )
    batch.batch["aux_attention_mask"][indices] = tmp_attention_mask.to(
        dtype=batch.batch["aux_attention_mask"].dtype, device=batch.batch["aux_attention_mask"].device
    )
    batch.batch["aux_position_ids"][indices] = tmp_position_ids.to(
        dtype=batch.batch["aux_position_ids"].dtype, device=batch.batch["aux_position_ids"].device
    )
    
    return batch

def cross_encode_with_tokenizers(
        batch: DataProto,
        source_tokenizer,
        target_tokenizer,
        prompt_length: int,
        response_length: int,
        target_prefix: str = "aux_",
    ):
        """
        Cross-encode batch data from source tokenizer to target tokenizer.
        
        This function decodes responses with the source tokenizer, then re-encodes them with 
        the target tokenizer. It combines the re-encoded responses with the existing 
        aux_input_ids, aux_attention_mask, aux_position_ids (which are already tokenized 
        by target_tokenizer) to generate complete sequences.
        
        Args:
            batch (DataProto): The batch containing:
                - responses: [bs, response_length] - responses encoded by source_tokenizer
                - aux_input_ids: [bs, prompt_length] - prompts encoded by target_tokenizer
                - aux_attention_mask: [bs, prompt_length] - attention mask for prompts
                - aux_position_ids: [bs, prompt_length] - position ids for prompts
            source_tokenizer: The tokenizer used to encode the current responses
            target_tokenizer: The tokenizer to re-encode the responses
            prompt_length (int): Maximum prompt length
            response_length (int): Maximum response length
            target_prefix (str): Prefix for the new fields (e.g., "aux_")
            
        Returns:
            DataProto: The batch with additional fields:
                - {prefix}input_ids: Re-encoded full sequence (prompt + response)
                - {prefix}responses: Re-encoded response part
                - {prefix}response_mask: Mask for valid response tokens
                - {prefix}attention_mask: Attention mask for the full sequence
                - {prefix}position_ids: Position ids for the full sequence
        """
        from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length, tokenize_and_postprocess_data
        
        # Get original data
        responses = batch.batch["responses"]  # [bs, response_length] - source_tokenizer encoded
        aux_input_ids = batch.batch["aux_input_ids"]  # [bs, prompt_length] - target_tokenizer encoded
        aux_attention_mask = batch.batch["aux_attention_mask"]  # [bs, prompt_length]
        aux_position_ids = batch.batch["aux_position_ids"]  # [bs, prompt_length]
        
        batch_size = responses.shape[0]
        device = responses.device
        
        # Handle pad_token_id - use eos_token_id as fallback
        source_pad_token_id = source_tokenizer.pad_token_id if source_tokenizer.pad_token_id is not None else source_tokenizer.eos_token_id
        target_pad_token_id = target_tokenizer.pad_token_id if target_tokenizer.pad_token_id is not None else target_tokenizer.eos_token_id
        target_eos_token_id = target_tokenizer.eos_token_id
        
        # Step 1: Decode responses with source tokenizer
        response_texts = []
        for i in range(batch_size):
            response_ids = responses[i]
            # Use response_mask to determine valid length if available, otherwise filter padding tokens
            if "response_mask" in batch.batch.keys():
                valid_length = batch.batch["response_mask"][i].sum().item()
                response_ids = response_ids[:valid_length]
            else:
                # Remove padding tokens
                response_ids = response_ids[response_ids != source_pad_token_id]
            response_text = source_tokenizer.decode(response_ids, skip_special_tokens=False)
            response_texts.append(response_text)
        
        # Step 2: Re-encode responses with target tokenizer
        # tokenize_and_postprocess_data returns (input_ids, attention_mask) where input_ids shape is [1, max_length]
        target_responses_list = []
        target_response_attention_masks = []
        for response_text in response_texts:
            response_ids, response_attn_mask = tokenize_and_postprocess_data(
                prompt=response_text,
                tokenizer=target_tokenizer,
                max_length=response_length,
                pad_token_id=target_pad_token_id,
                left_pad=False,  # response uses right padding
                truncation="right"  # truncate from right if exceeds max_length
            )
            # response_ids shape: [1, response_length], squeeze to [response_length]
            target_responses_list.append(response_ids.squeeze(0))
            target_response_attention_masks.append(response_attn_mask.squeeze(0))
        
        target_responses_tensor = torch.stack(target_responses_list, dim=0).to(device)
        target_response_attention_mask = torch.stack(target_response_attention_masks, dim=0).to(device)

        # Step 4: Combine prompt and response to form complete sequences
        # Concatenate aux_input_ids (prompt) and target_responses_tensor (response)
        target_input_ids_tensor = torch.cat([aux_input_ids, target_responses_tensor], dim=-1)
        
        # Concatenate attention masks to form complete sequence
        target_attention_mask = torch.cat([aux_attention_mask, target_response_attention_mask], dim=-1)
        
        # Step 5: Generate response_mask using the same method as compute_response_mask
        # Extract the response part from the full attention_mask (last response_length positions)
        # This matches the approach in compute_response_mask: attention_mask[:, -response_length:]
        actual_response_length = target_responses_tensor.size(1)
        target_response_mask = target_attention_mask[:, -actual_response_length:]
        
        # Generate position_ids for response part
        # Similar to vllm_rollout_spmd.py: response_position_ids = position_ids[..., -1:] + delta_position_id
        delta_position_id = torch.arange(1, actual_response_length + 1, device=device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        response_position_ids = aux_position_ids[..., -1:] + delta_position_id
        target_position_ids_tensor = torch.cat([aux_position_ids, response_position_ids], dim=-1)

        batch.batch[f"{target_prefix}input_ids"] = target_input_ids_tensor
        batch.batch[f"{target_prefix}responses"] = target_responses_tensor
        batch.batch[f"{target_prefix}response_mask"] = target_response_mask
        batch.batch[f"{target_prefix}attention_mask"] = target_attention_mask
        batch.batch[f"{target_prefix}position_ids"] = target_position_ids_tensor
        return batch

class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
        aux_tokenizer=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
            aux_tokenizer: Tokenizer for auxiliary model. If None, uses same tokenizer as actor.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer        
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_aux_model = need_aux_model(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.actor_tokenizer = tokenizer
        self.aux_tokenizer = aux_tokenizer if aux_tokenizer is not None and self.use_aux_model else tokenizer
        
        # Determine if tokenizers are different by comparing vocab sizes and identities
        if not self.use_aux_model:
            self.use_different_tokenizers = False
        elif aux_tokenizer is None:
            self.use_different_tokenizers = False
        elif aux_tokenizer is tokenizer:
            self.use_different_tokenizers = False
        else:
            self.use_different_tokenizers = True
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        
        Note: Tokenizer conversion is now handled in _get_gen_batch by saving decoded prompt texts,
        so we don't need to recreate datasets with return_raw_chat=True.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor,
                aux_tokenizer=self.aux_tokenizer if self.use_different_tokenizers else None
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor,
                aux_tokenizer=self.aux_tokenizer if self.use_different_tokenizers else None
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    def _get_aux_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # rename the keys to input_ids, attention_mask, position_ids
        input_ids = batch.batch.pop("input_ids")
        attention_mask = batch.batch.pop("attention_mask")
        position_ids = batch.batch.pop("position_ids")
        raw_prompt_ids = batch.non_tensor_batch.pop("raw_prompt_ids")

        batch.batch["input_ids"] = batch.batch["aux_input_ids"]
        batch.batch["attention_mask"] = batch.batch["aux_attention_mask"]
        batch.batch["position_ids"] = batch.batch["aux_position_ids"]
        batch.non_tensor_batch["raw_prompt_ids"] = batch.non_tensor_batch["aux_raw_prompt_ids"]

        batch.batch["aux_input_ids"] = input_ids
        batch.batch["aux_attention_mask"] = attention_mask
        batch.batch["aux_position_ids"] = position_ids
        batch.non_tensor_batch["aux_raw_prompt_ids"] = raw_prompt_ids

        batch.meta_info["aux_gen"] = True
        batch.meta_info["aux_eos_token_id"] = self.aux_tokenizer.eos_token_id
        batch.meta_info["aux_pad_token_id"] = self.aux_tokenizer.pad_token_id
        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        aux_gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )


        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            aux_gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return aux_gen_batch

    def _validate(self, worker: str = "actor"):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []

        target_wg = self.actor_rollout_wg if worker == "actor" else self.aux_model_wg
        target_tokenizer = self.actor_tokenizer if worker == "actor" else self.aux_tokenizer
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )
            # breakpoint()
            if worker == "aux":
                # swap the test_batch's input_ids, attention_mask, position_id
                tmp_input_ids = test_batch.batch["input_ids"]
                tmp_attention_mask = test_batch.batch["attention_mask"]
                tmp_position_ids = test_batch.batch["position_ids"]
                tmp_raw_prompt_ids = test_batch.non_tensor_batch["raw_prompt_ids"]
                test_batch.batch["input_ids"] = test_batch.batch["aux_input_ids"]
                test_batch.batch["attention_mask"] = test_batch.batch["aux_attention_mask"]
                test_batch.batch["position_ids"] = test_batch.batch["aux_position_ids"]
                test_batch.non_tensor_batch["raw_prompt_ids"] = test_batch.non_tensor_batch["aux_raw_prompt_ids"]
                test_batch.batch["aux_input_ids"] = tmp_input_ids
                test_batch.batch["aux_attention_mask"] = tmp_attention_mask
                test_batch.batch["aux_position_ids"] = tmp_position_ids
                test_batch.non_tensor_batch["aux_raw_prompt_ids"] = tmp_raw_prompt_ids
                test_batch.meta_info["aux_gen"] = True
                test_batch.meta_info["aux_eos_token_id"] = target_tokenizer.eos_token_id
                test_batch.meta_info["aux_pad_token_id"] = target_tokenizer.pad_token_id
            # breakpoint()
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [target_tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)
            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": target_tokenizer.eos_token_id,
                "pad_token_id": target_tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                target_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = target_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")
            # import pdb; pdb.set_trace()
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [target_tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            if worker == "aux":
                # TODO: /root/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py generate "prompts", which is actually the input_ids(aux's tokenizer and chat template), causing that actor's tokenizer fails to show the prompt and chat tempalte. 
                # TODO: we can fix the bug, but it's actually not a problem because "prompts" is not a key used in train or valid progress. It is only to show.
                test_batch = cross_encode_with_tokenizers(
                    batch=test_batch,
                    source_tokenizer=self.aux_tokenizer,
                    target_tokenizer=self.actor_tokenizer,
                    prompt_length=self.config.data.max_prompt_length,
                    response_length=self.config.data.max_response_length,
                    target_prefix="aux_"
                )
                if "response_mask" not in test_batch.batch.keys():
                    test_batch.batch["response_mask"] = compute_response_mask(test_batch)
                test_batch = swap(test_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_ainfo" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()
        
        # add prefix to aux metrics to avoid key conflict with actor
        if worker == "aux":
            metric_dict = {f"val_aux/{k}": v for k, v in metric_dict.items()}
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # when ref and aux_model are enabled, create aux_ref to compute the ref_log_prob
        if self.use_reference_policy and self.use_aux_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            # copy the actor_rollout_ref config and change the model.path to aux_model
            aux_ref_cfg = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
            # override the model.path (keep other configs the same)
            aux_ref_cfg.model.path = self.config.aux_model.model.path

            aux_ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=aux_ref_cfg,
                role="ref",
            )
            # use a different key to avoid overwriting the original ref
            self.resource_pool_to_cls[resource_pool]["aux_ref"] = aux_ref_policy_cls

        # Initialize auxiliary model worker group if needed
        if self.use_aux_model:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.AuxModel)
            
            # Create aux_model config by inheriting from actor_rollout_ref
            # This ensures aux_model gets all the same settings:
            # - VLLM configuration (free_cache_engine=False)
            # - GPU memory utilization
            # - Sampling parameters
            # - All other rollout settings
            aux_config = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
            aux_config.model.path = self.config.aux_model.model.path            
            print(f"   Model path: {aux_config.model.path}")
            
            aux_model_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.AuxModel],
                config=aux_config,
                role="aux_model"
            )
            self.resource_pool_to_cls[resource_pool]["aux_model"] = aux_model_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            # If aux_model is enabled, spawn it as a dedicated WorkerGroup (separate process)
            # within the same resource pool to satisfy vLLM's "one sleep-mode instance per process" rule.
            if self.use_aux_model and ("aux_model" in class_dict):
                main_class_dict = {k: v for k, v in class_dict.items() if k != "aux_model"}
                if len(main_class_dict) > 0:
                    worker_dict_cls_main = create_colocated_worker_cls(class_dict=main_class_dict)
                    wg_dict_main = self.ray_worker_group_cls(
                        resource_pool=resource_pool,
                        ray_cls_with_init=worker_dict_cls_main,
                        **wg_kwargs,
                    )
                    spawn_wg_main = wg_dict_main.spawn(prefix_set=main_class_dict.keys())
                    all_wg.update(spawn_wg_main)

                aux_only_dict = {"aux_model": class_dict["aux_model"]}
                worker_dict_cls_aux = create_colocated_worker_cls(class_dict=aux_only_dict)
                wg_dict_aux = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls_aux,
                    **wg_kwargs,
                )
                spawn_wg_aux = wg_dict_aux.spawn(prefix_set=aux_only_dict.keys())
                all_wg.update(spawn_wg_aux)
            else:
                worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
                wg_dict = self.ray_worker_group_cls(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
                all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        # when ref and aux_model are enabled, create aux_ref to compute the ref_log_prob
        if self.use_reference_policy and self.use_aux_model and not self.ref_in_actor:
            self.aux_ref_policy_wg = all_wg["aux_ref"]
            self.aux_ref_policy_wg.init_model()
        
        # Initialize auxiliary model worker group if needed
        if self.use_aux_model:
            self.aux_model_wg = all_wg["aux_model"]
            self.aux_model_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_aux_ckpt_to_keep = (
            self.config.trainer.get("max_aux_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        if self.use_aux_model:
            aux_local_path = os.path.join(local_global_step_folder, "aux_model")
            aux_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "aux_model")
            )
            self.aux_model_wg.save_checkpoint(
                aux_local_path, aux_remote_path, self.global_steps, max_ckpt_to_keep=max_aux_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        aux_path = os.path.join(global_step_folder, "aux_model")
        
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        
        # load aux model
        if self.use_aux_model:
            self.aux_model_wg.load_checkpoint(
                aux_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            # import pdb; pdb.set_trace()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.use_aux_model:
                aux_val_metrics = self._validate(worker="aux")
                assert aux_val_metrics, f"{aux_val_metrics=}"
                pprint(f"Initial auxiliary validation metrics: {aux_val_metrics}")
                logger.log(data=aux_val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        aux_model_performance = 0
        main_model_performance = 0

        
        # Initialize accuracy history for sliding window average
        self.main_model_accuracy_history = []
        self.aux_model_accuracy_history = []
        self.accuracy_window_size = self.config.actor_rollout_ref.actor.accuracy_window_size
        self.stable_perf = self.config.actor_rollout_ref.actor.stable_perf

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                """
                gen_batch: 
                input_ids, attention_mask, position_ids, raw_prompts (prompts with chat_template and tokenizer of actor)
                aux_input_ids, aux_attention_mask, aux_position_ids, aux_raw_prompts (prompts with chat_template and tokenizer of aux)

                aux_batch: 
                input_ids, attention_mask, position_ids, raw_prompts (prompts with chat_template and tokenizer of aux)
                aux_input_ids, aux_attention_mask, aux_position_ids, aux_raw_prompts (prompts with chat_template and tokenizer of actor)
                """
                aux_batch = None
                aux_gen_batch = None
                if self.use_aux_model:
                    aux_batch: DataProto = DataProto.from_single_dict(batch_dict)
                    aux_batch.non_tensor_batch["uid"] = batch.non_tensor_batch["uid"]
                    aux_gen_batch = self._get_aux_gen_batch(aux_batch)
                    aux_batch.meta_info.pop("aux_gen", None)
                    aux_batch.meta_info.pop("aux_eos_token_id", None)
                    aux_batch.meta_info.pop("aux_pad_token_id", None)
                    aux_gen_batch.meta_info["global_steps"] = self.global_steps
                    aux_gen_batch = aux_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            # actor model rollout, pi_theta_old
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                    
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
                            else:
                                ValueError("Async rollout mode is not supported for auxiliary model")
                            timing_raw.update(aux_gen_batch_output.meta_info["timing"])
                            aux_gen_batch_output.meta_info.pop("timing", None)
                            print(f"Generated {aux_gen_batch_output.batch.batch_size[0]} sequences from auxiliary model")

                    # Remax only
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    # if auxiliary model is enabled, add model_source to all data
                    if self.use_aux_model:
                        aux_batch = aux_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # merge main model rollouts to batch
                        batch = batch.union(gen_batch_output)
                        # merge main model rollouts to batch
                        aux_batch = aux_batch.union(aux_gen_batch_output)
                        """
                        batch:
                        input_ids, responses, response_mask, attention_mask, position_ids(prompt+response with chat_template and tokenizer of actor)
                        aux_input_ids, aux_responses, aux_response_mask, aux_attention_mask, aux_position_ids(prompt+response with chat_template and tokenizer of aux)

                        aux_batch:
                        input_ids, responses, response_mask, attention_mask, position_ids(prompt+response with chat_template and tokenizer of aux)
                        aux_input_ids, aux_responses, aux_response_mask, aux_attention_mask, aux_position_ids(prompt+response with chat_template and tokenizer of actor)
                        """
                        # Cross-encode batch and aux_batch with different tokenizers
                        # For batch: decode with actor tokenizer, re-encode with aux tokenizer
                        batch = cross_encode_with_tokenizers(
                            batch=batch,
                            source_tokenizer=self.actor_tokenizer,
                            target_tokenizer=self.aux_tokenizer,
                            prompt_length=self.config.data.max_prompt_length,
                            response_length=self.config.data.max_response_length,
                            target_prefix="aux_"
                        )
                        # For aux_batch: decode with aux tokenizer, re-encode with actor tokenizer
                        # Note: we use empty prefix "" to directly replace the main fields
                        # or use a different prefix like "actor_" if you want to keep both
                        aux_batch = cross_encode_with_tokenizers(
                            batch=aux_batch,
                            source_tokenizer=self.aux_tokenizer,
                            target_tokenizer=self.actor_tokenizer,
                            prompt_length=self.config.data.max_prompt_length,
                            response_length=self.config.data.max_response_length,
                            target_prefix="aux_"
                        )
                        
                        print(f"After cross-encoding, batch.batch keys: {batch.batch.keys()}")
                        print(f"After cross-encoding, aux_batch.batch keys: {aux_batch.batch.keys()}")

                        batch = DataProto.concat([batch, aux_batch])
                        
                        print(f"Combined batch size after adding auxiliary model data: {batch.batch.batch_size[0]}")
                    else:
                        # standard single model processing
                        batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                        batch.batch["old_log_prob_mask"] = batch.batch["response_mask"]
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        print(f"Balancing batch")
                        self._balance_batch(batch, metrics=metrics)
                    else:
                        print(f"Not balancing batch")
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            if self.use_aux_model:
                                '''
                                first swap: all batch's input_ids, responses, response_mask, attention_mask, position_ids are from chat_template and tokenizer of actor
                                second swap: restoring the initial state
                                '''
                                main_mask = batch.batch["model_source"] == 0
                                aux_mask = batch.batch["model_source"] == 1
                                # aux_batch_dict = batch[aux_mask].batch
                                # print(f"input_ids:{aux_batch_dict[0]['input_ids']}")
                                # breakpoint()
                                swap(batch, mask=aux_mask)
                                # aux_batch_dict = batch[aux_mask].batch
                                # print(f"input_ids:{aux_batch_dict[0]['input_ids']}")
                                # breakpoint()
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                                swap(batch, mask=aux_mask)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        if self.use_aux_model:
                            # separate main model and auxiliary model data
                            main_mask = batch.batch["model_source"] == 0
                            aux_mask = batch.batch["model_source"] == 1
                            
                            # compute old_log_prob for each model
                            main_batch = batch.select_idxs(main_mask)
                            aux_batch = batch.select_idxs(aux_mask)
                            
                            old_log_prob_main = self.actor_rollout_wg.compute_log_prob(main_batch)
                            old_log_prob_aux = self.aux_model_wg.compute_log_prob(aux_batch)
                            
                            old_log_prob = DataProto.from_dict(
                                tensors={
                                    "old_log_probs": torch.zeros_like(batch.batch["responses"], dtype=torch.float, device=old_log_prob_main.batch["old_log_probs"].device),
                                    "entropys": torch.zeros_like(batch.batch["responses"], dtype=torch.float, device=old_log_prob_main.batch["entropys"].device)
                                },
                                meta_info={}
                            )
                            
                            # temperature in all models are same
                            if "temperature" in old_log_prob_main.meta_info:
                                old_log_prob.meta_info["temperature"] = old_log_prob_main.meta_info["temperature"]
                            elif "temperature" in old_log_prob_aux.meta_info:
                                old_log_prob.meta_info["temperature"] = old_log_prob_aux.meta_info["temperature"]
                            else:
                                old_log_prob.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                            
                            # fill in the results of each model
                            old_log_prob.batch["old_log_probs"][main_mask] = old_log_prob_main.batch["old_log_probs"]
                            old_log_prob.batch["entropys"][main_mask] = old_log_prob_main.batch["entropys"]
                            old_log_prob.batch["old_log_probs"][aux_mask] = old_log_prob_aux.batch["old_log_probs"]
                            old_log_prob.batch["entropys"][aux_mask] = old_log_prob_aux.batch["entropys"]
                            
                            # compute entropy for actor
                            main_entropys = old_log_prob_main.batch["entropys"]
                            main_response_masks = batch.batch["response_mask"][main_mask]
                            main_loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            main_entropy_agg = agg_loss(loss_mat=main_entropys, loss_mask=main_response_masks, loss_agg_mode=main_loss_agg_mode)
                            main_old_log_prob_metrics = {"actor/entropy": main_entropy_agg.detach().item()}
                            metrics.update(main_old_log_prob_metrics)
                            # compute entropy for auxiliary model
                            aux_entropys = old_log_prob_aux.batch["entropys"]
                            aux_response_masks = batch.batch["response_mask"][aux_mask]
                            aux_loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            aux_entropy_agg = agg_loss(loss_mat=aux_entropys, loss_mask=aux_response_masks, loss_agg_mode=aux_loss_agg_mode)
                            aux_old_log_prob_metrics = {"aux/entropy": aux_entropy_agg.detach().item()}
                            metrics.update(aux_old_log_prob_metrics)

                            # main/aux_old_seq_ratio have no meaning!
                            main_old_seq_ratio = torch.ones_like(batch.batch["model_source"], dtype=torch.float)
                            aux_old_seq_ratio = torch.ones_like(batch.batch["model_source"], dtype=torch.float)
                            batch.batch["old_seq_ratio"] = main_old_seq_ratio
                        else:
                            # standard single model processing
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            # old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        # If aux_model is enabled, compute ref_log_prob separately for each model's samples
                        with marked_timer("ref", timing_raw, color="olive"):
                            if self.use_aux_model:
                                # Separate main and aux samples
                                main_mask = batch.batch["model_source"] == 0
                                aux_mask = batch.batch["model_source"] == 1
                                main_batch = batch.select_idxs(main_mask)
                                aux_batch = batch.select_idxs(aux_mask)
                                
                                # Compute ref_log_prob for main model samples using main ref
                                if not self.ref_in_actor:
                                    main_ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(main_batch)
                                else:
                                    main_ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(main_batch)
                                
                                # Compute ref_log_prob for aux model samples using aux ref
                                if not self.ref_in_actor:
                                    aux_ref_log_prob = self.aux_ref_policy_wg.compute_ref_log_prob(aux_batch)
                                else:
                                    aux_ref_log_prob = self.aux_model_wg.compute_ref_log_prob(aux_batch)
                                
                                # Combine the results by filling in the correct positions
                                # Create a combined result with the same shape as batch, on the same device as the returned values
                                ref_log_prob = DataProto.from_dict(
                                    tensors={
                                        "ref_log_prob": torch.zeros_like(
                                            batch.batch["responses"], 
                                            dtype=torch.float,
                                            device=main_ref_log_prob.batch["ref_log_prob"].device
                                        )
                                    }
                                )
                                
                                # Fill in the results of each model at the correct positions
                                ref_log_prob.batch["ref_log_prob"][main_mask] = main_ref_log_prob.batch["ref_log_prob"]
                                ref_log_prob.batch["ref_log_prob"][aux_mask] = aux_ref_log_prob.batch["ref_log_prob"]
                                
                                batch = batch.union(ref_log_prob)
                            else:
                                # Standard single model processing
                                if not self.ref_in_actor:
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                else:
                                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        # update accuracy of main_model and aux_model
                        if self.use_aux_model:
                            main_mask = batch.batch["model_source"] == 0
                            aux_mask = batch.batch["model_source"] == 1
                            scores_per_seq = batch.batch["token_level_scores"].detach().sum(-1)  # (bs,)
                            # compute the accuracy of main_model and aux_model, sum of rewards > 0.0 divided by number of problems
                            all_main_model_accuracy = torch.sum(scores_per_seq[main_mask][scores_per_seq[main_mask] > 0.0]) / scores_per_seq[main_mask].numel()
                            all_aux_model_accuracy  = torch.sum(scores_per_seq[aux_mask][scores_per_seq[aux_mask] > 0.0]) / scores_per_seq[aux_mask].numel()
                            # 每个题目回进行32次回答，同一题目的所有回答会共用一个 uid（与 MAPO 的 index 一致）
                            # 只统计每个 prompt 下前 8 条 response；遍历顺序与 compute_mapo_outcome_advantage 相同：for i in range(bsz)
                            index = batch.non_tensor_batch["uid"]
                            model_source = batch.batch["model_source"]
                            bsz = scores_per_seq.shape[0]
                            k_first = 8
                            main_keep = []
                            aux_keep = []
                            main_cnt = {}
                            aux_cnt = {}
                            for i in range(bsz):
                                idx = index[i]
                                if model_source[i] == 0:
                                    c = main_cnt.get(idx, 0)
                                    if c < k_first:
                                        main_keep.append(i)
                                    main_cnt[idx] = c + 1
                                elif model_source[i] == 1:
                                    c = aux_cnt.get(idx, 0)
                                    if c < k_first:
                                        aux_keep.append(i)
                                    aux_cnt[idx] = c + 1

                            main_idx = torch.tensor(main_keep, device=scores_per_seq.device, dtype=torch.long)
                            aux_idx = torch.tensor(aux_keep, device=scores_per_seq.device, dtype=torch.long)
                            main_model_accuracy = (scores_per_seq[main_idx] > 0.0).float().mean()
                            aux_model_accuracy = (scores_per_seq[aux_idx] > 0.0).float().mean()
                            print(f"Main model accuracy: {main_model_accuracy}, Aux model accuracy: {aux_model_accuracy}")
                            
                            # Store accuracy values in history
                            self.main_model_accuracy_history.append(main_model_accuracy.item())
                            self.aux_model_accuracy_history.append(aux_model_accuracy.item())
                            
                            # Keep only the last k accuracy values
                            if len(self.main_model_accuracy_history) > self.accuracy_window_size:
                                self.main_model_accuracy_history.pop(0)
                            if len(self.aux_model_accuracy_history) > self.accuracy_window_size:
                                self.aux_model_accuracy_history.pop(0)
                            
                            # Calculate performance as average of past k accuracy values
                            main_model_performance = sum(self.main_model_accuracy_history) / len(self.main_model_accuracy_history)
                            aux_model_performance = sum(self.aux_model_accuracy_history) / len(self.aux_model_accuracy_history)
                            
                            # Ensure performance values are not too small
                            main_model_performance = 1e-8 if main_model_performance < 1e-8 else main_model_performance
                            aux_model_performance = 1e-8 if aux_model_performance < 1e-8 else aux_model_performance    
                            # print(f"Main model performance: {main_model_performance}, Aux model performance: {aux_model_performance}")

                            # # compute the ratio of main_model_performance and aux_model_performance
                            performance_ratio = aux_model_performance / main_model_performance
                            if performance_ratio > 10.0:
                                performance_ratio = 10.0
                            elif performance_ratio < 0.1:
                                performance_ratio = 0.1

                            ## record the true ratio and the performance ratio in "performance.txt", and bias=| true_ratio - performance_ratio | / true_ratio
                            with open("new_performance.txt", "a") as f:
                                true_ratio = all_aux_model_accuracy.item()/(all_main_model_accuracy.item()+1e-8)
                                f.write(f"Global steps: {self.global_steps}, True ratio: {true_ratio}, Performance ratio: {performance_ratio}, bias: {abs(true_ratio - performance_ratio) / (true_ratio+1e-8)}\n")

                            if self.global_steps < self.stable_perf:
                                print(f"Global steps < stable perf, set performance ratio to 1.0")
                                performance_ratio = 1.0
                            performance = DataProto.from_dict(
                                tensors={
                                    "performance": torch.full_like(
                                        batch.batch["model_source"], 
                                        performance_ratio, 
                                        dtype=torch.float
                                    )
                                },
                                meta_info={}
                            )
                            performance_metrics = {"performance": performance_ratio}
                            metrics.update(performance_metrics)
                            batch = batch.union(performance)

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor - in multi-model training, all data are involved in training
                        with marked_timer("update_actor", timing_raw, color="red"):
                            if self.use_aux_model:
                                # use_aux_model update
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                batch.meta_info["metric_prefix"] = "actor"
                                print(f"Updating actor")
                                print(f"actor: Performance: {batch.batch['performance'][0].item()}")
                                '''
                                all batch's input_ids, responses, response_mask, attention_mask, position_ids are from chat_template and tokenizer of actor
                                '''
                                aux_mask = batch.batch["model_source"] == 1
                                swap(batch, mask=aux_mask)
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                                
                                # Record actor's data metrics right after actor update
                                # Split metrics into two parts:
                                # 1. Critic metrics (advantages, returns, etc.): from all samples
                                # 2. Length metrics (response_length, prompt_length): from actor samples only
                                
                                # Get critic metrics from all samples
                                all_samples_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                                critic_keys = [k for k in all_samples_metrics.keys() if k.startswith("critic/")]
                                actor_critic_metrics = {k: all_samples_metrics[k] for k in critic_keys}
                                
                                # Get length metrics from actor samples only
                                main_mask_for_actor = batch.batch["model_source"] == 0
                                actor_data_batch = batch.select_idxs(main_mask_for_actor)
                                actor_samples_metrics = compute_data_metrics(batch=actor_data_batch, use_critic=self.use_critic)
                                length_keys = [k for k in actor_samples_metrics.keys() if not k.startswith("critic/")]
                                actor_length_metrics = {k: actor_samples_metrics[k] for k in length_keys}
                                
                                # Combine and add prefix
                                actor_data_metrics = {**actor_critic_metrics, **actor_length_metrics}
                                actor_data_metrics = {f"actor/{k}" : v for k, v in actor_data_metrics.items()}
                                metrics.update(actor_data_metrics)

                                # reverse the model_source to get the auxiliary model data
                                print(f"Updating aux model")
                                '''
                                all batch's input_ids, responses, response_mask, attention_mask, position_ids are from chat_template and tokenizer of aux
                                '''
                                swap(batch)
                                batch.meta_info["metric_prefix"] = "aux"
                                batch.batch["model_source"] = 1 - batch.batch["model_source"]
                                # Note: ref_log_prob has already been computed for both main and aux samples above
                                # recompute the advantage, the same method but different in group baseline, kl_penalty
                                with marked_timer("adv_aux", timing_raw, color="brown"):
                                    # apply_kl_penalty if available
                                    if self.config.algorithm.use_kl_in_reward:
                                        batch, kl_metrics = apply_kl_penalty(
                                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                        )
                                        kl_metrics = {f"aux/{k}": v for k, v in kl_metrics.items()}
                                        metrics.update(kl_metrics)
                                    # recompute the performance
                                    batch.batch["performance"] = torch.reciprocal(batch.batch["performance"])
                                    print(f"aux: Performance: {batch.batch['performance'][0].item()}")
                                    # recompute the main_old_log_probs
                                    batch.batch["old_seq_ratio"] = aux_old_seq_ratio 
                                    # recompute the advantage
                                    if self.config.algorithm.adv_estimator == AdvantageEstimator.MAPO:
                                        batch = compute_advantage(
                                            batch,
                                            adv_estimator=self.config.algorithm.adv_estimator,
                                            gamma=self.config.algorithm.gamma,
                                            lam=self.config.algorithm.lam,
                                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                            config=self.config.algorithm,
                                        )
                                # modify the batch sequence position: the first half and the second half are swapped
                                batch_size = batch.batch.batch_size[0]
                                # create new indices: the second half + the first half
                                new_indices = torch.cat([
                                    torch.arange(batch_size // 2, batch_size),  # the second half
                                    torch.arange(0, batch_size // 2)           # the first half
                                ])
                                # use DataProto's reorder method to directly adjust the position
                                batch.reorder(new_indices)   
                                aux_output = self.aux_model_wg.update_actor(batch)
                                
                                # Record aux's data metrics right after aux update
                                # Split metrics into two parts:
                                # 1. Critic metrics (advantages, returns, etc.): from all samples
                                # 2. Length metrics (response_length, prompt_length): from aux samples only
                                
                                # Get critic metrics from all samples
                                all_samples_metrics_aux = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                                critic_keys_aux = [k for k in all_samples_metrics_aux.keys() if k.startswith("critic/")]
                                aux_critic_metrics = {k: all_samples_metrics_aux[k] for k in critic_keys_aux}
                                
                                # Get length metrics from aux samples only
                                # At this point, model_source has been reversed, so model_source=0 is aux model
                                aux_data_batch = batch.select_idxs(aux_mask)
                                aux_samples_metrics = compute_data_metrics(batch=aux_data_batch, use_critic=self.use_critic)
                                length_keys_aux = [k for k in aux_samples_metrics.keys() if not k.startswith("critic/")]
                                aux_length_metrics = {k: aux_samples_metrics[k] for k in length_keys_aux}                             
                                # Combine and add prefix
                                aux_data_metrics = {**aux_critic_metrics, **aux_length_metrics}
                                # add prefix with aux for all
                                aux_data_metrics = {f"aux/{k}" : v for k, v in aux_data_metrics.items()}
                                metrics.update(aux_data_metrics)

                                # pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, main_ppo_kl_scalar, aux_ppo_kl_scalar
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                                aux_output_metrics = reduce_metrics(aux_output.meta_info["metrics"])
                                metrics.update(aux_output_metrics)

                            else:
                                # standard single model update
                                batch.meta_info["metric_prefix"] = "actor"
                                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)
                    if self.use_aux_model:
                        with marked_timer("aux_testing", timing_raw, color="blue"): 
                            aux_val_metrics: dict = self._validate(worker="aux")
                        metrics.update(aux_val_metrics)
                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics - for aux_model, metrics are already recorded after each update
                # for single model, record metrics here
                if not self.use_aux_model:
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
