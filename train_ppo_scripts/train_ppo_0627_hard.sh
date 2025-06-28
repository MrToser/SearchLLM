# export HIP_VISIBLE_DEVICES=2,3,4,5
export HIP_VISIBLE_DEVICES=0,1,6,7
export DATA_DIR='data/ids_augument_filter_2_divide_hard'
export WANDB_API_KEY="c898593d367726b4fbe3d3468b734a49870a348d"
WAND_PROJECT='Search-R1'
# ids_augument_divide
# /home/avnet/mount_disk/sjh/SearchLLM/verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-it-em-amd-0626/actor/global_step_200

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em-amd-0628-hard
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em
export ACTOR_MODEL="/home/avnet/mount_disk/sjh/SearchLLM/verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-it-em-amd-0628-medium/actor/final"
# set -x
export CRITIC_MODEL="/home/avnet/mount_disk/sjh/SearchLLM/verl_checkpoints/nq-search-r1-ppo-qwen2.5-3b-it-em-amd-0628-medium/critic/final"
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    \
    searchllm.api_model="glm-4-air-250414" \
    searchllm.mode="base" \
    \
    data.train_files=$DATA_DIR/train_hard.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    +trainer.val_only=False \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=4 \
    trainer.total_training_steps=null \
    \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.gamma=1 \
    algorithm.lam=0.999 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    +actor_rollout_ref.actor.entropy_coeff_mode="base" \
    +actor_rollout_ref.actor.raw_entropy_coeff=0.001 \
    \
    do_search=True \
    actor_rollout_ref.actor.state_masking=True \
    \
    data.train_batch_size=64 \
    data.val_batch_size=1024 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    critic.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size=8 \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    +actor_rollout_ref.actor.optim.lr_warmup_direction="down_constant" \
    critic.optim.lr=1e-5 \
    critic.optim.lr_warmup_steps_ratio=0 \
    +critic.optim.lr_warmup_direction="up" \
    trainer.critic_warmup=0 \
    \
    actor_rollout_ref.model.path=$ACTOR_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    critic.model.use_remove_padding=True \
    \
    actor_rollout_ref.rollout.n_agent=1 \
    max_turns=4 \
    \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.no_think_rl=False \
    data.train_data_num=null \
    data.val_data_num=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1 \
    critic.model.path=$CRITIC_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    trainer.logger=['wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/home/avnet/mount_disk/sjh/SearchLLM/verl_checkpoints/$EXPERIMENT_NAME \
    retriever.url="http://127.0.0.1:8002/retrieve" \
    retriever.topk=3 \
    2>&1 | tee "/home/avnet/xiongjing/sjh/agent/Search-R1/logs/${EXPERIMENT_NAME}.log"