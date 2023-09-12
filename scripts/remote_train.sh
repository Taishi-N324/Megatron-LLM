#!/bin/bash

# Reload environment
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

cd /groups/gaf51217/taishi/Megatron-LLM/
source .env/bin/activate

NUM_GPU_PER_NODE=$1
NUM_NODES=$2
rank=$3
MASTER_ADDR=$4
MASTER_PORT=$5

export CUDA_DEVICE_MAX_CONNECTIONS=1



LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"


torchrun --nproc-per-node $NUM_GPU_PER_NODE \
        --nnodes $NUM_NODES \
        --node_rank $rank \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        finetune.py \
        --tensor_model_parallel_size 8 \
        --pipeline_model_parallel_size 4 \
        --load /groups/gaf51217/taishi/checkpoints/llama2-7b/tp8pp4/ \
        --save /groups/gaf51217/taishi/checkpoints/llama2-7b/tp8pp4/ \
        --tensorboard_dir tensorboard \
        --data_path /groups/gaf51217/taishi/Megatron-LLM/datasets/starcoder_text_document \
        --model_name llama2 \
        --tokenizer_type SentencePieceTokenizer \
        --vocab_file /groups/gaf51217/fujii/checkpoints/llama/tokenizer.model \
        --fp16 \
        --micro_batch_size 1 \
        --global_batch_size 1000 \
        --sequence_parallel \
        --no_bias_gelu_fusion \
        --recompute_granularity selective \
        --use_checkpoint_args \
        $LOG_ARGS $TRAIN_ARGS