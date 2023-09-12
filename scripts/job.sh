#!/bin/bash
#$ -l rt_F=8
#$ -l h_rt=0:30:00
#$ -l USE_SSH=1
#$ -v SSH_PORT=2200
#$ -j y
#$ -o outputs/
#$ -cwd

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

cd /groups/gaf51217/taishi/Megatron-LLM/
source .env/bin/activate
# distributed settings
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep 'inet ' | awk '{ print $2 }' | cut -d "/" -f 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "MASTER_ADDR=${MASTER_ADDR}"

# hostfile
NUM_NODES=$NHOSTS
NUM_GPU_PER_NODE=4
ssh_port=2200
# NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

LOG_ARGS="--log_interval 1 --save_interval 100 --eval_interval 50"
TRAIN_ARGS="--train_iters 500 --lr_decay_style cosine --lr_warmup_iters 50 --lr 3e-4 --min_lr 1e-6"




function run_slave_node() {
    local rank=1

    cat ${SGE_JOB_HOSTLIST} | grep -v $HOSTNAME | while read node; do
		ssh -p $ssh_port $node \
        	"bash /groups/gaf51217/taishi/Megatron-LLM/scripts/remote_train.sh $NUM_GPU_PER_NODE $NUM_NODES $rank $MASTER_ADDR $MASTER_PORT" &
        rank=$((rank+1))
    done
}

function run_master_node() {
    local rank=0
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

}



function main() {
    run_slave_node
    run_master_node

    wait
}

main