source "/home/l00618052/AscendSpeed/tests_extend/system_tests/env_npu.sh"

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
MASTER_ADDR=90.90.94.184
MASTER_PORT=6900
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


# DISTRIBUTED_ARGS="\
#     --nnodes 1 \
#     --nproc_per_node 8 \
#     --master-port 61883 \
# "

TP=1
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
    #         --num-layers-per-virtual-pipeline-stage 1 \

GPT_ARGS="
    --netmind-path /home/l00618052/netmind3.8/output/install/lib \
    --netmind-mode aaa \
    --netmind-weight tp_1,sp_2,ep_3,cp_4,dp_5,pp_6 \
    --netmind-comm-amount tp_1,sp_2,ep_3,cp_4,dp_5,pp_6 \
    --netmind-qos tp_1,sp_2,ep_3,cp_4,dp_5,pp_6 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \

    --micro-batch-size 4 \
    --global-batch-size 4 \
    --num-layers 12 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --seq-length 1024\
    --max-position-embeddings 1024 \
    --make-vocab-size-divisible-by 16032 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --tokenizer-type NullTokenizer \
    --vocab-size 0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --swiglu \
    --no-masked-softmax-fusion \
    --lr 2e-5 \
    --min-lr 2e-5 \
    --train-iters 2500 \
    --weight-decay 0 \
    --weight-decay 0.0 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16
"


DATA_ARGS="
    --data-path /home/l00618052/ModelZoo-PyTorch/PyTorch/built-in/mlm/OpenSora1.0/MSRVTT-train/annotations.csv \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 20 \
    --eval-interval 10000 \
    --eval-iters 10 \
"

CKPT_SAVE_DIR="/home/l00618052/Megatron-LM/ckpt"
CKPT_LOAD_DIR="/home/l00618052/Megatron-LM/ckpt/iter_0000005"

torchrun $DISTRIBUTED_ARGS pretrain_opensora.py /home/l00618052/Megatron-LM/16x256x256.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save ${CKPT_SAVE_DIR} \
    --load ${CKPT_LOAD_DIR}


# torchrun $DISTRIBUTED_ARGS scripts/train.py configs/opensora/train/16x256x256.py $DATA_ARGS

# torchrun --nnodes=1 \
#     --nproc_per_node=8 \
#     --master-port 61883 \
#     scripts/train.py configs/opensora/train/16x256x256.py \
#     --data-path /home2/l00618052/ModelZoo-PyTorch/PyTorch/built-in/mlm/OpenSora1.0/MSRVTT-train/annotations.csv
set +x
