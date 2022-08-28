#!/bin/bash

train() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/train.py ${EXTRA_ARGS}
}

train_clip() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/train_clip.py ${EXTRA_ARGS}
}

swa_finetune() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/swa_finetune.py ${EXTRA_ARGS}
}

test_clip_retrieval() {
    if [[ "${RANK}" == 0 ]]; then
        python3 tools/test_clip_retrieval.py ${EXTRA_ARGS}
    fi
}

test_clip_zeroshot() {
    if [[ "${RANK}" == 0 ]]; then
        python3 tools/test_clip_zeroshot.py ${EXTRA_ARGS}
    fi
}

test() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/test.py ${EXTRA_ARGS}
}

test_all() {
    echo "======== Test on Val and ReaL ========"
    python3 tools/test.py \
        ${EXTRA_ARGS} \
        VERBOSE False \
        TEST.REAL_LABELS True
    echo "======== Test on imagenet-a ========"
    python3 tools/test.py \
        ${EXTRA_ARGS} \
        VERBOSE False \
        DATASET.DATA_FORMAT jpg \
        DATASET.ROOT DATASET/imagenet \
        DATASET.TEST_SET imagenet-a \
        TEST.VALID_LABELS DATASET/imagenet/imagenet_a_indices.txt
    echo "======== Test on imagenet-r ========"
    python3 tools/test.py \
        ${EXTRA_ARGS} \
        VERBOSE False \
        DATASET.DATA_FORMAT jpg \
        DATASET.ROOT DATASET/imagenet \
        DATASET.TEST_SET imagenet-r \
        TEST.VALID_LABELS DATASET/imagenet/imagenet_r_indices.txt
    echo "======== Test on imagenet-v2 (matched-frequency) ========"
    python3 tools/test.py \
        ${EXTRA_ARGS} \
        VERBOSE False \
        DATASET.DATA_FORMAT jpg \
        DATASET.ROOT DATASET/imagenet \
        DATASET.TEST_SET imagenet-v2/matched-frequency
    echo "======== Test on imagenet-v2 (top-images) ========"
    python3 tools/test.py \
        ${EXTRA_ARGS} \
        VERBOSE False \
        DATASET.DATA_FORMAT jpg \
        DATASET.ROOT DATASET/imagenet \
        DATASET.TEST_SET imagenet-v2/top-images
    echo "======== Test on imagenet-v2 (threshold-0.7) ========"
    python3 tools/test.py \
        ${EXTRA_ARGS} \
        VERBOSE False \
        DATASET.DATA_FORMAT jpg \
        DATASET.ROOT DATASET/imagenet \
        DATASET.TEST_SET imagenet-v2/threshold-0.7
}

train_test() {
    train
    test_all
}

train_test_clip() {
    train_clip
    test_clip_zeroshot
}

swa_finetune_test() {
    swa_finetune
    test_all
}

bit_finetune() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/bit_finetune.py ${EXTRA_ARGS}
}


io() {
    python3 -m torch.distributed.launch \
        --nnodes ${NODE_COUNT} \
        --node_rank ${RANK} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        --nproc_per_node ${GPUS} \
        tools/test_io.py ${EXTRA_ARGS}
}

eval() {
    python3 tools/eval.py ${EXTRA_ARGS}
}

download_checkpoint() {
    wget "$CHECKPOINT" -O checkpoint.pth -q
}


download_azcopy() {
    echo "downloading azcopy ..."
    wget https://azcopyvnext.azureedge.net/release20210226/azcopy_linux_amd64_10.9.0.tar.gz -O azcopy.tar.gz -q
    tar -zxvf azcopy.tar.gz
    mv azcopy_linux_amd64_10.9.0/ azcopy/
    rm azcopy.tar.gz
}


############################ Main #############################
GPUS=`nvidia-smi -L | wc -l`
MASTER_PORT=9000
INSTALL_DEPS=false
DOWNLOAD_AZCOPY=false
TORCH_VERSION=1.6.0

while [[ $# -gt 0 ]]
do

key="$1"
case $key in
    -h|--help)
    echo "Usage: $0 [run_options]"
    echo "Options:"
    echo "  -g|--gpus <1> - number of gpus to be used"
    echo "  -t|--job-type <train> - job type (train|io|bit_finetune|test)"
    echo "  -p|--port <9000> - master port"
    echo "  -i|--install-deps - If install dependencies (default: False)"
    echo "  -d|--download-checkpoint - Download checkpoint file"
    echo "  -a|--download-azcopy - Download azcopy tool"
    echo "  -v|--torch-version - PyTorch version (default: 1.6.0)"
    exit 1
    ;;
    -g|--gpus)
    GPUS=$2
    shift
    ;;
    -t|--job-type)
    JOB_TYPE=$2
    shift
    ;;
    -p|--port)
    MASTER_PORT=$2
    shift
    ;;
    -i|--install-deps)
    INSTALL_DEPS=true
    ;;
    -d|--download-checkpoint)
    CHECKPOINT=$2
    shift
    ;;
    -a|--download-azcopy)
    DOWNLOAD_AZCOPY=true
    ;;
    -v|--torch-version)
    TORCH_VERSION=$2
    shift
    ;;
    *)
    EXTRA_ARGS="$EXTRA_ARGS $1"
    ;;
esac
shift
done

if $INSTALL_DEPS; then
    python -m pip install -r requirements.txt --user -q

    case $TORCH_VERSION in
        "1.6.0")
        python -m pip install --user -q torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
        ;;
        "1.7.1")
        python -m pip install --user -q torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
        ;;
        "1.7.1+cu101")
        python -m pip install --user torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
        ;;
        *)
        echo "unknow PyTorch version, install 1.6.0"
        python -m pip install --user -q torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
    esac
fi

if $DOWNLOAD_AZCOPY; then
    download_azcopy
fi

if [[ -z "$CHECKPOINT" ]]; then
    echo "checkpoint is not set"
else
    echo "downloaing checkpoint ..."
    download_checkpoint
fi

[[ -z "$RANK" ]] && RANK=0
[[ -z "$AZ_BATCHAI_TASK_INDEX" ]] && RANK=0 || RANK=$AZ_BATCHAI_TASK_INDEX
[[ -z "$AZUREML_NODE_COUNT" ]] && NODE_COUNT=1 || NODE_COUNT=$AZUREML_NODE_COUNT
[[ -z "$AZ_BATCH_MASTER_NODE" ]] && MASTER_ADDR=127.0.0.1 || MASTER_ADDR=$(echo "$AZ_BATCH_MASTER_NODE" | cut -d : -f 1)
echo "job type: ${JOB_TYPE}"
echo "rank: ${RANK}"
echo "node count: ${NODE_COUNT}"
echo "master addr: ${MASTER_ADDR}"

case $JOB_TYPE in
    train)
    train
    ;;
    train_clip)
    train_clip
    ;;
    swa_finetune)
    swa_finetune
    ;;
    swa_finetune_test)
    swa_finetune_test
    ;;
    io)
    io
    ;;
    bit_finetune)
    bit_finetune
    ;;
    test)
    test
    ;;
    test_clip_retrieval)
    test_clip_retrieval
    ;;
    test_clip_zeroshot)
    test_clip_zeroshot
    ;;
    train_test_clip)
    train_test_clip
    ;;
    test_all)
    test_all
    ;;
    train_test)
    train_test
    ;;
    eval)
    eval
    ;;
    *)
    echo "unknown job type"
    ;;
esac
