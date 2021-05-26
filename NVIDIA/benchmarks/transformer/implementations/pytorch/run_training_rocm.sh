#!/bin/bash

## DL params
## Borrowed from config_DGX2.sh 
export MAX_TOKENS=10240 ##8192 ##6912 ##5120
export LEARNING_RATE="1.976e-3" ##1.732e-3 ##1.9e-3"
export WARMUP_UPDATES=1000
export EXTRA_PARAMS="--distributed-weight-update 0 --dwu-num-blocks 4 --dwu-num-rs-pg 2 --dwu-num-ar-pg 2 --dwu-num-ag-pg 0 --dwu-overlap-reductions --dwu-num-chunks 1 --dwu-flat-mt --dwu-compute-L2-grad-norm --max-source-positions 64 --max-target-positions 64 --adam-betas (0.9,0.98) "

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=02:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=16 ##16 on MI100
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1

#DATADIR="/workspace/transformer-mlperf-data/v07-sumbission-data/wmt14_en_de/utf8"
LOGDIR="./results"

mkdir -p "${LOGDIR}"

SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-$DGXNGPU}


SEED=${SEED:-$RANDOM}
MAX_TOKENS=${MAX_TOKENS}
DATASET_DIR=${DATASET_DIR:-"/workspace/transformer-mlperf-data/wmt14_en_de/utf8"}
NUMEPOCHS=${NUMEPOCHS:-30}

# Start timing
START=$(date +%s)
START_FMT=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT ${START_FMT}"

if [[ ${WORLD_SIZE:-${SLURM_NTASKS}} -ne 1 ]]; then
    DISTRIBUTED_INIT_METHOD="--distributed-init-method env://"
else
    DISTRIBUTED_INIT_METHOD="--distributed-world-size 1"
fi

# These are scanned by train.py, so make sure they are exported
export DGXSYSTEM
export SLURM_NTASKS_PER_NODE
export SLURM_NNODES

declare -a CMD
if [ -n "${SLURM_LOCALID-}" ]; then
  # Mode 1: Slurm launched a task for each GPU and set some envvars; no need for parallel launch
  if [ "${SLURM_NTASKS}" -gt "${SLURM_JOB_NUM_NODES}" ]; then
    CMD=( './bind.sh' '--cpu=exclusive' '--' 'python' '-u' )
  else
    CMD=( 'python' '-u' )
  fi
else
  # Mode 2: Single-node Docker; need to launch tasks with Pytorch's distributed launch
  # TODO: use bind.sh instead of bind_launch.py
  #       torch.distributed.launch only accepts Python programs (not bash scripts) to exec
  CMD=( 'python' '-u' '-m' 'bind_launch' "--nsockets_per_node=${DGXNSOCKET}" \
    "--ncores_per_socket=${DGXSOCKETCORES}" "--nproc_per_node=${DGXNGPU}" )
fi

"${CMD[@]}" train.py ${DATASET_DIR} \
  --seed ${SEED} \
  --arch transformer_wmt_en_de_big_t2t \
  --share-all-embeddings \
  --optimizer adam \
  --adam-betas '(0.9, 0.997)' \
  --adam-eps "1e-9" \
  --clip-norm "0.0" \
  --lr-scheduler inverse_sqrt \
  --warmup-init-lr "0.0" \
  --warmup-updates ${WARMUP_UPDATES} \
  --lr ${LEARNING_RATE} \
  --min-lr "0.0" \
  --dropout "0.1" \
  --weight-decay "0.0" \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing "0.1" \
  --max-tokens ${MAX_TOKENS} \
  --max-epoch ${NUMEPOCHS} \
  --target-bleu "25.0" \
  --ignore-case \
  --no-save \
  --update-freq 1 \
  --fp16 \
  --seq-len-multiple 2 \
  --source_lang en \
  --target_lang de \
  --bucket_growth_factor 1.035 \
  --batching_scheme "v0p5_better" \
  --batch_multiple_strategy "dynamic" \
  --fast-xentropy \
  --max-len-a 1 \
  --max-len-b 50 \
  --lenpen 0.6 \
  --no-progress-bar \
  --dataloader-num-workers 2 \
  --enable-dataloader-pin-memory \
  ${DISTRIBUTED_INIT_METHOD} \
  ${EXTRA_PARAMS} ; ret_code=$?

sleep 3
if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# End timing
END=$(date +%s)
END_FMT=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT ${END_FMT}"

# Report result
RESULT=$(( ${END} - ${START} ))
RESULT_NAME="transformer"
echo "RESULT,${RESULT_NAME},${SEED},${RESULT},${USER},${START_FMT}"
