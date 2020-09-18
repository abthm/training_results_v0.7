## DL params
export BATCHSIZE=32
export LR=3.5e-4
export GRADIENT_STEPS=1
export MAX_STEPS=13700
export WARMUP_PROPORTION=0.0
export PHASE=2
export MAX_SAMPLES_TERMINATION=4500000
export EXTRA_PARAMS="--unpad"

## System run parms
export DGXNNODES=1
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )
export WALLTIME=02:00:00

## System config params
export DGXNGPU=8
export DGXSOCKETCORES=24
export DGXNSOCKET=2
export DGXHT=2         # HT is on is 2, HT off is 1
