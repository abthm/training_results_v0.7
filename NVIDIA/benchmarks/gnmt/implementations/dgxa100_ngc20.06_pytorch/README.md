## Steps to launch training

### NVIDIA DGX A100 (single node)
Launch configuration and system-specific hyperparameters for the NVIDIA DGX A100
single node submission are in the `config_DGXA100.sh` script.

Steps required to launch single node training on NVIDIA DGX A100:

1. Build the container and push to a docker registry:

```
cd ../implementations/pytorch
docker build --pull -t <docker/registry>/mlperf-nvidia:rnn_translator .
docker push <docker/registry>/mlperf-nvidia:rnn_translator
```

2. Launch the training:

```
source config_DGXA100.sh
CONT="<docker/registry>/mlperf-nvidia:rnn_translator" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N $DGXNNODES -t $WALLTIME run.sub
```

#### Alternative launch with nvidia-docker

When generating results for the official v0.7 submission with one node, the
benchmark was launched onto a cluster managed by a SLURM scheduler. The
instructions in [NVIDIA DGX A100 (single node)](#nvidia-dgx-a100-single-node) explain
how that is done.

However, to make it easier to run this benchmark on a wider set of machine
environments, we are providing here an alternate set of launch instructions
that can be run using nvidia-docker. Note that performance or functionality may
vary from the tested SLURM instructions.

```
cd ../implementations/pytorch
docker build --pull -t mlperf-nvidia:rnn_translator .
source config_DGXA100.sh
CONT="mlperf-nvidia:rnn_translator" DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> ./run_with_docker.sh
```
