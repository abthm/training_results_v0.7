#!/bin/bash

pip3 install -r requirements.txt
echo "installed requirements.txt"
pip3 install --no-cache-dir https://github.com/mlperf/logging/archive/9ea0afa.zip
echo "installed mlperf logging"
cd mhalib
python3.6 setup.py build && cp build/lib*/mhalib* ../
echo "built and installed mhalib"
cd /
git clone --recursive  https://github.com/ROCmSoftwarePlatform/apex
cd apex
python3.6 setup.py install --cuda_ext --cpp_ext
echo "installed apex"
cd /training_results_v0.7/NVIDIA/benchmarks/bert/implementations/pytorch/
