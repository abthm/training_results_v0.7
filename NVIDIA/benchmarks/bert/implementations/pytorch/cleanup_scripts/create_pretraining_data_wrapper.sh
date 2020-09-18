#!/bin/bash
# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
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

input_path=${1}
output_dir="hdf5"

#input_file=$(basename $input_path)
echo $input_path
#echo $input_file
mkdir -p ${output_dir}

for f in ${input_path}*
do
#f=$input_path/*
#f=$input_path
        echo $f
        input_file=$(basename $f)
        python3 ../create_pretraining_data.py \
        --input_file=$f \
        --output_file="${output_dir}/${input_file}" \
        --vocab_file='/workspace/bert-mlperf-data/vocab.txt' \
        --do_lower_case=True \
        --max_seq_length=512 \
        --max_predictions_per_seq=76 \
        --masked_lm_prob=0.15 \
        --random_seed=12345 \
        --dupe_factor=10
done

