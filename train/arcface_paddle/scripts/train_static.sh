# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
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

python -m paddle.distributed.launch --gpus=0,1 tools/train.py \
    --config_file configs/ms1mv3_r50.py \
    --is_static True \
    --backbone FresResNet101 \
    --classifier LargeScaleClassifier \
    --embedding_size 512 \
    --model_parallel True \
    --dropout 0.0 \
    --sample_ratio 0.1 \
    --loss ArcFace \
    --batch_size 64 \
    --num_classes 657077 \
    --data_dir ../dataset/ \
    --label_file ../dataset/label.txt \
    --is_bin False \
    --log_interval_step 100 \
    --validation_interval_step 2000 \
    --fp16 True \
    --use_dynamic_loss_scaling True \
    --init_loss_scaling 27648.0 \
    --num_workers 0 \
    --train_unit 'epoch' \
    --warmup_num 0 \
    --train_num 25 \
    --decay_boundaries "10,16,22" \
    --output model
