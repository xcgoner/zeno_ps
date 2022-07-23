# Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""This file is modified from
`gluon-cv/scripts/classification/cifar/train_cifar10.py`"""
import argparse
import logging
import subprocess
import time
import os
import random

import mxnet as mx

import byteps.mxnet as bps
from byteps.mxnet.ops import size, local_size, rank, local_rank
from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor, byteps_push, byteps_pull, byteps_declare_and_init_tensor

def main():
    bps.init()

    # ctx_idx = int(os.environ.get('NVIDIA_VISIBLE_DEVICES', '0'))
    # context = mx.gpu(ctx_idx)
    context = mx.cpu()

    # gradient = mx.nd.array([rank()])
    gradient_1 = mx.nd.array([(-1) ** rank()] * 2, ctx=context)
    gradient_2 = mx.nd.array([(-10) ** rank()] * 2, ctx=context)
    parameter = mx.nd.array([0] * 2, ctx=context)

    byteps_declare_and_init_tensor("gradient_1", gradient_1)
    byteps_declare_and_init_tensor("gradient_2", gradient_2)
    byteps_declare_and_init_tensor("parameter", parameter)

    mx.nd.waitall()
    print("worker %d has parameter=" % (rank()), parameter.asnumpy())

    byteps_pull(parameter, name="parameter", priority=0)
    mx.nd.waitall()
    print("worker %d pulled parameter=" % (rank()), parameter.asnumpy())
    
    for i in range(20):

        mx.nd.waitall()
        print("worker %d before push gradient=" % (rank()), gradient_1.asnumpy())

        byteps_push(gradient_2, name="gradient_2", priority=-2)
        byteps_push(gradient_1, name="gradient_1", priority=-1)

        mx.nd.waitall()
        print("worker %d after push gradient=" % (rank()), gradient_1.asnumpy())
        
        byteps_pull(parameter, name="parameter", priority=0)

        gradient_1 += ((-1) ** rank())

        mx.nd.waitall()
        print("worker %d has parameter=" % (rank()), parameter.asnumpy())

        time.sleep(0.005 * rank())
    
    mx.nd.waitall()
    print("worker %d has parameter=" % (rank()), parameter.asnumpy())
        

if __name__ == '__main__':
    main()

