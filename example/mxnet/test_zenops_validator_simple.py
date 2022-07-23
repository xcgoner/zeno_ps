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

import numpy as np

import mxnet as mx

import byteps.mxnet as bps
from byteps.mxnet.ops import size, local_size, rank, local_rank, worker_size, validator_size
from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor, byteps_push, byteps_pull, byteps_declare_and_init_tensor

def main():
    bps.init()

    # ctx_idx = int(os.environ.get('NVIDIA_VISIBLE_DEVICES', '0'))
    # context = mx.gpu(ctx_idx)
    context = mx.cpu()

    gradient_1 = mx.nd.array([(-1) ** rank()] * 2, ctx=context)
    gradient_2 = mx.nd.array([(-10) ** rank()] * 2, ctx=context)
    parameter = mx.nd.array([1] * 2, ctx=context)

    byteps_declare_and_init_tensor("gradient_1", gradient_1)
    byteps_declare_and_init_tensor("gradient_2", gradient_2)
    byteps_declare_and_init_tensor("parameter", parameter)

    gradient_list = []

    if rank() == 0:
        parameter[:] = -1
    else:
        parameter[:] = 1
    
    mx.nd.waitall()
    print("validator %d has parameter=" % (rank()), parameter.asnumpy())
    
    byteps_push(parameter, name="parameter", priority=0)
    mx.nd.waitall()
    print("validator %d pushed parameter=" % (rank()), parameter.asnumpy())
    
    byteps_pull(parameter, name="parameter", priority=0)
    mx.nd.waitall()
    print("validator %d pulled parameter=" % (rank()), parameter.asnumpy())

    for i in range(20):

        k = 0
        for j in range(worker_size()):
            if j % validator_size() == rank():
                mx.nd.waitall()
                byteps_pull(gradient_2, name="gradient_2", priority=-2)
                byteps_pull(gradient_1, name="gradient_1", priority=-1)
                print("validator %d pull gradient=" % (rank()), gradient_1.asnumpy())
                mx.nd.waitall()
                if k >= len(gradient_list):
                    gradient_list.append(gradient_1.copy())
                gradient_list[k][:] = gradient_1

                mx.nd.waitall()
                print("validator %d pull gradient=" % (rank()), gradient_list[k].asnumpy())
                
                k += 1
        for j in range(1, k):
            gradient_list[0][:] += gradient_list[j]
        
        mx.nd.waitall()
        print("validator %d aggregate gradient=" % (rank()), gradient_list[0].asnumpy())
        
        parameter[:] += gradient_list[0]

        byteps_push(parameter, name="parameter", priority=0)
        byteps_pull(parameter, name="parameter", priority=0)

        mx.nd.waitall()
        print("validator %d has parameter=" % (rank()), parameter.asnumpy())
    
    mx.nd.waitall()
    print("validator %d has parameter=" % (rank()), parameter.asnumpy())


if __name__ == '__main__':
    main()

