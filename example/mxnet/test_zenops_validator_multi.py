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

import numpy as np

import mxnet as mx

import byteps.mxnet as bps
from byteps.mxnet.ops import size, local_size, rank, local_rank, worker_size, validator_size
from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor, byteps_push, byteps_pull, byteps_declare_and_init_tensor

def main():
    bps.init()

    a = mx.nd.ones((5,5,5)) * (rank()+1) * 3
    b = mx.nd.ones((5,5,5)) * (rank()+1)

    byteps_declare_and_init_tensor("parameter_a", a)
    byteps_declare_and_init_tensor("update_b", b)

    print("validator ", rank())

    print("validator %d has parameter_a" % (rank()), a.asnumpy()[0,0])
    byteps_push(a, name="parameter_a", priority=0)
    byteps_pull(a, name="parameter_a", priority=0)
    print("validator %d has parameter_a" % (rank()), a.asnumpy()[0,0])

    for i in range(1):

        b[:] = 1

        for j in range(worker_size()):
            if j % validator_size() == rank():
                print("validator %d pulling update_b" % (rank()), b.asnumpy()[0, 0])
                byteps_pull(b, name="update_b", priority=0)
                print("validator %d pulled update_b" % (rank()), b.asnumpy()[0, 0])

                a[:] += b

                byteps_push(a, name="parameter_a", priority=0)
                byteps_pull(a, name="parameter_a", priority=0)
                print("validator %d has parameter_a" % (rank()), a.asnumpy()[0,0])


if __name__ == '__main__':
    main()

