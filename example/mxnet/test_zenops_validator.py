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
    b = mx.nd.ones((3,3,3)) * (rank()+1)
    c = mx.nd.array([0])

    byteps_declare_and_init_tensor("parameter_a", a)
    byteps_declare_and_init_tensor("update_b", b)
    byteps_declare_and_init_tensor("indicator_c", c)

    print("validator ", rank())

    for i in range(5):

        a[:] = (rank()+1) * 3 * (2**i)

        byteps_push(a, name="parameter_a", priority=0)
        mx.nd.waitall()
        # print("validator %d pushed parameter_a" % (rank()), a.asnumpy()[0])

        a[:] = 1
        mx.nd.waitall()
        # print("validator %d before pull parameter_a" % (rank()), a.asnumpy()[0])

        byteps_pull(a, name="parameter_a", priority=0)
        mx.nd.waitall()
        # print("validator %d pulled parameter_a" % (rank()), a.asnumpy()[0])
        # print(validator_size(), rank())
        validation_val = (mx.nd.norm(a/validator_size() - (validator_size()+1)/2.0*3*(2**i)).asscalar())
        print(validation_val)
        assert validation_val < 1e-5, "fail in push validation %f, %f" % (a[0,0,0].asscalar(), (validator_size()+1)/2.0*3*(2**i))

        b[:] = 1

        for j in range(worker_size()):
            if j % validator_size() == rank():
                # print("validator %d pulling update_b" % (rank()), b.asnumpy()[0])
                byteps_pull(b, name="update_b", priority=0)
                mx.nd.waitall()
                # print("validator %d pulled update_b" % (rank()), b.asnumpy()[0])

                validation_val = (mx.nd.norm(b - 4*(2**i)).asscalar())
                print(validation_val)
                assert validation_val < 1e-5, "fail in fetching from workers %f, %f" % (b[0,0].asscalar(), 4*(2**i))

                byteps_pull(c, name="indicator_c", priority=0)
                mx.nd.waitall()
                print("validator %d pulled indicator_c" % (rank()), c.asnumpy()[0])


if __name__ == '__main__':
    main()

