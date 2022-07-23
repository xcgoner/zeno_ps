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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import mxnet as mx

class Attack:
    def __init__(self, rescale):
        assert rescale > 0
        self.rescale = rescale
    def attack(self, params):
        """

        Parameters
        ----------
        params
            collections of model parameters
        """
        raise NotImplementedError

class RandomAttack(Attack):
    def attack(self, update):
        update[:] = mx.nd.random.normal(0, self.rescale, shape=update.shape)

class RandomAttack2(Attack):
    def attack(self, update):
        scale = mx.nd.norm(update) * self.rescale
        update[:] = mx.nd.random.normal(0, 1, shape=update.shape)
        update[:] /= mx.nd.norm(update)
        update[:] *= scale

class NegativeAttack(Attack):
    def attack(self, update):
        update[:] *= (-self.rescale)



