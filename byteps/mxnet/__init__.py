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

import warnings
import mxnet as mx
import os
import time
import random
import math

import numpy as np

from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor
from byteps.mxnet.ops import init, shutdown, suspend, resume
from byteps.mxnet.ops import size, local_size, rank, local_rank

# zeno ps
from byteps.mxnet.ops import worker_size, validator_size
from byteps.mxnet.ops import byteps_push, byteps_pull, byteps_declare_and_init_tensor

from byteps.mxnet.validator import NaiveValidator, TrimmedMeanValidator, PhocasValidator, ZenoValidator, ZenoppValidator, NaiveAsyncValidator, FedAsyncValidator
from byteps.mxnet.attacks import RandomAttack, RandomAttack2, NegativeAttack

parameter_index = 0


class DistributedOptimizer(mx.optimizer.Optimizer):
    """This is where BytePS's DistributedOptimizer wrapper for MXNet goes"""
    def __init__(self, optimizer):
        self._optimizer = optimizer
        self._enable_async = (int(os.getenv('BYTEPS_ENABLE_ASYNC', 0)) != 0)
        if self._enable_async:
            assert int(os.getenv('DMLC_NUM_WORKER'))>1, \
                "Async is only valid for distributed training"
            print('BytePS: enable asynchronous training')

    def __getattr__(self, item):
        return getattr(self._optimizer, item)

    def create_state_multi_precision(self, index, weight):
        return self._optimizer.create_state_multi_precision(index, weight)

    def _do_push_pull(self, index, grad):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor("gradient_" + str(index[i]))
                byteps_push_pull(grad[i], version=0, priority=-index[i],
                                 name="gradient_" + str(index[i]), is_average=True)
        else:
            byteps_declare_tensor("gradient_" + str(index))
            byteps_push_pull(grad, version=0, priority=-index,
                             name="gradient_" + str(index), is_average=True)

    def _do_push_pull_param(self, index, delta_weight):
        if isinstance(index, (tuple, list)):
            for i in range(len(index)):
                byteps_declare_tensor("weight_" + str(index[i]))
                byteps_push_pull(delta_weight[i], version=0, priority=-index[i],
                                 name="weight_" + str(index[i]), is_average=False)
        else:
            byteps_declare_tensor("weight_" + str(index))
            byteps_push_pull(delta_weight, version=0, priority=-index,
                             name="weight_" + str(index), is_average=False)

    def update(self, index, weight, grad, state):
        if self._enable_async:
            # create a tmp list for storing the original weight
            temp_weight_list = [w.copy() for w in weight]
            assert len(temp_weight_list) == len(weight)

            # update parameter locally
            self._optimizer.update(index, weight, grad, state)

            # get delta weight
            for i, temp_weight in enumerate(temp_weight_list):
                weight[i].__isub__(temp_weight)

            # push delta weight, and pull weight back to the same tensor
            self._do_push_pull_param(index, weight)

        else:
            self._do_push_pull(index, grad)
            self._optimizer.update(index, weight, grad, state)

    def update_multi_precision(self, index, weight, grad, state):
        if self._enable_async:
            # create a tmp list for storing the original weight
            temp_weight_list = [w.copy() for w in weight]
            assert len(temp_weight_list) == len(weight)

            # update parameter locally
            self._optimizer.update_multi_precision(index, weight, grad, state)

            # get delta weight
            for i, temp_weight in enumerate(temp_weight_list):
                weight[i].__isub__(temp_weight)

            # push delta weight, and pull weight back to the same tensor
            self._do_push_pull_param(index, weight)

        else:
            self._do_push_pull(index, grad)
            self._optimizer.update_multi_precision(index, weight, grad, state)

    def set_learning_rate(self, lr):
        self._optimizer.set_learning_rate(lr)

    def set_lr_mult(self, args_lr_mult):
        self._optimizer.set_lr_mult(args_lr_mult)

    def set_wd_mult(self, args_wd_mult):
        self._optimizer.set_wd_mult(args_wd_mult)


def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()`.

    Arguments:
        params: dict of parameters to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    global parameter_index

    if isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]

        # Run tensor initilization
        for i in range(len(tensors)):
            byteps_declare_tensor("parameter_" + str(parameter_index))
            # Broadcast is implemented as push + pull in BytePS
            # To broadcast: we should zero-out all non-root tensors, and disable push_pull average
            if rank() != root_rank:
                tensors[i].__imul__(0)
            byteps_push_pull(tensors[i], version=0, priority=0,
                             name="parameter_" + str(parameter_index), is_average=False)
            parameter_index += 1

        # Make sure tensors pushed to MXNet engine get processed such that all
        # workers are synced before starting training.
        for tensor in tensors:
            tensor.wait_to_read()

    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        raise TypeError("For gluon users, you should not call this function. "
                        "DistributedTrainer will broadcast all parameters at "
                        "the first training step.")

    else:
        raise ValueError('Invalid params of type: %s' % type(params))


class DistributedTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using BytePS push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs push_pull(summation) and average,
       while Trainer only performs push_pull(summation).

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, optimizer_params=None, root_rank=0):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        super(DistributedTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)

        # _scale is used to check and set rescale_grad for optimizer in Trainer.step()
        # function. Normalizing it by BytePS size, which is equivalent to performing
        # average in push_pull, has better performance.
        self._scale /= size()
        self.root_rank = root_rank
        for i, param in enumerate(self._params):
            byteps_declare_tensor("parameter_" + str(i))
            if param.grad_req != 'null':
                byteps_declare_tensor("gradient_" + str(i))


    def _allreduce_grads(self):
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                # byteps_push_pull(param.list_grad()[0], is_average=False,
                #                  name="gradient_" + str(i), priority=-i)
                byteps_push(param.list_grad()[0], name="gradient_" + str(i), priority=-i)
                byteps_pull(param.list_grad()[0], name="gradient_" + str(i), priority=-i)

    def _init_params(self):
        tensors = []
        for param in self._params_to_init:
            if param._deferred_init:
                tensors.append(param)
            else:
                param_arrays = param._check_and_get(param._data, list)
                idx = self._param2idx[param.name]

                if rank() != self.root_rank:
                    param_arrays[0].__imul__(0)
                # byteps_push_pull(param_arrays[0], version=0, priority=0,
                #                  name="parameter_" + str(idx), is_average=False)
                byteps_push(param_arrays[0], version=0, priority=0, name="parameter_" + str(idx))
                byteps_pull(param_arrays[0], version=0, priority=0, name="parameter_" + str(idx))

        self._params_to_init = tensors


# training with validators
class DistributedZenoWorkerSyncTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using BytePS push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs push_pull(summation) and average,
       while Trainer only performs push_pull(summation).

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, optimizer_params=None, sync_interval=1, worker_subsample_rate=1.0, sparse_rate=0.0, attack_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedZenoWorkerSyncTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        super(DistributedZenoWorkerSyncTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)

        self.sync_interval = sync_interval
        self.sync_counter = 0

        self.worker_subsample_rate = worker_subsample_rate
        self.sparse_rate = sparse_rate

        self.zenops_initialized = False

        self.attacker = None
        if attack_params is not None:
            if attack_params["byz_type"] == "random":
                self.attacker = RandomAttack(rescale=attack_params["byz_scale"])
                print("using RandomAttack with rescale=%f" % (self.attacker.rescale))
            elif attack_params["byz_type"] == "randomscale":
                self.attacker = RandomAttack2(rescale=attack_params["byz_scale"])
                print("using RandomAttack2 with rescale=%f" % (self.attacker.rescale))
            elif attack_params["byz_type"] == "negative":
                self.attacker = NegativeAttack(rescale=attack_params["byz_scale"])
                print("using NegativeAttack with rescale=%f" % (self.attacker.rescale))
            self.byz_rate = attack_params["byz_rate"] if "byz_rate" in attack_params else 0.0
    
    def _init_zenops(self):
        if self.zenops_initialized:
            return
        
        # indicator for worker subsampling
        self.worker_sparse_indicator = mx.nd.array([1])
        byteps_declare_and_init_tensor("worker_sparse_indicator", self.worker_sparse_indicator)
        # indicate whether to send this layer or not, for communication compression
        self.block_sparse_indicators = []

        # cache for the previous model
        self.cached_params = []

        for i, param in enumerate(self._params):
            self.block_sparse_indicators.append(mx.nd.array([1]))
            if param.grad_req != 'null':
                byteps_declare_and_init_tensor("block_sparse_indicator_" + str(i), self.block_sparse_indicators[-1])
                byteps_declare_and_init_tensor("parameter_" + str(i), param.list_data()[0])
                # initialize the parameters by pulling from validators
                byteps_pull(param.list_data()[0], priority=0, name="parameter_" + str(i))
                self.cached_params.append(param.list_data()[0].copy())
                byteps_declare_and_init_tensor("gradient_" + str(i), param.list_grad()[0])
            else:
                self.cached_params.append(None)
            
            # print("initialized " + str(i))
        
        self.zenops_initialized = True
        self.sync_counter = 0

        mx.nd.waitall()
        print("_init_zenops finished")


    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        self._update(ignore_stale_grad)

        self._init_zenops()

        self.sync_counter += 1
        if self.sync_counter == self.sync_interval:
            self.sync_counter = 0

            mx.nd.waitall()
            # time.sleep(0.05 * (rank()))

            # TODO: worker subsampling
            # tell the validator that this worker is going to send updates in this round
            if random.uniform(0, 1) <= self.worker_subsample_rate:
                self.worker_sparse_indicator[:] = rank() + 1
                byteps_push(self.worker_sparse_indicator, name="worker_sparse_indicator", priority=0)
                mx.nd.waitall()
                # time.sleep(0.05 * (rank()))
                for i, (param, cached_param_data, send_layer) in enumerate(zip(self._params, self.cached_params, self.block_sparse_indicators)):
                    if param.grad_req != 'null':
                        param.list_grad()[0][:] = param.list_data()[0] - cached_param_data

                        # TODO: layer sparsification
                        # tell the validator that this worker is going to send updates in this round
                        send_layer[:] = rank() + 1
                        if random.uniform(0, 1) < self.sparse_rate:
                            # skip communication
                            send_layer[:] *= (-1)

                            byteps_push(send_layer, name="block_sparse_indicator_" + str(i), priority=-i)

                            # error reset
                            cached_param_data[:] = param.list_grad()[0]

                        else:
                            cached_param_data[:] = 0
                            byteps_push(send_layer, name="block_sparse_indicator_" + str(i), priority=-i)

                            if self.attacker and self.byz_rate and random.uniform(0, 1) < self.byz_rate:
                                self.attacker.attack(param.list_grad()[0])

                            byteps_push(param.list_grad()[0], name="gradient_" + str(i), priority=-i)
                        
            else:
                self.worker_sparse_indicator[:] = -rank() - 1
                byteps_push(self.worker_sparse_indicator, name="worker_sparse_indicator", priority=0)

                for i, (param, cached_param_data, send_layer) in enumerate(zip(self._params, self.cached_params, self.block_sparse_indicators)):
                    if param.grad_req != 'null':
                        # error reset
                        cached_param_data[:] = param.list_data()[0] - cached_param_data
            
            for i, (param, cached_param_data, send_layer) in enumerate(zip(self._params, self.cached_params, self.block_sparse_indicators)):
                if param.grad_req != 'null':
                    param.list_data()[0][:] = 0
                    byteps_pull(param.list_data()[0], name="parameter_" + str(i), priority=-i)
                    param.list_data()[0][:] += cached_param_data
                    cached_param_data[:] = param.list_data()[0]
            
            mx.nd.waitall()
            # time.sleep(0.05 * (rank()))
            # time.sleep(random.uniform(0,1))

# training with validators
class DistributedZenoValidatorSyncTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using BytePS push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs push_pull(summation) and average,
       while Trainer only performs push_pull(summation).

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, optimizer_params=None, validation_type="average", sync_interval=1, zeno_eta=-0.001):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedZenoValidatorSyncTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        super(DistributedZenoValidatorSyncTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)

        self.sync_interval = sync_interval
        self.sync_counter = 0

        self.validation_type = validation_type
        self.zenops_initialized = False

        self.zeno_eta = zeno_eta
    
    def _init_zenops(self):
        if self.zenops_initialized:
            return
        
        # indicator for worker subsampling
        self.worker_sparse_indicator = mx.nd.array([1])
        byteps_declare_and_init_tensor("worker_sparse_indicator", self.worker_sparse_indicator)
        # indicate whether to send this layer or not, for communication compression
        self.block_sparse_indicators = []

        # cache for the previous model
        self.cached_params = []

        # cache for the updates sent from workers
        self.cached_updates = []

        self.validators = []

        for i, param in enumerate(self._params):
            self.block_sparse_indicators.append(mx.nd.array([1]))
            if param.grad_req != 'null':
                byteps_declare_and_init_tensor("block_sparse_indicator_" + str(i), self.block_sparse_indicators[-1])
                byteps_declare_and_init_tensor("parameter_" + str(i), param.list_data()[0])
                # initialize the parameters by pulling from validators
                if rank() != 0:
                    param.list_data()[0][:] = 0
                byteps_push(param.list_data()[0], priority=0, name="parameter_" + str(i))
                byteps_pull(param.list_data()[0], priority=0, name="parameter_" + str(i))
                self.cached_params.append(param.list_data()[0].copy())

                byteps_declare_and_init_tensor("gradient_" + str(i), param.list_grad()[0])
                self.cached_updates.append([param.list_grad()[0].copy()])

                if self.validation_type == "average":
                    self.validators.append(NaiveValidator())
                elif self.validation_type == "trimmed_mean":
                    self.validators.append(TrimmedMeanValidator(ratio_trimmed=0.2))
                elif self.validation_type == "phocas":
                    self.validators.append(PhocasValidator(ratio_trimmed=0.2))
                elif self.validation_type == "zeno":
                    self.validators.append(ZenoValidator(eta=self.zeno_eta, rho=0.6))
                else:
                    raise ValueError('Undefined validation_type: %s' % self.validation_type)
            else:
                self.cached_params.append(None)
                self.cached_updates.append(None)
                self.validators.append(None)
            # print("initialized " + str(i))
        
        self.zenops_initialized = True
        self.sync_counter = 0
        self.validation_counter = 0.0
        self.recv_counter = 0.0

        mx.nd.waitall()
        print("_init_zenops finished")


    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if not self.zenops_initialized or "zeno" in self.validation_type.lower():
            self._update(ignore_stale_grad)
            # print("model initialized, only for the first batch")
        # self._update(ignore_stale_grad)

        self._init_zenops()

        self.sync_counter += 1
        if self.sync_counter == self.sync_interval:
            self.sync_counter = 0

            mx.nd.waitall()
            # time.sleep(0.05 * (worker_size()+1))
            # time.sleep(0.05)

            num_active_workers = 0
            active_workers = set()
            for j in range(worker_size()):
                if j % validator_size() == rank():
                    byteps_pull(self.worker_sparse_indicator, name="worker_sparse_indicator", priority=0)
                    active_worker = self.worker_sparse_indicator[0].asscalar()
                    if active_worker in active_workers:
                        print("BytePS error active_worker")
                        time.sleep(0.1 * (worker_size()+1))
                    # assert active_worker not in active_workers
                    active_workers.add(active_worker)
                    num_active_workers += (1 if active_worker > 0 else 0)
            
            mx.nd.waitall()
            # time.sleep(0.05 * (worker_size()+1))
            # time.sleep(0.05)

            # debug
            # print("num_active_workers: " + str(num_active_workers), flush=True)

            for i, (param, cached_param_data, cached_update_list, send_layer, validator) \
                in enumerate(zip(self._params, self.cached_params, self.cached_updates, self.block_sparse_indicators, self.validators)):
                if param.grad_req != 'null':
                    active_layer_workers = set()
                    sender_error_counter = -num_active_workers - 1
                    for k in range(num_active_workers):
                        byteps_pull(send_layer, name="block_sparse_indicator_" + str(i), priority=-i)
                        active_layer_worker = np.asscalar(send_layer.asscalar())
                        if active_layer_worker in active_layer_workers:
                            print("BytePS error active_layer_worker")
                            active_layer_worker = sender_error_counter
                            sender_error_counter -= 1
                        # assert active_layer_worker not in active_layer_workers
                        active_layer_workers.add(active_layer_worker)
                        if k >= len(cached_update_list):
                            cached_update_list.append(param.list_grad()[0].copy())
                        if active_layer_worker > 0:
                            byteps_pull(param.list_grad()[0], name="gradient_" + str(i), priority=-i)
                            cached_update_list[k][:] = param.list_grad()[0]
                        else:
                            # sparse communication
                            cached_update_list[k][:] = 0
                    
                    validation_info = {'num_tensors': num_active_workers}
                    # validation_info = {'num_tensors': sum((active_layer > 0) for active_layer in active_layer_workers)}
                    if self.validation_type == "average":
                        pass
                    elif self.validation_type == "trimmed_mean":
                        pass
                    elif self.validation_type == "phocas":
                        pass
                    elif self.validation_type == "zeno":
                        param.list_grad()[0][:] = param.list_data()[0] - cached_param_data
                        param.list_data()[0][:] = cached_param_data
                        validation_info.update({'validation_tensor': param.list_grad()[0]})
                    else:
                        raise ValueError('Undefined validation_type: %s' % self.validation_type)
                    if "zeno" in self.validation_type:
                        self.validation_counter += validator.validate(cached_update_list, cached_update_list[0], validation_info)
                        self.recv_counter += num_active_workers
                    else:
                        validator.validate(cached_update_list, cached_update_list[0], validation_info)
                    
                    param.list_data()[0][:] += cached_update_list[0]
                    param.list_data()[0][:] /= validator_size()

                    # if self.validation_type == "average":
                    #     for k in range(1, num_active_layer_workers):
                    #         cached_update_list[0][:] += cached_update_list[k]
                    #     cached_update_list[0][:] /= (num_active_workers * validator_size())
                    #     param.list_data()[0][:] += cached_update_list[0]
                    # else:
                    #     raise ValueError('Undefined validation_type: %s' % self.validation_type)

                    # after validation is done, push the parameter
                    byteps_push(param.list_data()[0], name="parameter_" + str(i), priority=-i)

            mx.nd.waitall()

            for i, (param, cached_param_data) in enumerate(zip(self._params, self.cached_params)):
                if param.grad_req != 'null':
                    byteps_pull(param.list_data()[0], name="parameter_" + str(i), priority=-i)
                    cached_param_data[:] = param.list_data()[0]
            
            mx.nd.waitall()
            # time.sleep(0.05 * (worker_size()+1))
            # time.sleep(0.05)
                

# # Async

# async training with validators
class DistributedZenoWorkerAsyncTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using BytePS push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs push_pull(summation) and average,
       while Trainer only performs push_pull(summation).

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, optimizer_params=None, rho = 0, sync_interval=1, worker_subsample_rate=1.0, sparse_rate=0.0, attack_params=None):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedZenoWorkerAsyncTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        super(DistributedZenoWorkerAsyncTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)

        assert 0 <= rho < 1
        self.rho = rho

        self.sync_interval = sync_interval
        self.sync_counter = 0

        self.worker_subsample_rate = worker_subsample_rate

        self.sparse_rate = sparse_rate

        self.zenops_initialized = False

        self.attacker = None
        if attack_params is not None:
            if attack_params["byz_type"] == "random":
                self.attacker = RandomAttack(rescale=attack_params["byz_scale"])
                print("using RandomAttack with rescale=%f" % (self.attacker.rescale))
            elif attack_params["byz_type"] == "randomscale":
                self.attacker = RandomAttack2(rescale=attack_params["byz_scale"])
                print("using RandomAttack2 with rescale=%f" % (self.attacker.rescale))
            elif attack_params["byz_type"] == "negative":
                self.attacker = NegativeAttack(rescale=attack_params["byz_scale"])
                print("using NegativeAttack with rescale=%f" % (self.attacker.rescale))
            self.byz_rate = attack_params["byz_rate"] if "byz_rate" in attack_params else 0.0
    
    def _init_zenops(self):
        if self.zenops_initialized:
            return

        # timestamps
        self.global_timestamp = mx.nd.array([0])
        self.local_timestamp = mx.nd.array([0])
        byteps_declare_and_init_tensor("global_timestamp", self.global_timestamp)
        
        # indicator for worker subsampling
        self.worker_sparse_indicator = mx.nd.array([1])
        byteps_declare_and_init_tensor("worker_sparse_indicator", self.worker_sparse_indicator)
        # indicate whether to send this layer or not, for communication compression
        self.block_sparse_indicators = []

        # cache for the previous model
        self.cached_params = []

        for i, param in enumerate(self._params):
            self.block_sparse_indicators.append(mx.nd.array([1]))
            if param.grad_req != 'null':
                byteps_declare_and_init_tensor("block_sparse_indicator_" + str(i), self.block_sparse_indicators[-1])
                byteps_declare_and_init_tensor("parameter_" + str(i), param.list_data()[0])
                byteps_declare_and_init_tensor("gradient_" + str(i), param.list_grad()[0])
            
        while self.global_timestamp.asscalar() == 0:
            print("blocked until the model is initialized on server")
            time.sleep(0.1)
            byteps_pull(self.global_timestamp, priority=0, name="global_timestamp")
        print("current timestamp is %d" % (self.global_timestamp.asscalar()))

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                # initialize the parameters by pulling from validators
                byteps_pull(param.list_data()[0], priority=0, name="parameter_" + str(i))
                self.cached_params.append(param.list_data()[0].copy())
            else:
                self.cached_params.append(None)
        
        self.zenops_initialized = True
        self.sync_counter = 0

        mx.nd.waitall()
        print("_init_zenops finished")


    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        self._update(ignore_stale_grad)

        # proximal update
        if self.zenops_initialized and self.rho:
            for i, (param, cached_param_data) in enumerate(zip(self._params, self.cached_params)):
                if param.grad_req != 'null':
                    param.list_data()[0][:] *= (1-self.rho)
                    param.list_data()[0][:] += cached_param_data * self.rho

        self._init_zenops()

        self.sync_counter += 1
        if self.sync_counter == self.sync_interval:
            self.sync_counter = 0

            mx.nd.waitall()
            # add some random delay to simulate asynchrony
            time.sleep(1.0 * worker_size() * random.uniform(0, 1))

            # TODO: worker subsampling
            if random.uniform(0, 1) <= self.worker_subsample_rate:
                # tell the validator that this worker is going to send updates in this round
                self.worker_sparse_indicator[:] = rank() + 1
                byteps_push(self.worker_sparse_indicator, name="worker_sparse_indicator", priority=0)
                mx.nd.waitall()
                time.sleep(0.001 * worker_size() * random.uniform(0, 1))
                for i, (param, cached_param_data, send_layer) in enumerate(zip(self._params, self.cached_params, self.block_sparse_indicators)):
                    if param.grad_req != 'null':
                        param.list_grad()[0][:] = param.list_data()[0] - cached_param_data

                        # TODO: layer sparsification
                        # tell the validator that this worker is going to send updates in this round
                        send_layer[:] = rank() + 1
                        if random.uniform(0, 1) < self.sparse_rate:
                            # skip communication
                            send_layer[:] *= (-1)

                            byteps_push(send_layer, name="block_sparse_indicator_" + str(i), priority=-i)

                            # error reset
                            cached_param_data[:] = param.list_grad()[0]

                        else:
                            cached_param_data[:] = 0
                            byteps_push(send_layer, name="block_sparse_indicator_" + str(i), priority=-i)

                            if self.attacker and self.byz_rate and random.uniform(0, 1) < self.byz_rate:
                                self.attacker.attack(param.list_grad()[0])

                            byteps_push(param.list_grad()[0], name="gradient_" + str(i), priority=-i)
                
                while self.global_timestamp.asscalar() <= self.local_timestamp.asscalar():
                    time.sleep(0.01)
                    byteps_pull(self.global_timestamp, name="global_timestamp", priority=0)
                self.local_timestamp[:] = self.global_timestamp

                mx.nd.waitall()
                time.sleep(0.005 * worker_size())
                
                for i, (param, cached_param_data, send_layer) in enumerate(zip(self._params, self.cached_params, self.block_sparse_indicators)):
                    if param.grad_req != 'null':
                        param.list_data()[0][:] = 0
                        byteps_pull(param.list_data()[0], name="parameter_" + str(i), priority=-i)
                        param.list_data()[0][:] += cached_param_data
                        cached_param_data[:] = param.list_data()[0]
            else:
                self.worker_sparse_indicator[:] = -rank() - 1
                byteps_push(self.worker_sparse_indicator, name="worker_sparse_indicator", priority=0)
            
            mx.nd.waitall()
            time.sleep(0.01 * worker_size() * random.uniform(0, 1))

# async validation
class DistributedZenoValidatorAsyncTrainer(mx.gluon.Trainer):
    """A subclass of MXNet gluon.Trainer.

    There are two differences between DistributedTrainer and Trainer:
    1. DistributedTrainer calculates gradients using BytePS push pull
       API while Trainer does it using kvstore push/pull APIs;
    2. DistributedTrainer performs push_pull(summation) and average,
       while Trainer only performs push_pull(summation).

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use. See
        `help <http://mxnet.io/api/python/optimization/optimization.html#the-mxnet-optimizer-package>`_
        on Optimizer for a list of available optimizers.
    optimizer_params : dict
        Key-word arguments to be passed to optimizer constructor. For example,
        `{'learning_rate': 0.1}`. All optimizers accept learning_rate, wd (weight decay),
        clip_gradient, and lr_scheduler. See each optimizer's
        constructor for a list of additional supported arguments.
    """

    def __init__(self, params, optimizer, optimizer_params=None, rho = 0, alpha = 0.8, validation_type="zenopp", sync_interval=1):
        if isinstance(optimizer, DistributedOptimizer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedZenoValidatorAsyncTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        param_list = []
        if isinstance(params, mx.gluon.ParameterDict):
            for key in sorted(list(params.keys())):
                param_list.append(params[key])

        super(DistributedZenoValidatorAsyncTrainer, self).__init__(
            param_list, optimizer, optimizer_params=optimizer_params, kvstore=None)
        
        self.rho = rho
        self.alpha = alpha

        self.sync_interval = sync_interval
        self.sync_counter = 0

        self.validation_type = validation_type
        self.zenops_initialized = False
    
    def _init_zenops(self):
        if self.zenops_initialized:
            return

        # timestamps
        self.global_timestamp = mx.nd.array([0])
        self.local_timestamp = mx.nd.array([0])
        byteps_declare_and_init_tensor("global_timestamp", self.global_timestamp)
        
        # indicator for worker subsampling
        self.worker_sparse_indicator = mx.nd.array([1])
        byteps_declare_and_init_tensor("worker_sparse_indicator", self.worker_sparse_indicator)
        # indicate whether to send this layer or not, for communication compression
        self.block_sparse_indicators = []

        # cache for the previous model
        self.cached_params = []

        # cache for the updates sent from workers
        self.cached_updates = []

        self.validators = []

        for i, param in enumerate(self._params):
            self.block_sparse_indicators.append(mx.nd.array([1]))
            if param.grad_req != 'null':
                byteps_declare_and_init_tensor("block_sparse_indicator_" + str(i), self.block_sparse_indicators[-1])
                byteps_declare_and_init_tensor("parameter_" + str(i), param.list_data()[0])
                # initialize the parameters by pulling from validators
                if rank() != 0:
                    param.list_data()[0][:] = 0
                byteps_push(param.list_data()[0], priority=0, name="parameter_" + str(i))

                byteps_declare_and_init_tensor("gradient_" + str(i), param.list_grad()[0])
                self.cached_updates.append(param.list_grad()[0].copy())

                if self.validation_type == "zenopp":
                    # self.validators.append(ZenoppValidator(eta=-0.1, rho=0.4, alpha=0.2))
                    self.validators.append(ZenoppValidator(eta=-0.02, rho=0.2, alpha=self.alpha))
                    # self.validators.append(ZenoppValidator(eta=-0.01, rho=0.6, alpha=math.sqrt(1./worker_size())))
                    # self.validators.append(ZenoppValidator(eta=-0.001, rho=0.4, alpha=1./worker_size()))
                elif self.validation_type == "naive_async":
                    # self.validators.append(NaiveAsyncValidator(alpha=math.sqrt(1./worker_size())))
                    # self.validators.append(NaiveAsyncValidator(alpha=1./worker_size()))
                    self.validators.append(NaiveAsyncValidator(alpha=self.alpha))
                elif self.validation_type == "fed_async":
                    self.validators.append(FedAsyncValidator(alpha=math.sqrt(1./worker_size())))
                    # self.validators.append(FedAsyncValidator(alpha=1./worker_size()))
                else:
                    raise ValueError('Undefined validation_type: %s' % self.validation_type)
            else:
                self.cached_updates.append(None)
                self.validators.append(None)
            # print("initialized " + str(i))
        
        print("model initialized")
        
        if rank() == 0:
            self.global_timestamp[:] = 1
            byteps_push(self.global_timestamp, priority=-1, name="global_timestamp")
        self.global_timestamp[:] = 0
        while self.global_timestamp.asscalar() == 0:
            print("blocked until the model is initialized on server")
            time.sleep(0.1)
            byteps_pull(self.global_timestamp, priority=0, name="global_timestamp")
        print("current timestamp is %d" % (self.global_timestamp.asscalar()))

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                byteps_pull(param.list_data()[0], priority=0, name="parameter_" + str(i))
                self.cached_params.append(param.list_data()[0].copy())
            else:
                self.cached_params.append(None)
        
        self.zenops_initialized = True
        self.sync_counter = 0
        self.validation_counter = 0.0
        self.recv_counter = 0.0

        mx.nd.waitall()
        print("_init_zenops finished")


    def step(self, batch_size, ignore_stale_grad=False):
        """Makes one step of parameter update. Should be called after
        `autograd.backward()` and outside of `record()` scope.

        For normal parameter updates, `step()` should be used, which internally calls
        `allreduce_grads()` and then `update()`. However, if you need to get the reduced
        gradients to perform certain transformation, such as in gradient clipping, then
        you may want to manually call `allreduce_grads()` and `update()` separately.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        rescale_grad = self._scale / batch_size
        self._check_and_rescale_grad(rescale_grad)

        if not self._kv_initialized:
            self._init_kvstore()
        if self._params_to_init:
            self._init_params()

        if not self.zenops_initialized or "zeno" in self.validation_type.lower():
            self._update(ignore_stale_grad)
            
            # proximal update
            if self.zenops_initialized and self.rho:
                for i, (param, cached_param_data) in enumerate(zip(self._params, self.cached_params)):
                    if param.grad_req != 'null':
                        param.list_data()[0][:] *= (1-self.rho)
                        param.list_data()[0][:] += cached_param_data * self.rho

        self._init_zenops()

        self.sync_counter += 1
        if self.sync_counter == self.sync_interval:
            self.sync_counter = 0

            for i, (param, cached_param_data, cached_update_data) \
                in enumerate(zip(self._params, self.cached_params, self.cached_updates)):
                if param.grad_req != 'null':
                    cached_update_data[:] = param.list_data()[0] - cached_param_data
                    param.list_data()[0][:] = cached_param_data

            mx.nd.waitall()
            # time.sleep(0.01 * (worker_size()+rank()+1))

            # TODO: worker subsampling
            for j in range(worker_size()):
                if j % validator_size() == rank():
                    byteps_pull(self.worker_sparse_indicator, name="worker_sparse_indicator", priority=0)
                    active_worker = self.worker_sparse_indicator[0].asscalar()
                    mx.nd.waitall()
                    # time.sleep(1.0 * random.uniform(0, 1))
                    if active_worker > 0:

                        for i, (param, cached_param_data, cached_update_data, send_layer, validator) \
                            in enumerate(zip(self._params, self.cached_params, self.cached_updates, self.block_sparse_indicators, self.validators)):
                            if param.grad_req != 'null':
                                byteps_pull(send_layer, name="block_sparse_indicator_" + str(i), priority=-i)
                                active_layer_worker = send_layer[0].asscalar()
                                if active_layer_worker > 0:
                                    byteps_pull(param.list_grad()[0], name="gradient_" + str(i), priority=-i)
                            
                                    if self.validation_type == "zenopp":
                                        validation_info = {'validation_tensor': cached_update_data}
                                    elif self.validation_type == "naive_async":
                                        validation_info = None
                                    else:
                                        raise ValueError('Undefined validation_type: %s' % self.validation_type)
                                    if "zeno" in self.validation_type:
                                        self.validation_counter += validator.validate(param.list_grad()[0], param.list_grad()[0], validation_info)
                                        self.recv_counter += 1
                                    else:
                                        validator.validate(param.list_grad()[0], param.list_grad()[0], validation_info)
                                    
                                    param.list_data()[0][:] = param.list_grad()[0]

                                    # after validation is done, push the update to the server
                                    byteps_push(param.list_data()[0], name="parameter_" + str(i), priority=-i)
                        self.global_timestamp[:] = 1
                        byteps_push(self.global_timestamp, priority=-1, name="global_timestamp")
            
                        mx.nd.waitall()
                        time.sleep(0.01)
            
            for i, (param, cached_param_data) \
                in enumerate(zip(self._params, self.cached_params)):
                if param.grad_req != 'null':
                    byteps_pull(param.list_data()[0], name="parameter_" + str(i), priority=-i)
                    cached_param_data[:] = param.list_data()[0]
            mx.nd.waitall()
            # time.sleep(0.01)