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
import math
import numpy as np

import gluoncv as gcv
import matplotlib
import mxnet as mx
from gluoncv.data import transforms as gcv_transforms
from gluoncv.model_zoo import get_model
from gluoncv.utils import LRScheduler, LRSequential, makedirs
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import byteps.mxnet as bps
from byteps.mxnet.ops import size, local_size, rank, local_rank, worker_size, validator_size

matplotlib.use('Agg')


gcv.utils.check_version('0.6.0')


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for image classification.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--model', type=str, default='cifar_resnet20_v1',
                        help='model to use. options are resnet and wrn. default is cifar_resnet20_v1.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=200,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--optimizer', type=str, default='nag',
                        help='optimization algorithm')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='period in epoch for learning rate decays. default is 0 (has no effect).')
    parser.add_argument('--lr-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 100,150.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--drop-rate', type=float, default=0.0,
                        help='dropout rate for wide resnet. default is 0.')
    parser.add_argument('--mode', type=str, default='hybrid', 
                        help='mode in which to train the model. options are imperative, hybrid')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-from', type=str,
                        help='resume training from the model')
    parser.add_argument('--logging-file', type=str, default='baseline',
                        help='name of training log file')
    # options for zeno ps
    parser.add_argument('--sync-mode', type=str, default='sync', choices=["sync", "async"], 
                        help='sync or async')
    parser.add_argument('--sync-interval', type=int, default=1,
                        help='number of local steps.')
    parser.add_argument('--validation-type', type=str, default='average',
                        help='method for validator')
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='the mixing parameter. default is 0.8.')
    parser.add_argument('--alpha-decay', type=float, default=0.5,
                        help='decay rate of the mixing parameter. default is 0.1.')
    parser.add_argument('--alpha-decay-epoch', type=str, default='100,150',
                        help='epochs at which learning rate decays. default is 100,150.')
    parser.add_argument('--zeno-eta', type=float, default=-0.001,
                        help='eta of zeno, default is -0.001')
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()

    bps.init()

    if opt.num_gpus > 0:
        gpu_name = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
        gpu_name = gpu_name.decode('utf8').split('\n')[-2]
        gpu_name = '-'.join(gpu_name.split())
        device_name = gpu_name
    else: device_name = "cpu"
    filename = "cifar10-validator-%d-%s-%s.log" % (bps.rank(),
                                          device_name, opt.logging_file)
    filehandler = logging.FileHandler(filename)
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.info(opt)

    batch_size = opt.batch_size
    classes = 10

    num_gpus = opt.num_gpus
    # batch_size *= max(1, num_gpus)
    # context = mx.gpu(bps.local_rank()) if num_gpus > 0 else mx.cpu(
    #     bps.local_rank())
    ctx_idx = int(os.environ.get('NVIDIA_VISIBLE_DEVICES', '0'))
    context = mx.gpu(ctx_idx) if num_gpus > 0 else mx.cpu()
    num_workers = opt.num_workers
    nworker = worker_size()
    rank = bps.rank()

    lr_decay = opt.lr_decay
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]
    alpha_decay_epoch = [int(i) for i in opt.alpha_decay_epoch.split(',')] + [np.inf]

    # num_batches = 50000 // (opt.batch_size * nworker)
    # # base_lr = opt.lr * nworker / bps.local_size()
    # base_lr = opt.lr
    # lr_scheduler = LRSequential([
    #     LRScheduler('linear', base_lr=opt.warmup_lr,
    #                 target_lr=base_lr,
    #                 nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
    #     LRScheduler('step', base_lr=base_lr,
    #                 target_lr=0,
    #                 nepochs=opt.num_epochs - opt.warmup_epochs,
    #                 iters_per_epoch=num_batches,
    #                 step_epoch=lr_decay_epoch,
    #                 step_factor=lr_decay, power=2)
    # ])

    model_name = opt.model
    if model_name.startswith('cifar_wideresnet'):
        kwargs = {'classes': classes,
                  'drop_rate': opt.drop_rate}
    else:
        kwargs = {'classes': classes}
    net = get_model(model_name, **kwargs)
    if opt.resume_from:
        net.load_parameters(opt.resume_from, ctx=context)

    save_period = opt.save_period
    if opt.save_dir and save_period:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_period = 0

    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4),
        transforms.RandomFlipLeftRight(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    def test(ctx, test_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(test_data):
            data = gluon.utils.split_and_load(
                batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(
                batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

    def train(epochs, ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        net.initialize(mx.init.Xavier(), ctx=ctx)

        train_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=True).shard(
                bps.worker_size()+1, rank).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard',
            num_workers=num_workers)
        
        val_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=True).shard(
                bps.worker_size()+1, bps.worker_size()).transform_first(transform_train),
            batch_size=batch_size, shuffle=True, last_batch='discard',
            num_workers=num_workers)


        test_data = gluon.data.DataLoader(
            gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
            batch_size=128, shuffle=False, num_workers=num_workers)

        params = net.collect_params()

        optimizer_params = {'learning_rate': opt.lr,
                            'wd': opt.wd, 'momentum': opt.momentum}

        if opt.sync_mode == "sync":
            trainer = bps.DistributedZenoValidatorSyncTrainer(params,
                                            opt.optimizer,
                                            optimizer_params, 
                                            validation_type = opt.validation_type, 
                                            sync_interval = opt.sync_interval,
                                            zeno_eta = opt.zeno_eta)
        else:
            optimizer_params['learning_rate'] *= math.sqrt(1./(worker_size()+1))
            # optimizer_params['learning_rate'] /= (worker_size()+1)
            trainer = bps.DistributedZenoValidatorAsyncTrainer(params,
                                            opt.optimizer,
                                            optimizer_params, 
                                            rho = 0.2, 
                                            alpha = opt.alpha, 
                                            validation_type = opt.validation_type, 
                                            sync_interval = opt.sync_interval)
        metric = mx.metric.Accuracy()
        train_metric = mx.metric.Accuracy()
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

        iteration = 0
        lr_decay_count = 0
        alpha_decay_count = 0
        best_val_score = 0
        # bps.byteps_declare_tensor("acc")
        logger.info('Validator %d, started' %(rank))
        for epoch in range(epochs):
            tic = time.time()
            train_metric.reset()
            metric.reset()
            train_loss = 0
            num_batch = len(val_data)

            # if epoch == lr_decay_epoch[lr_decay_count]:
            #     if opt.sync_mode == "sync":
            #         trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            #         lr_decay_count += 1
            #     elif opt.sync_mode == "async":
            #         for validator in trainer.validators:
            #             if validator:
            #                 validator.alpha *= lr_decay

            if epoch == lr_decay_epoch[lr_decay_count]:
                trainer.set_learning_rate(trainer.learning_rate*lr_decay)
                lr_decay_count += 1
            if opt.sync_mode == "async":
                if epoch == alpha_decay_epoch[alpha_decay_count]:
                    for validator in trainer.validators:
                        if validator:
                            validator.alpha *= opt.alpha_decay
                    logger.info('[Epoch %d] validation alpha decayed: %f' % (epoch, trainer.validators[0].alpha))   
                    lr_decay_count += 1

            for i, batch in enumerate(val_data):
                # if first_batch or opt.validation_type == "zeno++":
                #     first_batch = False
                data = gluon.utils.split_and_load(
                    batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(
                    batch[1], ctx_list=ctx, batch_axis=0)

                with ag.record():
                    output = [net(X) for X in data]
                    loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
                for l in loss:
                    l.backward()
                trainer.step(batch_size)
                # train_loss += sum([l.sum().asscalar() for l in loss])

                # train_metric.update(label, output)
                # name, train_acc = train_metric.get()
                iteration += 1

                # mx.nd.waitall()
                # print("iteration %d finished" % (iteration))

            # train_loss /= batch_size * num_batch
            # name, train_acc = train_metric.get()
            if rank == 0:
                throughput = int(batch_size * nworker * i / (time.time() - tic))

                logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f lr=%f' %
                            (epoch, throughput, time.time()-tic, trainer.learning_rate))

                name, test_acc = test(ctx, test_data)
                # name, val_acc = test(ctx, val_data)
                # acc = mx.nd.array([train_acc, val_acc, test_acc], ctx=ctx[0])
                # bps.byteps_push_pull(acc, name="acc", is_average=False)
                # acc /= bps.size()
                # train_acc, val_acc = acc[0].asscalar(), acc[1].asscalar()
                # logger.info('[Epoch %d] training: %s=%f' %
                #             (epoch, name, train_acc))
                # logger.info('[Epoch %d] validation: %s=%f' %
                #             (epoch, name, val_acc))
                logger.info('[Epoch %d] test: %s=%f' %
                            (epoch, name, test_acc))
                if "zeno" in opt.validation_type:
                    logger.info('[Epoch %d] validation rate: %f' % (epoch, trainer.validation_counter / trainer.recv_counter))    

        #     if val_acc > best_val_score:
        #         best_val_score = val_acc
        #         net.save_parameters('%s/%.4f-cifar-%s-%d-best.params' %
        #                             (save_dir, best_val_score, model_name,
        #                              epoch))

        #     if save_period and save_dir and (epoch + 1) % save_period == 0:
        #         net.save_parameters('%s/cifar10-%s-%d.params' %
        #                             (save_dir, model_name, epoch))

        # if save_period and save_dir:
        #     net.save_parameters('%s/cifar10-%s-%d.params' %
        #                         (save_dir, model_name, epochs-1))

    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)


if __name__ == '__main__':
    main()
