# Zeno-PS Instructions

## MXNet

## Setup

```
mkdir -p ~/src
cd ~/src
git clone https://github.com/01org/mkl-dnn.git
cd mkl-dnn
mkdir -p build && cd build && cmake .. && make -j64
make test
sudo make install

pip uninstall mxnet-cu100 -y
pip install --pre --upgrade 'mxnet-cu100==1.7.0b20200728' -f https://dist.mxnet.io/python --user
pip3 install gluoncv
pip3 uninstall byteps -y
mkdir -p ~/src/zeno_ps
cd ~/src/zeno_ps
rm -rf byteps
git clone https://github.com/xcgoner/byteps.git -b zeno_ps --recursive
cd byteps/
python3 setup.py install --user
```

## Distributed Training (TCP)

Let's say you have two workers (1 GPU per worker), and one validator (1 GPU per worker). For simplicity we use one server. In practice, you need more servers (at least equal to the number of workers) to achieve high performance.


For the workers, you need to pay attention to `DMLC_WORKER_ID`. 

Also note that set `DMLC_WORKER_TYPE=worker` for workers, and `DMLC_WORKER_TYPE=validator` for validators. 
Set `DMLC_WORKER_ID` for workers, and `DMLC_VALIDATOR_ID` for validators.
Both of them nees `DMLC_ROLE=worker`.

------------

### Naive (simple average) validation with 2 workers:

For the scheduler:
```
export DMLC_NUM_WORKER=2
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 launch.py
```

For the server:
```
export DMLC_NUM_WORKER=2
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export BYTEPS_SERVER_ENGINE_THREAD=$((vCPUs / 2))

python3 launch.py
```


For worker-0:
```
export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=2
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/test_zenops_worker.py

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 64 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 1
```

For worker-1:

```
export NVIDIA_VISIBLE_DEVICES=1
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=2
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/test_zenops_worker.py

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 64 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 1
```


For validator-0:

```
export NVIDIA_VISIBLE_DEVICES=2
export DMLC_VALIDATOR_ID=0
export DMLC_NUM_WORKER=2
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=validator
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/test_zenops_validator.py

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 64 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 1
```

------------

### Trimmed mean validation with 3 workers:

For the scheduler:
```
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 launch.py
```

For the server:
```
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export BYTEPS_SERVER_ENGINE_THREAD=$((vCPUs / 2))

python3 launch.py
```


For worker-0:
```
export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 16 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 16 --byz-type negative --byz-scale 200
```

For worker-1:

```
export NVIDIA_VISIBLE_DEVICES=1
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 16 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 16
```

For worker-2:

```
export NVIDIA_VISIBLE_DEVICES=2
export DMLC_WORKER_ID=2
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 16 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 16
```


For validator-0:

```
export NVIDIA_VISIBLE_DEVICES=3
export DMLC_VALIDATOR_ID=0
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=validator
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zeno --sync-interval 16

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 16
```


------------

### Asynchronous training:

```
export BYTEPS_ENABLE_ASYNC=1
```

For the scheduler:
```
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 launch.py
```

For the server:
```
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1
export BYTEPS_ENABLE_ASYNC=1

export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export BYTEPS_SERVER_ENGINE_THREAD=$((vCPUs / 2))

python3 launch.py
```


For worker-0:
```
export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --sync-interval 16 --sync-mode async --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2
```

For worker-1:

```
export NVIDIA_VISIBLE_DEVICES=1
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --sync-interval 16 --sync-mode async --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2
```

For worker-2:

```
export NVIDIA_VISIBLE_DEVICES=2
export DMLC_WORKER_ID=2
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --sync-interval 16 --sync-mode async --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2
```


For validator-0:

```
export NVIDIA_VISIBLE_DEVICES=3
export DMLC_VALIDATOR_ID=0
export DMLC_NUM_WORKER=3
export DMLC_NUM_VALIDATOR=1
export DMLC_ROLE=worker
export DMLC_WORKER_TYPE=validator
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=172.31.48.15
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type naive_async --sync-interval 16 --sync-mode async

python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 42 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zenopp --sync-interval 16 --sync-mode async
```

----------
### Synchronous experiments:

```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 172.31.48.15 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 20 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 20'
```

batch size 16

```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32'
```

batch size 32
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 20 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 20'
```

-------------------
15 workers

batch size 16

sync

non-byz

average
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --logging-file nobyz_average'
```

trimmed_mean
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 32 --logging-file nobyz_trimmed'
```

phocas
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 210 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type phocas --sync-interval 32 --logging-file nobyz_phocas'
```

zeno
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 210 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zeno --sync-interval 32 --logging-file nobyz_zeno'
```

-------------------
15 workers

batch size 16

sync

negative 200

average
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --logging-file neg200_average'
```

trimmed_mean
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 32 --logging-file neg200_trimmed'
```

phocas
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 210 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type phocas --sync-interval 32 --logging-file neg200_phocas'
```

zeno
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 210 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 200 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zeno --sync-interval 32 --logging-file neg200_zeno'
```

-------------------
15 workers

batch size 16

sync

negative 6

average
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 6 --byz-rate 0.1' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --logging-file neg6_average'
```

trimmed_mean
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 6 --byz-rate 0.1' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 32 --logging-file neg6_trimmed'
```

phocas
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 6 --byz-rate 0.1' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type phocas --sync-interval 32 --logging-file neg6_phocas'
```

zeno
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type negative --byz-scale 6 --byz-rate 0.1' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zeno --sync-interval 32 --logging-file neg6_zeno'
```

-------------------
15 workers

batch size 16

sync

random 10

average
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 10 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --logging-file random10_average'
```

trimmed_mean
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 10 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 32 --logging-file random10_trimmed'
```

phocas
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 10 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type phocas --sync-interval 32 --logging-file random10_phocas'
```

zeno
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 10 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zeno --sync-interval 32 --logging-file random10_zeno'
```


-------------------
15 workers

batch size 16

sync

random 1

average
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 1 --byz-rate 0.5' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --logging-file random1_average'
```

trimmed_mean
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 1 --byz-rate 0.5' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type trimmed_mean --sync-interval 32 --logging-file random1_trimmed'
```

phocas
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 1 --byz-rate 0.5' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type phocas --sync-interval 32 --logging-file random1_phocas'
```

zeno
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 220 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --byz-type random --byz-scale 1 --byz-rate 0.5' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zeno --sync-interval 32 --logging-file random1_zeno'
```

### Asynchronous experiments:

15 workers

batch size 16

async

non-byz

naive_async

```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 10.0.0.6 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --worker-subsample-rate 0.5 --sync-mode async' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type naive_async --sync-interval 32 --sync-mode async --alpha-decay 1.0 --logging-file nobyz_naive_async' --sync-mode async
```

zenopp
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 10.0.0.6 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zenopp --sync-interval 32 --sync-mode async --alpha 0.5 --alpha-decay 0.4 --alpha-decay-epoch 100 --logging-file nobyz_zenopp_async' --sync-mode async
```

----------
negative 200

naive_async
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type negative --byz-scale 200 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type naive_async --sync-interval 32 --sync-mode async --alpha-decay 1.0 --logging-file neg200_naive' --sync-mode async
```

zenopp
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type negative --byz-scale 200 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zenopp --sync-interval 32 --sync-mode async --alpha-decay 1.0 --logging-file neg200_zenopp' --sync-mode async
```

----------
negative 2

naive_async
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type negative --byz-scale 2 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type naive_async --sync-interval 32 --sync-mode async --alpha-decay 1.0' --sync-mode async
```

zenopp
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type negative --byz-scale 2 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zenopp --sync-interval 32 --sync-mode async --alpha-decay 1.0' --sync-mode async
```


----------
random 10

naive_async
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type random --byz-scale 10 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type naive_async --sync-interval 32 --sync-mode async --alpha-decay 1.0' --sync-mode async
```

zenopp
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type random --byz-scale 10 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zenopp --sync-interval 32 --sync-mode async --alpha-decay 1.0' --sync-mode async
```

----------
random 1

naive_async
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type random --byz-scale 1 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type naive_async --sync-interval 32 --sync-mode async --alpha-decay 1.0' --sync-mode async
```

zenopp
```
python3 dist_launcher_zeno.py --worker-hostfile worker-hostfile --validator-hostfile validator-hostfile --server-hostfile server-hostfile --scheduler-ip 127.0.0.1 --scheduler-port 1234 --server-command "python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py" --worker-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 230 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type average --sync-interval 32 --sparse-rate 0.2 --sync-mode async --byz-type random --byz-scale 1 --byz-rate 0.2' --validator-command 'python3 /home/ubuntu/src/zeno_ps/byteps/launcher/launch.py python3 /home/ubuntu/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 16 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1 --validation-type zenopp --sync-interval 32 --sync-mode async --alpha-decay 1.0' --sync-mode async
```

