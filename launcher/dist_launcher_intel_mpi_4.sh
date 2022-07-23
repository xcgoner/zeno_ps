#!/usr/bin/env bash
#PBS -l select=18:ncpus=56
module load intel
### OPA FABRIC ###
export I_MPI_FABRICS=shm:tmi
export I_MPI_TMI_PROVIDER=psm2
export HFI_NO_CPUAFFINITY=1
export I_MPI_FALLBACK=0
export OMP_NUM_THREADS=11
export KMP_AFFINITY=granularity=fine,compact,1,0;
### OPA FABRIC ###
export I_MPI_HYDRA_BOOTSTRAP=rsh
export I_MPI_HYDRA_BOOTSTRAP_EXEC=pbs_tmrsh

watchfile=$PBS_O_WORKDIR/zeno_ps.log

hostfile=$PBS_O_WORKDIR/hostfile
> $hostfile

input=`echo $PBS_NODEFILE`
while IFS= read -r line
do
  dig +short "$line" | awk '{ print ; exit }' >> $hostfile
done < "$input"
cat $hostfile

configfile=$PBS_O_WORKDIR/mpi_configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 3 --num-validators 1 --scheduler-port 1234 \
#     --server-command "echo server" \
#     --worker-command 'echo worker' \
#     --validator-command 'echo validator' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zenopp --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file nobyz_zenopp_async' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type naive_async --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file nobyz_naive_async' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type negative --byz-scale 200 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zenopp --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file neg200_zenopp' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type negative --byz-scale 200 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type naive_async --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file neg200_naive' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type negative --byz-scale 2 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zenopp --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file neg2_zenopp' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type negative --byz-scale 2 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type naive_async --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file neg2_naive' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type random --byz-scale 1 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zenopp --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file random1_zenopp' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type random --byz-scale 1 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type naive_async --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file random1_naive' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type randomscale --byz-scale 8 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type naive_async --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file randomscale8_naive' \
#     --sync-mode async \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 8 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 400 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.7 --sync-mode async --byz-type randomscale --byz-scale 8 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 64 --wd 0.0001 --lr 0.4 --lr-decay 0.2 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zenopp --sync-interval 8 --sync-mode async --alpha 0.4 --alpha-decay 0.5 --alpha-decay-epoch 100,150 --logging-file randomscale8_zenopp' \
#     --sync-mode async \
#     -o $configfile

# sync

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --logging-file nobyz_average' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type trimmed_mean --sync-interval 8 --logging-file nobyz_trimmed_0.5' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file nobyz_phocas_0.5' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zeno --sync-interval 8 --logging-file nobyz_zeno_0.5' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5 --byz-type negative --byz-scale 6 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file neg6_phocas_0.5' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5 --byz-type random --byz-scale 1 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file random1_phocas_0.5' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 0.5 --byz-type randomscale --byz-scale 8 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file randomscale8_phocas_0.5' \
#     -o $configfile

# sync 2 validators
# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 2 --num-workers 16 --num-validators 2 --scheduler-port 1234 --server-validator-colocated \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 1.0' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file nobyz_phocas_1.0_val2' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 2 --num-workers 16 --num-validators 2 --scheduler-port 1234 --server-validator-colocated \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 1.0 --byz-type negative --byz-scale 6 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file neg6_phocas_1.0_val2' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 2 --num-workers 16 --num-validators 2 --scheduler-port 1234 --server-validator-colocated \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 1.0 --byz-type random --byz-scale 1 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file random1_phocas_1.0_val2' \
#     -o $configfile

# python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 2 --num-workers 16 --num-validators 2 --scheduler-port 1234 --server-validator-colocated \
#     --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
#     --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 1.0 --byz-type randomscale --byz-scale 8 --byz-rate 0.2' \
#     --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type phocas --sync-interval 8 --logging-file randomscale8_phocas_1.0_val2' \
#     -o $configfile

# tune eta
python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 2 --num-workers 16 --num-validators 2 --scheduler-port 1234 --server-validator-colocated \
    --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
    --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 8 --sparse-rate 0.2 --worker-subsample-rate 1.0 --byz-type random --byz-scale 1 --byz-rate 0.2' \
    --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type zeno --sync-interval 8 --zeno-eta 0.001 --logging-file random1_zeno_1.0_val2_eta_0.001' \
    -o $configfile

# echo -------------
# cat $configfile

mpirun -configfile $configfile 2>&1 | tee $watchfile