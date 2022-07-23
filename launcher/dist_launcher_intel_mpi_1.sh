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

# sgd baseline
# 1 server and 1 validator
python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 1 --num-workers 16 --num-validators 1 --scheduler-port 1234 \
    --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
    --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 1 --sparse-rate 0.0 --worker-subsample-rate 1.0' \
    --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 1 --logging-file nobyz_sgd_baseline' \
    -o $configfile

# 2 servers and 2 validators
python /homes/cx2/src/zeno_ps/byteps/launcher/configfile_zeno_intel_mpi.py --hostfile $hostfile --num-servers 2 --num-workers 16 --num-validators 2 --scheduler-port 1234 --server-validator-colocated \
    --server-command "python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py" \
    --worker-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_worker.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 1 --sparse-rate 0.0 --worker-subsample-rate 1.0' \
    --validator-command 'python /homes/cx2/src/zeno_ps/byteps/launcher/launch.py python /homes/cx2/src/zeno_ps/byteps/example/mxnet/train_cifar10_byteps_zeno_validator.py --num-epochs 200 --mode hybrid --num-gpus 0 -j 2 --batch-size 32 --wd 0.0001 --lr 0.2 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet20_v1 --validation-type average --sync-interval 1 --logging-file nobyz_sgd_baseline' \
    -o $configfile


# echo -------------
# cat $configfile

mpirun -configfile $configfile 2>&1 | tee $watchfile