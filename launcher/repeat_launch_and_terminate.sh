#!/usr/bin/env bash

for i in `seq 1 1`;
do
   > zeno_ps.log
  taskid=`qsub ./dist_launcher_intel_mpi_1.sh`
  echo "task $taskid is submitted"

  while ! grep -Fq '[Epoch 0]' zeno_ps.log
  do
    sleep 1m
    if grep -Fq 'Signals.SIGABRT' zeno_ps.log
    then
      sleep 5m
      qdel "$taskid"
      sleep 5
      qdel "$taskid"
      sleep 5
      qdel "$taskid"
      sleep 10
      taskid=`qsub ./dist_launcher_intel_mpi_1.sh`
      echo "task $taskid is re-submitted"
    fi
  done
  sleep 5m
  echo "task $taskid is successfully started"
  while ! grep -Fq '[Epoch 199]' zeno_ps.log
  do
    sleep 5m
  done
  sleep 5m
  qdel "$taskid"
  sleep 5
  qdel "$taskid"
  sleep 5
  echo "task $taskid is terminated"
  sleep 1m
done 

# for i in `seq 1 1`;
# do
#    > zeno_ps.log
#   taskid=`qsub ./dist_launcher_intel_mpi_2.sh`
#   echo "task $taskid is submitted"

#   while ! grep -Fq '[Epoch 0]' zeno_ps.log
#   do
#     sleep 1m
#     if grep -Fq 'Signals.SIGABRT' zeno_ps.log
#     then
#       sleep 5m
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 10
#       taskid=`qsub ./dist_launcher_intel_mpi_2.sh`
#       echo "task $taskid is re-submitted"
#     fi
#   done
#   sleep 5m
#   echo "task $taskid is successfully started"
#   while ! grep -Fq '[Epoch 199]' zeno_ps.log
#   do
#     sleep 5m
#   done
#   sleep 5m
#   qdel "$taskid"
#   sleep 5
#   qdel "$taskid"
#   sleep 5
#   echo "task $taskid is terminated"
#   sleep 1m
# done 

# for i in `seq 1 1`;
# do
#    > zeno_ps.log
#   taskid=`qsub ./dist_launcher_intel_mpi_3.sh`
#   echo "task $taskid is submitted"

#   while ! grep -Fq '[Epoch 0]' zeno_ps.log
#   do
#     sleep 1m
#     if grep -Fq 'Signals.SIGABRT' zeno_ps.log
#     then
#       sleep 5m
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 10
#       taskid=`qsub ./dist_launcher_intel_mpi_3.sh`
#       echo "task $taskid is re-submitted"
#     fi
#   done
#   sleep 5m
#   echo "task $taskid is successfully started"
#   while ! grep -Fq '[Epoch 199]' zeno_ps.log
#   do
#     sleep 5m
#   done
#   sleep 5m
#   qdel "$taskid"
#   sleep 5
#   qdel "$taskid"
#   sleep 5
#   echo "task $taskid is terminated"
#   sleep 1m
# done 

# for i in `seq 1 1`;
# do
#    > zeno_ps.log
#   taskid=`qsub ./dist_launcher_intel_mpi_4.sh`
#   echo "task $taskid is submitted"

#   while ! grep -Fq '[Epoch 0]' zeno_ps.log
#   do
#     sleep 1m
#     if grep -Fq 'Signals.SIGABRT' zeno_ps.log
#     then
#       sleep 5m
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 10
#       taskid=`qsub ./dist_launcher_intel_mpi_4.sh`
#       echo "task $taskid is re-submitted"
#     fi
#   done
#   sleep 5m
#   echo "task $taskid is successfully started"
#   while ! grep -Fq '[Epoch 199]' zeno_ps.log
#   do
#     sleep 5m
#   done
#   sleep 5m
#   qdel "$taskid"
#   sleep 5
#   qdel "$taskid"
#   sleep 5
#   echo "task $taskid is terminated"
#   sleep 1m
# done 

# for i in `seq 1 1`;
# do
#    > zeno_ps.log
#   taskid=`qsub ./dist_launcher_intel_mpi_5.sh`
#   echo "task $taskid is submitted"

#   while ! grep -Fq '[Epoch 0]' zeno_ps.log
#   do
#     sleep 1m
#     if grep -Fq 'Signals.SIGABRT' zeno_ps.log
#     then
#       sleep 5m
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 10
#       taskid=`qsub ./dist_launcher_intel_mpi_5.sh`
#       echo "task $taskid is re-submitted"
#     fi
#   done
#   sleep 5m
#   echo "task $taskid is successfully started"
#   while ! grep -Fq '[Epoch 199]' zeno_ps.log
#   do
#     sleep 5m
#   done
#   sleep 5m
#   qdel "$taskid"
#   sleep 5
#   qdel "$taskid"
#   sleep 5
#   echo "task $taskid is terminated"
#   sleep 1m
# done 


# for i in `seq 1 1`;
# do
#    > zeno_ps.log
#   taskid=`qsub ./dist_launcher_intel_mpi_6.sh`
#   echo "task $taskid is submitted"

#   while ! grep -Fq '[Epoch 0]' zeno_ps.log
#   do
#     sleep 1m
#     if grep -Fq 'Signals.SIGABRT' zeno_ps.log
#     then
#       sleep 5m
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 10
#       taskid=`qsub ./dist_launcher_intel_mpi_6.sh`
#       echo "task $taskid is re-submitted"
#     fi
#   done
#   sleep 5m
#   echo "task $taskid is successfully started"
#   while ! grep -Fq '[Epoch 199]' zeno_ps.log
#   do
#     sleep 5m
#   done
#   sleep 5m
#   qdel "$taskid"
#   sleep 5
#   qdel "$taskid"
#   sleep 5
#   echo "task $taskid is terminated"
#   sleep 1m
# done 



# for i in `seq 1 1`;
# do
#    > zeno_ps.log
#   taskid=`qsub ./dist_launcher_intel_mpi_7.sh`
#   echo "task $taskid is submitted"

#   while ! grep -Fq '[Epoch 0]' zeno_ps.log
#   do
#     sleep 1m
#     if grep -Fq 'Signals.SIGABRT' zeno_ps.log
#     then
#       sleep 5m
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 5
#       qdel "$taskid"
#       sleep 10
#       taskid=`qsub ./dist_launcher_intel_mpi_7.sh`
#       echo "task $taskid is re-submitted"
#     fi
#   done
#   sleep 5m
#   echo "task $taskid is successfully started"
#   while ! grep -Fq '[Epoch 199]' zeno_ps.log
#   do
#     sleep 5m
#   done
#   sleep 5m
#   qdel "$taskid"
#   sleep 5
#   qdel "$taskid"
#   sleep 5
#   echo "task $taskid is terminated"
#   sleep 1m
# done 