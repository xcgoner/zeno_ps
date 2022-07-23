#!/usr/bin/python
"""
Launch a distributed job for BytePS
"""
import argparse
import os, sys, time
import signal
import logging
import subprocess
from multiprocessing import Pool, Process
from threading import Thread

def preprocess_envs(args_envs):
    envs_map = {}
    for item in args_envs:
        i = item.find(":")
        if i != -1:
            key = item[:i]
            val = item[i+1:]
        envs_map[key] = val
    return envs_map

def get_env(envs_map):
    envs = []
    # get system envs
    keys = ['OMP_NUM_THREADS', 'KMP_AFFINITY']
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            envs.append('-x ' + k + '=' + v)
    # get ass_envs
    for k, v in envs_map.items():
        envs.append('-x ' + str(k) + '=' + str(v))
    return (' '.join(envs))

def get_hosts_from_file(filename):
    with open(filename) as f:
        tmp = f.readlines()
    assert len(tmp) > 0
    hosts = []
    for h in tmp:
        if len(h.strip()) > 0:
            # parse addresses of the form ip:port:gpu_id
            cols = h[:-1].split(':')
            host = cols[0]
            gpu_id = "-1" if len(cols) <= 1 or cols[1] == "" else cols[1]
            # hosts now contain the pair ip, gpu_id
            hosts.append((host, gpu_id))
    # print(hosts)
    return hosts


def start_mpi(prog, envs, node, fname):
    def run(prog):
        subprocess.check_call(prog, shell=True)

    dirname = 'mpilog'
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    pname = dirname + '/' + fname
    prog = 'mpirun -n 1 --bind-to none -H ' + node + ' ' + envs + ' ' + prog \
            + ' >> ' + pname + '.stdout' + ' 2>>' + pname + '.stderr&'
    print(prog)

    thread = Thread(target=run, args=(prog,))
    thread.setDaemon(True)
    thread.start()
    return thread


def submit(args):
    num_workers = args.num_workers
    num_validators = args.num_validators
    num_servers = args.num_servers
    assert num_workers >= 1
    assert num_validators >= 1
    assert num_servers >= 1
    print('Launch %d workers, %d validators, and %d servers' % (num_workers, num_validators, num_servers))

    hosts = get_hosts_from_file(args.hostfile)
    worker_hosts = hosts[:num_workers]
    validator_hosts = hosts[num_workers:(num_workers+num_validators)]
    server_hosts = hosts[(num_workers+num_validators):(num_servers+num_workers+num_validators)]
    print("servers:")
    print(server_hosts)
    print("workers:")
    print(worker_hosts)
    print("validators:")
    print(validator_hosts)

    scheduler_ip = server_hosts[0][0]

    # common env
    pass_envs = preprocess_envs(args.env)
    pass_envs['DMLC_NUM_WORKER'] = str(num_workers)
    pass_envs['DMLC_NUM_VALIDATOR'] = str(num_validators)
    pass_envs['DMLC_NUM_SERVER'] = str(num_servers)
    # pass_envs['DMLC_INTERFACE'] = str(args.interface)
    pass_envs['DMLC_PS_ROOT_URI'] = str(scheduler_ip)
    pass_envs['DMLC_PS_ROOT_PORT'] = str(args.scheduler_port)
    pass_envs['BYTEPS_FORCE_DISTRIBUTED'] = str(1)

    # debug
    pass_envs['PS_VERBOSE'] = str(0)
    pass_envs['BYTEPS_LOG_LEVEL'] = 'INFO'

    threads = []
    name = 'scheduler'
    pass_envs['DMLC_ROLE'] = name
    prog = args.server_command
    envs = get_env(pass_envs)
    threads.append(start_mpi(prog, envs, scheduler_ip, name))
    time.sleep(0.3)
    for i, (node, _) in enumerate(server_hosts):
        name = 'server'
        pass_envs['DMLC_ROLE'] = name
        if args.sync_mode == "async":
            pass_envs['BYTEPS_ENABLE_ASYNC'] = 1
        prog = args.server_command
        envs = get_env(pass_envs)
        threads.append(start_mpi(prog, envs, node, name + str(i)))
        time.sleep(0.3)
    for i, (node, gpu_id) in enumerate(worker_hosts):
        name = 'worker'
        pass_envs['DMLC_ROLE'] = "worker"
        pass_envs['DMLC_WORKER_ID'] = str(i)
        pass_envs['DMLC_WORKER_TYPE'] = name
        if gpu_id != "-1":
            pass_envs['NVIDIA_VISIBLE_DEVICES'] = str(gpu_id)
        prog = args.worker_command
        envs = get_env(pass_envs)
        threads.append(start_mpi(prog, envs, node, name + str(i)))
        time.sleep(0.3)
    del pass_envs['DMLC_WORKER_ID']
    for i, (node, gpu_id) in enumerate(validator_hosts):
        name = 'validator'
        pass_envs['DMLC_ROLE'] = "worker"
        pass_envs['DMLC_VALIDATOR_ID'] = str(i)
        pass_envs['DMLC_WORKER_TYPE'] = name
        if gpu_id != "-1":
            pass_envs['NVIDIA_VISIBLE_DEVICES'] = str(gpu_id)
        prog = args.validator_command
        envs = get_env(pass_envs)
        threads.append(start_mpi(prog, envs, node, name + str(i)))
        time.sleep(0.3)

    for t in threads:
        t.join()


def main():
    parser = argparse.ArgumentParser(description='Launch a distributed training job for BytePS')
    parser.add_argument('--num-workers', required=True, type=int,
                        help = 'the number of workers.')
    parser.add_argument('--num-servers', required=True, type=int,
                        help = 'the number of servers.')
    parser.add_argument('--num-validators', required=True, type=int,
                        help = 'the number of validators.')
    parser.add_argument('-H', '--hostfile', required=True, type=str,
                        help = 'the hostfile of machines which will run the job.')
    parser.add_argument('--scheduler-port', required=True, type=int,
                        help = 'the port of the scheduler')
    parser.add_argument('--interface', type=str, default='eth0',
                        help = 'the network interface to use')
    parser.add_argument('--env', action='append', default=[],
                        help = 'Given a pair of environment_variable:value, sets this value of \
                        environment variable for all workers and servers. Example OMP_NUM_THREADS:3')
    parser.add_argument('--server-command', type=str, required=True,
                        help = 'command for servers')
    parser.add_argument('--worker-command', type=str, required=True,
                        help = 'command for workers')
    parser.add_argument('--validator-command', type=str, required=True,
                        help = 'command for validators')
    parser.add_argument('--sync-mode', type=str, default='sync',
                        help = 'sync or async')

    args = parser.parse_args()

    # check necessary args
    assert args.hostfile
    assert args.scheduler_port

    submit(args)


def signal_handler(signal, frame):
    logging.info('Stop launcher')
    sys.exit(0)

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()