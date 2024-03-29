import os
import sys
import time

import torch
from torch import multiprocessing

from federatedscope.contrib.trainer.laplacian_trainer import call_laplacian_trainer
from federatedscope.contrib.workers.laplacian_client import LaplacianClient
from federatedscope.contrib.workers.laplacian_server import LaplacianServer
from federatedscope.register import register_trainer

# sys.path = ['~/Master-Thesis/CKIM_Competition/federatedscope',
# '~/Master-Thesis/CKIM_Competition',] + sys.path
sys.path = ['/home/michael/Projects/CKIM_Competition/federatedscope', '/home/michael/Projects/CKIM_Competition',] + sys.path

print(sys.path)
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.core.fed_runner import FedRunner
from yacs.config import CfgNode

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

register_trainer('laplacian_trainer', call_laplacian_trainer)

# model: fraph_level_default_double_params
def train(lr, csd_imp):
    cfg_file = 'scripts/B-FHTL_exp_scripts/Graph-DC/fedFOLA.yaml'
    cfg_client = 'scripts/B-FHTL_exp_scripts/Graph-DC/cfg_per_client.yaml'
    # cfg_per_Client_ours_lr
    # cfg_per_client_ours_lr_local_steps

    #'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(cfg_file)

    # init_cfg.data.subdirectory = 'graph_dt_backup/processed'
    # init_cfg.merge_from_list(args.opts)
    init_cfg.data.save_dir = 'Graph-DC_FedFOLA_two_branches_lr_' + str(lr).replace('.',
                                                                        '_') + '_local_update_steps_1_csd_imp_' + str(csd_imp).replace('.', '_')
    init_cfg.train.optimizer.lr = lr

    init_cfg.params = CN()
    init_cfg.params.eps = 1e-15
    init_cfg.params.csd_importance = csd_imp
    init_cfg.params.p = 0.
    init_cfg.params.alpha = 0.1
    init_cfg.params.save_client_always = True
    init_cfg.federate.client_num = 13

    init_cfg.model.dropout = 0.5
    update_logger(init_cfg)

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global cfg object
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    # allow different settings for different clients
    # cfg_client.merge_from_file(args.cfg_client)
    if cfg_client is None:
        cfg_client = None
    else:
        cfg_client = CfgNode.load_cfg(open(cfg_client, 'r')).clone()
    runner = FedRunner(data=data,
                   server_class=LaplacianServer,
                   client_class=LaplacianClient,
                   config=init_cfg.clone(),
                   client_config=cfg_client)
    _ = runner.run()


if __name__ == '__main__':
    num_trainings = 7
    csd_imps = [10]
    # lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lrs = [0.05]
    pool = multiprocessing.Pool(7)
    processes = []
    for lr in lrs:
        for csd_imp in csd_imps:
            for i in range(num_trainings):
                time.sleep(10)
                setup_seed(i+3)
                processes.append(pool.apply_async(train, args=(lr, csd_imp)))
    result = [p.get() for p in processes]
