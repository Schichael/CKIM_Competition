import os
import sys

sys.path = ['/home/ms234795/Master Thesis/CKIM_Competition/federatedscope', '/home/ms234795/Master Thesis/CKIM_Competition',] + sys.path

print(sys.path)
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.core.fed_runner import FedRunner
from yacs.config import CfgNode
from federatedscope.contrib.workers.fedavg_VAE_client import Fedavg_VAE_client

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']
from federatedscope.register import register_trainer
from federatedscope.contrib.trainer.FedAvg_VAE_trainer import call_fedavg_VAE_trainer

register_trainer('FedAvg_VAE_trainer', call_fedavg_VAE_trainer)

def train():
    cfg_file = 'scripts/B-FHTL_exp_scripts/Graph-DT/fedavg_VAE.yaml'
    cfg_client = 'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client_theirs.yaml'
    # cfg_per_Client_ours_lr
    # cfg_per_client_ours_lr_local_steps

    #'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(cfg_file)
    init_cfg.federate.client_num = 16
    init_cfg.params = CN()
    init_cfg.params.alpha = 0.1
    # init_cfg.data.subdirectory = 'graph_dt_backup/processed'
    # init_cfg.merge_from_list(args.opts)
    # init_cfg.data.client = 5
    # init_cfg.train.optgraph_level_defaultimizer.lr = 0.01
    init_cfg.data.save_dir = 'FedAvg_with_encoder_KLD_their_lrs'
    update_logger(init_cfg)
    setup_seed(init_cfg.seed)

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
                       server_class=get_server_cls(init_cfg),
                       client_class=Fedavg_VAE_client,
                       config=init_cfg.clone(),
                       client_config=cfg_client, )
    _ = runner.run()


if __name__ == '__main__':
    num_trainings = 1
    for i in range(num_trainings):
        print(f"training run: {i + 1}")
        train()