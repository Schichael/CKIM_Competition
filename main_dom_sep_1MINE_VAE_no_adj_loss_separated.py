import os
import sys

from federatedscope.contrib.trainer.laplacian_trainer_with_domain_separation_with_summation_1MINE_VAE import \
    call_laplacian_trainer
# sys.path = ['/home/ms234795/Master Thesis/CKIM_Competition/federatedscope', '/home/ms234795/Master Thesis/CKIM_Competition',] + sys.path
sys.path = ['~/Master-Thesis/CKIM_Competition/federatedscope', '~/Master-Thesis/CKIM_Competition',] + sys.path

print(sys.path)
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from yacs.config import CfgNode
from federatedscope.register import register_data
from federatedscope.register import register_trainer
from federatedscope.contrib.data.cikm_cup import call_cikm_cup_data
from federatedscope.core.configs.config import global_cfg, CN
from federatedscope.contrib.workers.laplacian_with_domain_separation_MI_client import LaplacianDomainSeparationMIClient
from federatedscope.contrib.workers.laplacian_server_dom_sep import LaplacianServerDomSep
from federatedscope.contrib.workers.laplacian_server import LaplacianServer
from federatedscope.contrib.workers.laplacian_with_domain_separation_1MINE_VAE_Separated_client import LaplacianDomainSeparation1MINE_Separated_Client

from federatedscope.contrib.trainer.laplacian_trainer_with_domain_separation_with_summation_1MINE_VAE_separated import call_laplacian_trainer

register_trainer('laplacian_trainer', call_laplacian_trainer)




if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']


def train():
    cfg_file = 'scripts/B-FHTL_exp_scripts/Graph-DT/fed_dom_sep_local_MI.yaml'
    cfg_client = 'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client_theirs.yaml'
    # cfg_per_Client_ours_lr
    # cfg_per_client_ours_lr_local_steps

    #'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(cfg_file)

    # init_cfg.data.subdirectory = 'graph_dt_backup/processed'
    # init_cfg.merge_from_list(args.opts)
    init_cfg.data.save_dir = 'SEPARATED_NEW_local_MINE_12_01_[2]csd_1e2_diff_imp_0_1_lam_0_kld_imp_0_recon_imp_0'
    init_cfg.model.dropout = 0.5
    init_cfg.params = CN()
    init_cfg.params.alpha = 0.1
    init_cfg.params.diff_importance = 0.1
    init_cfg.params.csd_importance = 1e2  # 1e2 #  1e2  # 1e2
    init_cfg.params.mine_lr = 0.01
    init_cfg.params.lam = 0  # 0.01
    init_cfg.params.eps = 1e-20
    init_cfg.params.kld_importance = 0.01  # 0.01
    init_cfg.params.recon_importance = 0.01  # 0.01
    init_cfg.federate.client_num = 16
    init_cfg.params.p = 0.
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
                   server_class=LaplacianServer,
                   client_class=LaplacianDomainSeparation1MINE_Separated_Client,
                   config=init_cfg.clone(),
                   client_config=cfg_client)
    _ = runner.run()


if __name__ == '__main__':
    num_trainings = 1
    for i in range(num_trainings):
        print(f"training run: {i + 1}")
        train()