import os
import sys

from federatedscope.contrib.metrics.custom_losses import call_kld_loss_encoder_metric
from federatedscope.contrib.trainer.FedAvg_VAE_trainer import call_fedavg_VAE_trainer
from federatedscope.contrib.workers.fedavg_VAE_client import Fedavg_VAE_client
from federatedscope.register import register_trainer, register_metric

# sys.path = ['~/Master-Thesis/CKIM_Competition/federatedscope', '~/Master-Thesis/CKIM_Competition',] + sys.path
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

register_trainer('laplacian_trainer', call_fedavg_VAE_trainer)

metrics = [
 ('kld_loss_encoder', call_kld_loss_encoder_metric),
           ]
for metric in metrics:
    register_metric(metric[0], metric[1])



def train(lr, kld_imp):
    cfg_file = 'scripts/B-FHTL_exp_scripts/Graph-DC/fedavg.yaml'
    cfg_client = 'scripts/B-FHTL_exp_scripts/Graph-DC/cfg_per_client.yaml'
    # cfg_per_Client_ours_lr
    # cfg_per_client_ours_lr_local_steps


    #'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(cfg_file)

    # init_cfg.data.subdirectory = 'graph_dt_backup/processed'
    # init_cfg.merge_from_list(args.opts)
    init_cfg.data.save_dir = 'Graph-DC_FedAvg_NE_KLD_withRepara_newRepara_lr_' + str(
        lr).replace(
        '.', '_')+ '_local_update_steps_1_KLD_imp_' + str(kld_imp).replace('.', '_')
    init_cfg.train.optimizer.lr = lr
    init_cfg.params = CN()

    init_cfg.params.vae_importance = kld_imp

    init_cfg.model.dropout = 0.5

    init_cfg.federate.client_num = 13

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
                   client_config=cfg_client)
    _ = runner.run()


if __name__ == '__main__':

    lrs = [0.1]
    kld_imps = [0.01,10,20,50,100]
    num_trainings = 1
    for lr in lrs:
        for kld_imp in kld_imps:
            for i in range(num_trainings):
                print(f"training run: {i + 1}")
                train(lr, kld_imp)