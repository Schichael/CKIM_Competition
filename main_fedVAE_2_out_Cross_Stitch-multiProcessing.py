import os
import sys
import time
from multiprocessing import set_start_method

import torch
from torch import multiprocessing

from federatedscope.contrib.workers.laplacian_Cross_Stitch_client import \
    Laplacian_Cross_Stitch_Client
from federatedscope.contrib.workers.laplacian_server_cross_stitch import \
    LaplacianServer_CrossStitch
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_2_out_only_COSINEdiff_MSEsim_global_private_clf_NEW import \
    LaplacianDomainSeparationVAE_2Out_Only_COSINE_Diff_MSE_Sim_global_private_NEW_Trainer
from federatedscope.contrib.workers.laplacian_server_dom_sep_VAE_1_out_global_private import \
    LaplacianServerDomSepVAE_1_out_global_private
from federatedscope.contrib.workers.laplacian_with_domain_separation_1_out_onlyDiff_Sim_global_private_clf_NEW_client import \
    LaplacianDomainSeparationVAE_1_out_onlyDiff_Sim_global_private_clf_NEW_Client
from federatedscope.contrib.workers.laplacian_with_domain_separation_2_out_onlyDiff_Sim_global_private_clf_NEW_client import \
    LaplacianDomainSeparationVAE_2_out_onlyCosineDiff_MSESim_global_private_clf_NEW_Client
from federatedscope.contrib.workers\
    .laplacian_with_domain_separation_2_out_onlyDiff_only2branches_global_clf_NEW_client import \
    LaplacianDomainSeparationVAE_2_out_onlyDiff_only2branches_global_clf_NEW_Client
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_2_out_only_diff_sim_only2branches_NEW import \
    call_laplacian_trainer
#from federatedscope.contrib.trainer.laplacian_trainer import call_laplacian_trainer
from federatedscope.contrib.workers.laplacian_client import LaplacianClient
from federatedscope.contrib.workers.laplacian_diff_global_out_client import LaplacianDiffGlobalOutClient
from federatedscope.contrib.workers.laplacian_server import LaplacianServer
from federatedscope.contrib.workers.laplacian_server_dom_sep_VAE_1_out import LaplacianServerDomSepVAE_1_out
from federatedscope.contrib.workers.laplacian_server_dom_sep_without_fixed import LaplacianServerDomSepWithoutFixed
from federatedscope.contrib.workers.laplacian_with_domain_separation_2_out_onlyDiff_only2branches_NEW_client import \
    LaplacianDomainSeparationVAE_2_out_onlyDiff_only2branches_NEW_Client
from federatedscope.contrib.workers.laplacian_with_domain_separation_VAE_1_out_client import \
    LaplacianDomainSeparationVAE_1_out_Client
from federatedscope.contrib.workers.laplacian_with_domain_separation_VAE_2_out_NEW_client import \
    LaplacianDomainSeparationVAE_2_out_NEW_Client
from federatedscope.contrib.workers.laplacian_with_domain_separation_VAE_2_out_client import \
    LaplacianDomainSeparationVAE_2_out_Client
from federatedscope.contrib.workers.laplacian_with_domain_separation_2_out_onlyDiffProxLoss_NEW_client import \
    LaplacianDomainSeparationVAE_2_out_onlyDiffProxLoss_NEW_Client
from federatedscope.contrib.workers.laplacian_with_domain_separation_VAE_2_out_onlyDiffSim_NEW_client import \
    LaplacianDomainSeparationVAE_2_out_onlyDiffSim_NEW_Client
from federatedscope.register import register_trainer
from federatedscope.register import register_metric
from federatedscope.contrib.metrics.custom_losses import call_recon_loss_metric, \
    call_kld_loss_encoder_metric, call_kld_global_metric, call_kld_interm_metric, \
    call_kld_local_metric, \
    call_diff_local_interm_metric, call_sim_global_interm_metric, \
    call_loss_out_interm_metric, \
    call_loss_out_local_interm_metric, call_loss_batch_csd_metric, \
    call_prox_loss_metric, call_diff_local_global_metric, call_loss_global_clf_metric, \
    call_num_local_features_not_0_metric, call_avg_local_features_not_0_metric, \
    call_num_global_features_not_0_metric, call_avg_global_features_not_0_metric, \
    call_num_local_global_features_not_0_metric, \
    call_avg_local_global_features_not_0_metric, call_num_features_global_local_metric, \
    call_num_local_out_features_not_0_metric, call_avg_local_out_features_not_0_metric, \
    call_num_features_global_local_out_metric, call_num_features_local_out_local_metric, \
    call_diff_local_local_out_metric, call_sim_interm_local_out_metric, \
    call_diff_1_metric, call_diff_2_metric, call_diff_3_metric, \
    call_local_alpha_1_global_metric, call_local_alpha_1_local_metric, \
    call_local_alpha_2_global_metric, call_local_alpha_2_local_metric, \
    call_local_alpha_3_global_metric, call_local_alpha_3_local_metric, \
    call_local_alpha_1_metric, call_local_alpha_2_metric, call_local_alpha_3_metric, \
    call_num_local_features_not_0_1_metric, call_avg_local_features_not_0_1_metric, \
    call_num_global_features_not_0_1_metric, call_avg_global_features_not_0_1_metric, \
    call_num_global_combined_features_not_0_1_metric, \
    call_avg_global_combined_features_not_0_1_metric, \
    call_num_features_global_local_1_metric, \
    call_cos_sim_local_global_combined_1_metric, \
    call_cos_sim_global_local_combined_1_metric, \
    call_cos_sim_global_global_combined_1_metric, \
    call_cos_sim_local_local_combined_1_metric, call_num_local_features_not_0_2_metric, \
    call_avg_local_features_not_0_2_metric, call_num_global_features_not_0_2_metric, \
    call_avg_global_features_not_0_2_metric, \
    call_num_global_combined_features_not_0_2_metric, \
    call_avg_global_combined_features_not_0_2_metric, \
    call_num_features_global_local_2_metric, \
    call_cos_sim_local_global_combined_2_metric, \
    call_cos_sim_global_local_combined_2_metric, \
    call_cos_sim_global_global_combined_2_metric, \
    call_cos_sim_local_local_combined_2_metric, call_num_local_features_not_0_3_metric, \
    call_avg_local_features_not_0_3_metric, call_num_global_features_not_0_3_metric, \
    call_num_global_combined_features_not_0_3_metric, \
    call_avg_global_combined_features_not_0_3_metric, \
    call_avg_global_features_not_0_3_metric, call_num_features_global_local_3_metric, \
    call_cos_sim_local_global_combined_3_metric, \
    call_cos_sim_global_local_combined_3_metric, \
    call_cos_sim_global_global_combined_3_metric, \
    call_cos_sim_local_local_combined_3_metric

try:
    torch.multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass



metrics = [
    ('kld_loss_encoder', call_kld_loss_encoder_metric),
    ('diff_local_global', call_diff_local_global_metric),
    ('loss_batch_csd', call_loss_batch_csd_metric),
    ('loss_global_clf_metric', call_loss_global_clf_metric),
('num_local_features_not_0_1_metric', call_num_local_features_not_0_1_metric),
('avg_local_features_not_0_1_metric', call_avg_local_features_not_0_1_metric),
('num_global_features_not_0_1_metric', call_num_global_features_not_0_1_metric),
('avg_global_features_not_0_1_metric', call_avg_global_features_not_0_1_metric),
('num_global_combined_features_not_0_1_metric', call_num_global_combined_features_not_0_1_metric),
('avg_global_combined_features_not_0_1_metric', call_avg_global_combined_features_not_0_1_metric),
    ('num_features_global_local_1_metric', call_num_features_global_local_1_metric),
('cos_sim_local_global_combined_1_metric', call_cos_sim_local_global_combined_1_metric),
('cos_sim_global_local_combined_1_metric', call_cos_sim_global_local_combined_1_metric),
('cos_sim_global_global_combined_1_metric', call_cos_sim_global_global_combined_1_metric),
('cos_sim_local_local_combined_1_metric', call_cos_sim_local_local_combined_1_metric),

('num_local_features_not_0_2_metric', call_num_local_features_not_0_2_metric),
('avg_local_features_not_0_2_metric', call_avg_local_features_not_0_2_metric),
('num_global_features_not_0_2_metric', call_num_global_features_not_0_2_metric),
('avg_global_features_not_0_2_metric', call_avg_global_features_not_0_2_metric),
('num_global_combined_features_not_0_2_metric', call_num_global_combined_features_not_0_2_metric),
('avg_global_combined_features_not_0_2_metric', call_avg_global_combined_features_not_0_2_metric),
    ('num_features_global_local_2_metric', call_num_features_global_local_2_metric),
('cos_sim_local_global_combined_2_metric', call_cos_sim_local_global_combined_2_metric),
('cos_sim_global_local_combined_2_metric', call_cos_sim_global_local_combined_2_metric),
('cos_sim_global_global_combined_2_metric', call_cos_sim_global_global_combined_2_metric),
('cos_sim_local_local_combined_2_metric', call_cos_sim_local_local_combined_2_metric),

('num_local_features_not_0_3_metric', call_num_local_features_not_0_3_metric),
('avg_local_features_not_0_3_metric', call_avg_local_features_not_0_3_metric),
('num_global_features_not_0_3_metric', call_num_global_features_not_0_3_metric),
('avg_global_features_not_0_3_metric', call_avg_global_features_not_0_3_metric),
('num_global_combined_features_not_0_3_metric', call_num_global_combined_features_not_0_3_metric),
('avg_global_combined_features_not_0_3_metric', call_avg_global_combined_features_not_0_3_metric),
('num_features_global_local_3_metric', call_num_features_global_local_3_metric),
('cos_sim_local_global_combined_3_metric', call_cos_sim_local_global_combined_3_metric),
('cos_sim_global_local_combined_3_metric', call_cos_sim_global_local_combined_3_metric),
('cos_sim_global_global_combined_3_metric', call_cos_sim_global_global_combined_3_metric),
('cos_sim_local_local_combined_3_metric', call_cos_sim_local_local_combined_3_metric),


('diff_1_metric', call_diff_1_metric),
('diff_2_metric', call_diff_2_metric),
('diff_3_metric', call_diff_3_metric),

('local_alpha_1_global_metric', call_local_alpha_1_global_metric),
('local_alpha_1_local_metric', call_local_alpha_1_local_metric),
('local_alpha_2_global_metric', call_local_alpha_2_global_metric),
('local_alpha_2_local_metric', call_local_alpha_2_local_metric),
('local_alpha_3_global_metric', call_local_alpha_3_global_metric),
('local_alpha_3_local_metric', call_local_alpha_3_local_metric),








]

"""
ctx.num_local_out_features_not_0_metric.append(num_local_out_features_not_0)
        ctx.avg_local_out_features_not_0_metric.append(avg_local_out_features_not_0)
        ctx.num_features_global_local_out_metric.append(num_features_global_local_out)
        ctx.num_features_local_out_local_metric.append(num_features_local_out_local)
"""

for metric in metrics:
    register_metric(metric[0], metric[1])

#sys.path = ['~/Master-Thesis/CKIM_Competition/federatedscope',
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

# register_trainer('laplacian_trainer', call_laplacian_trainer)


def train(lr, diff_imp, csd_imp):



    cfg_file = 'scripts/B-FHTL_exp_scripts/Graph-DC/fedDomSep_2_out_CrossStitch.yaml'
    cfg_client = 'scripts/B-FHTL_exp_scripts/Graph-DC/cfg_per_client.yaml'
    # cfg_per_Client_ours_lr
    # cfg_per_client_ours_lr_local_steps

    #'scripts/B-FHTL_exp_scripts/Graph-DT/cfg_per_client.yaml'

    init_cfg = global_cfg.clone()
    init_cfg.merge_from_file(cfg_file)
    # init_cfg.data.subdirectory = 'graph_dt_backup/processed'
    # init_cfg.merge_from_list(args.opts)
    init_cfg.data.save_dir = \
        'Graph-DC_2_out_Cross_Stitch_multi_runs_alpha_non_trainable_init_0_95global_0_05local_lr_' + \
        str(lr).replace(
            '.', '_') + \
    '_F' + str(diff_imp).replace('.', '_') + '_F' + '_H' + str(csd_imp).replace(
        '.', '_')
    """
        kld_ne_imps = [1] #A
        kld_local_imp = 1 #B
        kld_interm_imp = 1 #C
        kld_global_imps = [0.1] #D
        recon_imp = 0.1 #E
        diff_interm_imp = [0.1] #F
        diff_local_imp = 0. #G
        csd_imp = 10 #H
        sims = [0.1] #I
    """

    init_cfg.params = CN()
    init_cfg.params.diff_imp = diff_imp
    init_cfg.params.csd_imp = csd_imp


    init_cfg.federate.client_num = 13
    init_cfg.params.eps = 1e-15
    init_cfg.federate.total_round_num = 1000

    init_cfg.params.save_client_always = True
    init_cfg.params.p = 0.
    init_cfg.params.alpha = 0.1

    init_cfg.model.dropout = 0.5
    init_cfg.train.optimizer.lr = lr
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
                   server_class = LaplacianServer_CrossStitch,
                   client_class = Laplacian_Cross_Stitch_Client,
                   config=init_cfg.clone(),
                   client_config=cfg_client)
    _ = runner.run()

def tmp(a):
    print(a)
    return a

if __name__ == '__main__':

    num_trainings = 3
    kld_ne_imps = [0] #A
    diff_imps = [0.5, 0.05, 0.005, 0]
    #diff_imps = [0.05]
    #diff_imps = [0.1]
    #diff_global_imps = [0] #F    HERE  [0.0001, 0.001]
    #diff_local_imps = [0] #G
    csd_imp = 10 #H
    #global_clf_imps = [0.1]
    #sim_losses = ["mse", "cosine"]

    # lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lrs = [0.05]
    pool = multiprocessing.Pool(4)
    processes = []
    for lr in lrs:
        for diff_imp in diff_imps:
                for kld_ne_imp in kld_ne_imps:
                        for i in range(num_trainings):
                            time.sleep(10)
                            setup_seed(i)
                            processes.append(pool.apply_async(train, args=(lr,
                                                                           diff_imp,
                                                                           csd_imp)))
    result = [p.get() for p in processes]

    #kld=0 mit repara: ~1.00 - 1.05
    #kld=0.01 mit repara: ~1.00 - 1.05
    # kld=0.01 mit repara: ~1.37
    #  nur mu: ~1.00