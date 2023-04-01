import numpy as np
def recon_loss_metric(ctx, **kwargs):
    return np.mean(ctx.rec_loss_metric)

def kld_loss_encoder_metric(ctx, **kwargs):
    return np.mean(ctx.kld_loss_encoder_metric)

def kld_global_metric(ctx, **kwargs):
    return np.mean(ctx.kld_global_metric)

def kld_interm_metric(ctx, **kwargs):
    return np.mean(ctx.kld_interm_metric)

def kld_local_metric(ctx, **kwargs):
    return np.mean(ctx.kld_local_metric)

def diff_local_interm_metric(ctx, **kwargs):
    return np.mean(ctx.diff_local_interm_metric)

def diff_local_global_metric(ctx, **kwargs):
    return np.mean(ctx.diff_local_global_metric)

def diff_local_fixed_metric(ctx, **kwargs):
    return np.mean(ctx.diff_local_fixed_metric)

def sim_global_interm_metric(ctx, **kwargs):
    return np.mean(ctx.sim_global_interm_metric)

def sim_global_fixed_metric(ctx, **kwargs):
    return np.mean(ctx.sim_global_fixed_metric)


def loss_out_interm_metric(ctx, **kwargs):
    return np.mean(ctx.loss_out_interm_metric)

def loss_out_local_interm_metric(ctx, **kwargs):
    return np.mean(ctx.loss_out_local_interm_metric)

def loss_batch_csd_metric(ctx, **kwargs):
    return np.mean(ctx.loss_batch_csd_metric)

def prox_loss_metric(ctx, **kwargs):
    return np.mean(ctx.prox_loss_metric)

def mmd_loss_metric(ctx, **kwargs):
    return np.mean(ctx.mmd_loss_metric)

def swd_loss_metric(ctx, **kwargs):
    return np.mean(ctx.swd_loss_metric)

def mi_estimation_metric(ctx, **kwargs):
    return np.mean(ctx.MI_metric)

def diff_resnet_1_metric(ctx, **kwargs):
    return np.mean(ctx.diff_loss_1_metric)

def diff_resnet_2_metric(ctx, **kwargs):
    return np.mean(ctx.diff_loss_2_metric)

def call_mi_estimation_metric(types):
    if "MI_estimation" in types:
        return "MI_estimation", mi_estimation_metric
def call_prox_loss_metric(types):
    if "prox_loss" in types:
        return "prox_loss", prox_loss_metric

def call_recon_loss_metric(types):
    if "recon_loss" in types:
        return "recon_loss", recon_loss_metric


def call_kld_loss_encoder_metric(types):
    if "kld_loss_encoder" in types:
        return "kld_loss_encoder", kld_loss_encoder_metric

def call_kld_global_metric(types):
    if "kld_global" in types:
        return "kld_global", kld_global_metric

def call_kld_interm_metric(types):
    if "kld_interm" in types:
        return "kld_interm", kld_interm_metric

def call_kld_local_metric(types):
    if "kld_local" in types:
        return "kld_local", kld_local_metric

def call_diff_local_interm_metric(types):
    if "diff_local_interm" in types:
        return "diff_local_interm", diff_local_interm_metric

def call_diff_resnet_1_metric(types):
    if "diff_resnet_1" in types:
        return "diff_resnet_1", diff_resnet_1_metric

def call_diff_resnet_2_metric(types):
    if "diff_resnet_2" in types:
        return "diff_resnet_2", diff_resnet_2_metric
def call_diff_local_global_metric(types):
    if "diff_local_global" in types:
        return "diff_local_global", diff_local_global_metric

def call_diff_local_fixed_metric(types):
    if "diff_local_fixed" in types:
        return "diff_local_fixed", diff_local_fixed_metric

def call_sim_global_interm_metric(types):
    if "sim_global_interm" in types:
        return "sim_global_interm", sim_global_interm_metric

def call_sim_global_fixed_metric(types):
    if "sim_global_fixed" in types:
        return "sim_global_fixed", sim_global_fixed_metric

def call_loss_out_interm_metric(types):
    if "loss_out_interm" in types:
        return "loss_out_interm", loss_out_interm_metric

def call_loss_out_local_interm_metric(types):
    if "loss_out_local_interm" in types:
        return "loss_out_local_interm", loss_out_local_interm_metric

def call_loss_batch_csd_metric(types):
    if "loss_batch_csd" in types:
        return "loss_batch_csd", loss_batch_csd_metric

def call_mmd_loss_metric(types):
    if "mmd_loss" in types:
        return "mmd_loss", mmd_loss_metric

def call_swd_loss_metric(types):
    if "swd_loss" in types:
        return "swd_loss", swd_loss_metric

