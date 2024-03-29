import numpy as np
def recon_loss_metric(ctx, **kwargs):
    return np.mean(ctx.recon_loss_metric)

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

def loss_global_clf_metric(ctx, **kwargs):
    return np.mean(ctx.loss_global_clf_metric)

def num_local_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.num_local_features_not_0_metric)

def avg_local_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.avg_local_features_not_0_metric)

def num_global_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_features_not_0_metric)

def avg_global_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_features_not_0_metric)

def num_local_global_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.num_local_global_features_not_0_metric)


def diff_1_metric(ctx, **kwargs):
    return np.mean(ctx.diff_1_metric)

def diff_2_metric(ctx, **kwargs):
    return np.mean(ctx.diff_2_metric)

def diff_3_metric(ctx, **kwargs):
    return np.mean(ctx.diff_3_metric)

def local_alpha_1_global_metric(ctx, **kwargs):
    return np.mean(ctx.local_alpha_1_global_metric)

def local_alpha_1_local_metric(ctx, **kwargs):
    return np.mean(ctx.local_alpha_1_local_metric)

def local_alpha_2_global_metric(ctx, **kwargs):
    return np.mean(ctx.local_alpha_2_global_metric)

def local_alpha_1_metric(ctx, **kwargs):
    return ctx.local_alpha_1_metric

def local_alpha_2_metric(ctx, **kwargs):
    return ctx.local_alpha_2_metric

def local_alpha_3_metric(ctx, **kwargs):
    return ctx.local_alpha_1_metric

def local_alpha_2_local_metric(ctx, **kwargs):
    return np.mean(ctx.local_alpha_2_local_metric)

def local_alpha_3_global_metric(ctx, **kwargs):
    return np.mean(ctx.local_alpha_3_global_metric)

def local_alpha_3_local_metric(ctx, **kwargs):
    return np.mean(ctx.local_alpha_3_local_metric)

def avg_local_global_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.avg_local_global_features_not_0_metric)

def num_features_global_local_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_global_local_metric)

def num_local_out_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.num_local_out_features_not_0_metric)

def avg_local_out_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.avg_local_out_features_not_0_metric)

def num_features_global_local_out_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_global_local_out_metric)

def num_features_local_out_local_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_local_out_local_metric)

def diff_local_local_out_metric(ctx, **kwargs):
    return np.mean(ctx.diff_local_local_out_metric)

def sim_interm_local_out_metric(ctx, **kwargs):
    return np.mean(ctx.sim_interm_local_out_metric)

def cos_sim_local_local_combined_3_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_local_combined_3_metric)


def cos_sim_global_global_combined_3_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_global_combined_3_metric)


def cos_sim_global_local_combined_3_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_local_combined_3_metric)


def cos_sim_local_global_combined_3_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_global_combined_3_metric)


def num_features_global_local_3_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_global_local_3_metric)


def avg_global_combined_features_not_0_3_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_combined_features_not_0_3_metric)


def num_global_combined_features_not_0_3_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_combined_features_not_0_3_metric)


def avg_global_features_not_0_3_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_features_not_0_3_metric)


def num_global_features_not_0_3_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_features_not_0_3_metric)


def avg_local_features_not_0_3_metric(ctx, **kwargs):
    return np.mean(ctx.avg_local_features_not_0_3_metric)


def num_local_features_not_0_3_metric(ctx, **kwargs):
    return np.mean(ctx.num_local_features_not_0_3_metric)

def call_mi_estimation_metric(types):
    if "MI_estimation" in types:
        return "MI_estimation", mi_estimation_metric
def call_prox_loss_metric(types):
    if "prox_loss_metric" in types:
        return "prox_loss_metric", prox_loss_metric

def call_recon_loss_metric(types):
    if "recon_loss_metric" in types:
        return "recon_loss_metric", recon_loss_metric

def call_kld_loss_encoder_metric(types):
    if "kld_loss_encoder_metric" in types:
        return "kld_loss_encoder_metric", kld_loss_encoder_metric

def call_kld_global_metric(types):
    if "kld_global_metric" in types:
        return "kld_global_metric", kld_global_metric

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
def call_loss_global_clf_metric(types):
    if "loss_global_clf" in types:
        return "loss_global_clf", loss_global_clf_metric
def call_mmd_loss_metric(types):
    if "mmd_loss" in types:
        return "mmd_loss", mmd_loss_metric

def call_swd_loss_metric(types):
    if "swd_loss" in types:
        return "swd_loss", swd_loss_metric

def call_num_local_features_not_0_metric(types):
    if "num_local_features_not_0_metric" in types:
        return "num_local_features_not_0_metric", num_local_features_not_0_metric

def call_avg_local_features_not_0_metric(types):
    if "avg_local_features_not_0_metric" in types:
        return "avg_local_features_not_0_metric", avg_local_features_not_0_metric

def call_num_global_features_not_0_metric(types):
    if "num_global_features_not_0_metric" in types:
        return "num_global_features_not_0_metric", num_global_features_not_0_metric

def call_avg_global_features_not_0_metric(types):
    if "avg_global_features_not_0_metric" in types:
        return "avg_global_features_not_0_metric", avg_global_features_not_0_metric

def call_num_local_global_features_not_0_metric(types):
    if "num_local_global_features_not_0_metric" in types:
        return "num_local_global_features_not_0_metric", num_local_global_features_not_0_metric

def call_avg_local_global_features_not_0_metric(types):
    if "avg_local_global_features_not_0_metric" in types:
        return "avg_local_global_features_not_0_metric", avg_local_global_features_not_0_metric

def call_num_features_global_local_metric(types):
    if "num_features_global_local_metric" in types:
        return "num_features_global_local_metric", num_features_global_local_metric

def call_num_local_out_features_not_0_metric(types):
    if "num_local_out_features_not_0_metric" in types:
        return "num_local_out_features_not_0_metric", num_local_out_features_not_0_metric

def call_avg_local_out_features_not_0_metric(types):
    if "avg_local_out_features_not_0_metric" in types:
        return "avg_local_out_features_not_0_metric", avg_local_out_features_not_0_metric

def call_num_features_global_local_out_metric(types):
    if "num_features_global_local_out_metric" in types:
        return "num_features_global_local_out_metric", num_features_global_local_out_metric

def call_num_features_local_out_local_metric(types):
    if "num_features_local_out_local_metric" in types:
        return "num_features_local_out_local_metric", num_features_local_out_local_metric

def call_diff_local_local_out_metric(types):
    if "diff_local_local_out_metric" in types:
        return "diff_local_local_out_metric", diff_local_local_out_metric

def call_sim_interm_local_out_metric(types):
    if "sim_interm_local_out_metric" in types:
        return "sim_interm_local_out_metric", sim_interm_local_out_metric


def call_diff_1_metric(types):
    if "diff_1_metric" in types:
        return "diff_1_metric", diff_1_metric

def call_diff_2_metric(types):
    if "diff_2_metric" in types:
        return "diff_2_metric", diff_2_metric

def call_diff_3_metric(types):
    if "diff_3_metric" in types:
        return "diff_3_metric", diff_3_metric

def call_local_alpha_1_global_metric(types):
    if "local_alpha_1_global_metric" in types:
        return "local_alpha_1_global_metric", local_alpha_1_global_metric

def call_local_alpha_1_local_metric(types):
    if "local_alpha_1_local_metric" in types:
        return "local_alpha_1_local_metric", local_alpha_1_local_metric

def call_local_alpha_2_global_metric(types):
    if "local_alpha_2_global_metric" in types:
        return "local_alpha_2_global_metric", local_alpha_2_global_metric

def call_local_alpha_2_local_metric(types):
    if "local_alpha_2_local_metric" in types:
        return "local_alpha_2_local_metric", local_alpha_2_local_metric

def call_local_alpha_3_global_metric(types):
    if "local_alpha_3_global_metric" in types:
        return "local_alpha_3_global_metric", local_alpha_3_global_metric

def call_local_alpha_3_local_metric(types):
    if "local_alpha_3_local_metric" in types:
        return "local_alpha_3_local_metric", local_alpha_3_local_metric

def call_local_alpha_1_metric(types):
    if "local_alpha_1_metric" in types:
        return "local_alpha_1_metric", local_alpha_1_metric

def call_local_alpha_2_metric(types):
    if "local_alpha_2_metric" in types:
        return "local_alpha_2_metric", local_alpha_2_metric

def call_local_alpha_3_metric(types):
    if "local_alpha_3_metric" in types:
        return "local_alpha_3_metric", local_alpha_3_metric



def call_num_local_features_not_0_1_metric(types):
    if "num_local_features_not_0_1_metric" in types:
        return "num_local_features_not_0_1_metric", num_local_features_not_0_1_metric

def call_avg_local_features_not_0_1_metric(types):
    if "avg_local_features_not_0_1_metric" in types:
        return "avg_local_features_not_0_1_metric", avg_local_features_not_0_1_metric

def call_num_global_features_not_0_1_metric(types):
    if "num_global_features_not_0_1_metric" in types:
        return "num_global_features_not_0_1_metric", num_global_features_not_0_1_metric

def call_avg_global_features_not_0_1_metric(types):
    if "avg_global_features_not_0_1_metric" in types:
        return "avg_global_features_not_0_1_metric", avg_global_features_not_0_1_metric

def call_num_global_combined_features_not_0_1_metric(types):
    if "num_global_combined_features_not_0_1_metric" in types:
        return "num_global_combined_features_not_0_1_metric", num_global_combined_features_not_0_1_metric

def call_avg_global_combined_features_not_0_1_metric(types):
    if "avg_global_combined_features_not_0_1_metric" in types:
        return "avg_global_combined_features_not_0_1_metric", avg_global_combined_features_not_0_1_metric

def call_num_features_global_local_1_metric(types):
    if "num_features_global_local_1_metric" in types:
        return "num_features_global_local_1_metric", num_features_global_local_1_metric

def call_cos_sim_local_global_combined_1_metric(types):
    if "cos_sim_local_global_combined_1_metric" in types:
        return "cos_sim_local_global_combined_1_metric", cos_sim_local_global_combined_1_metric

def call_cos_sim_global_local_combined_1_metric(types):
    if "cos_sim_global_local_combined_1_metric" in types:
        return "cos_sim_global_local_combined_1_metric", cos_sim_global_local_combined_1_metric

def call_cos_sim_global_global_combined_1_metric(types):
    if "cos_sim_global_global_combined_1_metric" in types:
        return "cos_sim_global_global_combined_1_metric", cos_sim_global_global_combined_1_metric

def call_cos_sim_local_local_combined_1_metric(types):
    if "cos_sim_local_local_combined_1_metric" in types:
        return "cos_sim_local_local_combined_1_metric", cos_sim_local_local_combined_1_metric


def cos_sim_local_local_combined_1_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_local_combined_1_metric)

def cos_sim_global_global_combined_1_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_global_combined_1_metric)

def cos_sim_global_local_combined_1_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_local_combined_1_metric)
def cos_sim_local_global_combined_1_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_global_combined_1_metric)
def num_features_global_local_1_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_global_local_1_metric)
def avg_global_combined_features_not_0_1_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_combined_features_not_0_1_metric)
def num_global_combined_features_not_0_1_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_combined_features_not_0_1_metric)
def avg_global_features_not_0_1_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_features_not_0_1_metric)
def num_global_features_not_0_1_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_features_not_0_1_metric)

def avg_local_features_not_0_1_metric(ctx, **kwargs):
    return np.mean(ctx.avg_local_features_not_0_1_metric)

def num_local_features_not_0_1_metric(ctx, **kwargs):
    return np.mean(ctx.num_local_features_not_0_1_metric)


def call_num_local_features_not_0_2_metric(types):
    if "num_local_features_not_0_2_metric" in types:
        return "num_local_features_not_0_2_metric", num_local_features_not_0_2_metric


def call_avg_local_features_not_0_2_metric(types):
    if "avg_local_features_not_0_2_metric" in types:
        return "avg_local_features_not_0_2_metric", avg_local_features_not_0_2_metric


def call_num_global_features_not_0_2_metric(types):
    if "num_global_features_not_0_2_metric" in types:
        return "num_global_features_not_0_2_metric", num_global_features_not_0_2_metric


def call_avg_global_features_not_0_2_metric(types):
    if "avg_global_features_not_0_2_metric" in types:
        return "avg_global_features_not_0_2_metric", avg_global_features_not_0_2_metric


def call_num_global_combined_features_not_0_2_metric(types):
    if "num_global_combined_features_not_0_2_metric" in types:
        return "num_global_combined_features_not_0_2_metric", num_global_combined_features_not_0_2_metric


def call_avg_global_combined_features_not_0_2_metric(types):
    if "avg_global_combined_features_not_0_2_metric" in types:
        return "avg_global_combined_features_not_0_2_metric", avg_global_combined_features_not_0_2_metric


def call_num_features_global_local_2_metric(types):
    if "num_features_global_local_2_metric" in types:
        return "num_features_global_local_2_metric", num_features_global_local_2_metric


def call_cos_sim_local_global_combined_2_metric(types):
    if "cos_sim_local_global_combined_2_metric" in types:
        return "cos_sim_local_global_combined_2_metric", cos_sim_local_global_combined_2_metric


def call_cos_sim_global_local_combined_2_metric(types):
    if "cos_sim_global_local_combined_2_metric" in types:
        return "cos_sim_global_local_combined_2_metric", cos_sim_global_local_combined_2_metric


def call_cos_sim_global_global_combined_2_metric(types):
    if "cos_sim_global_global_combined_2_metric" in types:
        return "cos_sim_global_global_combined_2_metric", cos_sim_global_global_combined_2_metric


def call_cos_sim_local_local_combined_2_metric(types):
    if "cos_sim_local_local_combined_2_metric" in types:
        return "cos_sim_local_local_combined_2_metric", cos_sim_local_local_combined_2_metric


def cos_sim_local_local_combined_2_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_local_combined_2_metric)


def cos_sim_global_global_combined_2_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_global_combined_2_metric)


def cos_sim_global_local_combined_2_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_local_combined_2_metric)


def cos_sim_local_global_combined_2_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_global_combined_2_metric)


def num_features_global_local_2_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_global_local_2_metric)


def avg_global_combined_features_not_0_2_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_combined_features_not_0_2_metric)


def num_global_combined_features_not_0_2_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_combined_features_not_0_2_metric)


def avg_global_features_not_0_2_metric(ctx, **kwargs):
    return np.mean(ctx.avg_global_features_not_0_2_metric)


def num_global_features_not_0_2_metric(ctx, **kwargs):
    return np.mean(ctx.num_global_features_not_0_2_metric)


def avg_local_features_not_0_2_metric(ctx, **kwargs):
    return np.mean(ctx.avg_local_features_not_0_2_metric)


def num_local_features_not_0_2_metric(ctx, **kwargs):
    return np.mean(ctx.num_local_features_not_0_2_metric)


def call_num_local_features_not_0_3_metric(types):
    if "num_local_features_not_0_3_metric" in types:
        return "num_local_features_not_0_3_metric", num_local_features_not_0_3_metric


def call_avg_local_features_not_0_3_metric(types):
    if "avg_local_features_not_0_3_metric" in types:
        return "avg_local_features_not_0_3_metric", avg_local_features_not_0_3_metric


def call_num_global_features_not_0_3_metric(types):
    if "num_global_features_not_0_3_metric" in types:
        return "num_global_features_not_0_3_metric", num_global_features_not_0_3_metric


def call_avg_global_features_not_0_3_metric(types):
    if "avg_global_features_not_0_3_metric" in types:
        return "avg_global_features_not_0_3_metric", avg_global_features_not_0_3_metric


def call_num_global_combined_features_not_0_3_metric(types):
    if "num_global_combined_features_not_0_3_metric" in types:
        return "num_global_combined_features_not_0_3_metric", num_global_combined_features_not_0_3_metric


def call_avg_global_combined_features_not_0_3_metric(types):
    if "avg_global_combined_features_not_0_3_metric" in types:
        return "avg_global_combined_features_not_0_3_metric", avg_global_combined_features_not_0_3_metric


def call_num_features_global_local_3_metric(types):
    if "num_features_global_local_3_metric" in types:
        return "num_features_global_local_3_metric", num_features_global_local_3_metric


def call_cos_sim_local_global_combined_3_metric(types):
    if "cos_sim_local_global_combined_3_metric" in types:
        return "cos_sim_local_global_combined_3_metric", cos_sim_local_global_combined_3_metric


def call_cos_sim_global_local_combined_3_metric(types):
    if "cos_sim_global_local_combined_3_metric" in types:
        return "cos_sim_global_local_combined_3_metric", cos_sim_global_local_combined_3_metric


def call_cos_sim_global_global_combined_3_metric(types):
    if "cos_sim_global_global_combined_3_metric" in types:
        return "cos_sim_global_global_combined_3_metric", cos_sim_global_global_combined_3_metric


def call_cos_sim_local_local_combined_3_metric(types):
    if "cos_sim_local_local_combined_3_metric" in types:
        return "cos_sim_local_local_combined_3_metric", cos_sim_local_local_combined_3_metric


def call_num_fixed_features_not_0_metric(types):
    if "num_fixed_features_not_0_metric" in types:
        return "num_fixed_features_not_0_metric", num_fixed_features_not_0_metric

def call_avg_fixed_features_not_0_metric(types):
    if "avg_fixed_features_not_0_metric" in types:
        return "avg_fixed_features_not_0_metric", avg_fixed_features_not_0_metric

def call_num_features_local_fixed_metric(types):
    if "num_features_local_fixed_metric" in types:
        return "num_features_local_fixed_metric", num_features_local_fixed_metric

def call_cos_sim_local_global_metric(types):
    if "cos_sim_local_global_metric" in types:
        return "cos_sim_local_global_metric", cos_sim_local_global_metric

def call_cos_sim_global_combined_metric(types):
    if "cos_sim_global_combined_metric" in types:
        return "cos_sim_global_combined_metric", cos_sim_global_combined_metric

def call_cos_sim_local_combined_metric(types):
    if "cos_sim_local_combined_metric" in types:
        return "cos_sim_local_combined_metric", cos_sim_local_combined_metric

def call_cos_sim_global_fixed_metric(types):
    if "cos_sim_global_fixed_metric" in types:
        return "cos_sim_global_fixed_metric", cos_sim_global_fixed_metric

def call_cos_sim_local_fixed_metric(types):
    if "cos_sim_local_fixed_metric" in types:
        return "cos_sim_local_fixed_metric", cos_sim_local_fixed_metric

def num_fixed_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.num_fixed_features_not_0_metric)

def avg_fixed_features_not_0_metric(ctx, **kwargs):
    return np.mean(ctx.avg_fixed_features_not_0_metric)

def num_features_local_fixed_metric(ctx, **kwargs):
    return np.mean(ctx.num_features_local_fixed_metric)

def cos_sim_local_global_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_global_metric)

def cos_sim_global_combined_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_combined_metric)

def cos_sim_local_combined_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_combined_metric)

def cos_sim_global_fixed_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_global_fixed_metric)

def cos_sim_local_fixed_metric(ctx, **kwargs):
    return np.mean(ctx.cos_sim_local_fixed_metric)