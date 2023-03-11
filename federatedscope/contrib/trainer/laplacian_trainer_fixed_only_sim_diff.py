import copy
import logging
import time
from copy import deepcopy

import torch
from torch_geometric.graphgym import optim
from federatedscope.core.auxiliaries.utils import param2tensor

from federatedscope.core.auxiliaries.decorators import use_diff, use_diff_laplacian
from federatedscope.core.auxiliaries.eunms import MODE
from federatedscope.gfl.trainer import GraphMiniBatchTrainer

logger = logging.getLogger(__name__)

class LaplacianTrainer_fixed_ONLY_SIM_DIFF(GraphMiniBatchTrainer):
    def __init__(self,
                 model,
                 omega,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):

        self.omega = omega
        super().__init__(model,
                 data,
                 device,
                 config,
                 only_for_eval,
                 monitor)

        self.ctx.omega = self.omega
        self.device = device
        self.config=config
        self.first_round = True
        self.in_finetune = False
        self.round_num = 0
        self.grad_params = [param[0] for param in self.ctx.model.named_parameters() if param[1].requires_grad]

    def _align_fixed_parameters(self, model):
        for name, param in deepcopy(model).named_parameters():
            if name.startswith('fixed'):
                stripped_name = name[len('fixed'):]
                global_name = 'global' + stripped_name
                model.state_dict()[name].data.copy_(copy.deepcopy(model.state_dict()[global_name].data))

    def _align_global_local_parameters(self, model):
        for name, param in deepcopy(model).named_parameters():
            if name.startswith('local'):
                stripped_name = name[len('local'):]
                global_name = 'global' + stripped_name
                model.state_dict()[name].data.copy_(copy.deepcopy(model.state_dict()[global_name].data))

    def update(self, content, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        new_model_params, new_omega = content
        model_params = copy.deepcopy(new_model_params)
        #print("model params: ")
        for key in model_params:
            #print(key)
            model_params[key] = param2tensor(model_params[key])
        #self.ctx.model.load_state_dict(self._param_filter(model_params),
        #                               strict=strict)

        trainable_parameters = self._param_filter(model_params)
        #print(f"trainable_parameters: \n{trainable_parameters.keys()}")
        #print(f"omega keys: \n{self.ctx.omega.keys()}")
        share_rate = 1.
        for key in trainable_parameters:

            #self.ctx.model.state_dict()[key].data.copy_(new_model_params[key])
            if key in self.ctx.omega:
                #print(f"sample_sizeshared: {key}")
                old_val = self.ctx.model.state_dict()[key]
                new_val = trainable_parameters[key]

                updated_val = old_val * (1-share_rate) + share_rate * new_val

                old_omega = self.ctx.omega[key]
                new_omega_ = new_omega[key]

                updated_omega = old_omega * (1-share_rate) + share_rate * new_omega_

                self.ctx.model.state_dict()[key].data.copy_(updated_val)
                self.ctx.omega[key] = copy.deepcopy(updated_omega)

        self._align_fixed_parameters(self.ctx.model)

        if self.first_round:
            self._align_global_local_parameters(self.ctx.model)

        #trainable_parameters = self._param_filter(model_parameters)
        #for key in trainable_parameters:
        #    self.ctx.model.state_dict()[key].data.copy_(trainable_parameters[key])s

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
        if ctx.cur_data_split == "train" and not self.in_finetune:
            self.round_num += 1
            self.in_finetune = True
        elif ctx.cur_data_split == "train" and self.in_finetune:
            self.in_finetune = False


        ctx.log_ce_loss = 0
        ctx.log_csd_loss = 0
        new_omega = dict()
        new_mu = dict()
        server_model_state_dict = ctx.model.state_dict()
        for name, param in ctx.model.named_parameters():
            # new_omega[name] = 1 / (1 - data_alpha) * (server_omega[name] - data_alpha * client_omega_set[client_idx][name])
            # new_mu[name] = 1 / (1 - data_alpha) * (server_omega[name] * server_model_state_dict[name] -
            #                 data_alpha * client_omega_set[client_idx][name] * client_model_set[client_idx].state_dict()[name]) /\
            #                (new_omega[name] + args.eps)
            new_omega[name] = deepcopy(ctx.omega[name])
            new_mu[name] = deepcopy(server_model_state_dict[name])
        ctx.new_omega = new_omega
        ctx.new_mu = new_mu

    def _hook_on_batch_forward(self, ctx):

        batch = ctx.data_batch.to(ctx.device)
        pred, kld_loss_encoder, diff_local_fixed, sim_global_fixed = ctx.model(batch, self.config.params.sim_loss)
        csd_loss = CSDLoss(self._param_filter, ctx)
        # TODO: deal with the type of data within the dataloader or dataset
        if 'regression' in ctx.cfg.model.task.lower():
            label = batch.y
        else:
            label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.kld_loss_encoder = kld_loss_encoder
        ctx.kld_loss_encoder_metric = kld_loss_encoder.detach().item()

        ctx.diff_local_fixed = diff_local_fixed
        ctx.diff_local_fixed_metric = diff_local_fixed.detach().item()

        ctx.sim_global_fixed = sim_global_fixed
        ctx.sim_global_fixed_metric = sim_global_fixed.detach().item()

        ctx.loss_batch_ce = ctx.criterion(pred, label)
        ctx.loss_batch = ctx.loss_batch_ce
        ctx.loss_batch_csd = csd_loss(ctx.model.state_dict(), ctx.new_mu,
                                      ctx.new_omega, self.round_num)
        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred

        # record the index of the ${MODE} samples
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )


    def _hook_on_batch_backward(self, ctx):

        ctx.optimizer.zero_grad()
        loss = ctx.loss_batch_ce + self.config.params.sim_imp * ctx.sim_global_fixed + self.config.params.diff_imp * ctx.diff_local_fixed
        loss.backward(retain_graph=False)
        for name, param in ctx.model.named_parameters():
            if param.grad is not None:
                ctx.omega[name] += (len(ctx.data_batch.y) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()


    @use_diff_laplacian
    def train(self, state:int, target_data_split_name="train", hooks_set=None):
        # state = round number
        self.ctx.state = state
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_data_split(target_data_split_name)

        self._run_routine(MODE.TRAIN, hooks_set, target_data_split_name)

        return self.ctx.num_samples_train, self.get_model_para(
        ), self.get_omega_para(), self.ctx.eval_metrics

    def get_omega_para(self):
        return self._param_filter(
            self.ctx.omega)


class CSDLoss(torch.nn.Module):
    def __init__(self, param_filter, ctx):
        super(CSDLoss, self).__init__()
        self._param_filter = param_filter
        self.ctx = ctx

    def forward(self, model_params, mu, omega, round_num):
        loss_set = []
        loss = None
        trainable_parameters = self._param_filter(model_params)
        for name in trainable_parameters:
            #if 'norm' in name:
            #    continue
            if name in omega:
                theta = None
                for param in self.ctx.model.named_parameters():
                    if param[0] == name:
                        theta = param[1]
                        break
                # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
                # omega_dropout[omega_dropout>0.5] = 1.0
                # omega_dropout[omega_dropout <= 0.5] = 0.0
                if loss is None:
                    loss = (0.5 / round_num) * (omega[name] * ((theta - mu[name]) **
                                                               2)).sum()
                else:
                    loss += (0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum()
                if theta is None:
                    print('theta is None')
                # loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return loss  # return sum(loss_set)


def call_laplacian_trainer(trainer_type):
    if trainer_type == 'laplacian_trainer':
        trainer_builder = LaplacianTrainer_fixed_ONLY_SIM_DIFF
        return trainer_builder