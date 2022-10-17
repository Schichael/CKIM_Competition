import copy
import logging
from copy import deepcopy
import random
import torch
from torch_geometric.graphgym import optim

from federatedscope.core.auxiliaries.decorators import use_diff, use_diff_laplacian
from federatedscope.core.auxiliaries.eunms import MODE
from federatedscope.core.trainers import GeneralTorchTrainer
from federatedscope.gfl.trainer import GraphMiniBatchTrainer

logger = logging.getLogger(__name__)

class LaplacianTrainerNoGraph(GeneralTorchTrainer):
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

    def update(self, content, strict=False):
        """
            Called by the FL client to update the model parameters
        Arguments:
            model_parameters (dict): PyTorch Module object's state_dict.
        """
        new_model_params, new_omega = content

        trainable_parameters = self._param_filter(new_model_params)
        for key in trainable_parameters:
            self.ctx.model.state_dict()[key].data.copy_(new_model_params[key])
            if key in self.ctx.omega:
                self.ctx.omega[key] = copy.deepcopy(new_omega[key])
        #trainable_parameters = self._param_filter(model_parameters)
        #for key in trainable_parameters:
        #    self.ctx.model.state_dict()[key].data.copy_(trainable_parameters[key])s

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
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
        batch = ctx.data_batch
        pred = ctx.model(batch[0].to(ctx.device))
        # TODO: deal with the type of data within the dataloader or dataset
        if 'regression' in ctx.cfg.model.task.lower():
            label = batch[1].to(ctx.device)
        else:
            label = batch[1].squeeze(-1).long().to(ctx.device)
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.loss_batch_ce = ctx.criterion(pred, label)
        ctx.loss_batch = ctx.loss_batch_ce
        ctx.loss_batch_csd = self.get_csd_loss(ctx.new_mu, ctx.new_omega, ctx.cur_epoch_i + 1)

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

    def get_csd_loss(self, mu, omega, round_num):
        loss_set = []
        for name, param in self.ctx.model.named_parameters():
            theta = self.ctx.model.state_dict()[name]
            # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
            # omega_dropout[omega_dropout>0.5] = 1.0
            # omega_dropout[omega_dropout <= 0.5] = 0.0

            loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())
        return sum(loss_set)

    def _hook_on_batch_backward(self, ctx):
        ctx.optimizer.zero_grad()
        ctx.loss_batch_ce.backward(retain_graph=True)
        for name, param in ctx.model.named_parameters():
            if param.grad is not None:
                ctx.omega[name] += (len(ctx.data_batch[1]) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2
        ctx.optimizer.zero_grad()
        loss = ctx.loss_batch_ce + self.config.params.csd_importance * ctx.loss_batch_csd
        loss.backward(retain_graph=True)

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

        return self.ctx.cfg.params.alpha, self.get_model_para(
        ), self.get_omega_para(), self.ctx.eval_metrics

    def get_omega_para(self):
        return self._param_filter(
            self.ctx.omega)



def call_laplacian_trainer(trainer_type):
    if trainer_type == 'laplacian_trainer_no_graph':
        trainer_builder = LaplacianTrainerNoGraph
        return trainer_builder