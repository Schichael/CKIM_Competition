import copy
import logging
from copy import deepcopy
from typing import Callable

import torch
from torch_geometric.graphgym import optim
from federatedscope.core.auxiliaries.utils import param2tensor

from federatedscope.core.auxiliaries.decorators import use_diff, use_diff_laplacian
from federatedscope.core.auxiliaries.eunms import MODE
from federatedscope.gfl.trainer import GraphMiniBatchTrainer

logger = logging.getLogger(__name__)


class LaplacianDiffGlobalOutTrainer(GraphMiniBatchTrainer):
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
        self.config = config
        self.first_round = True
        self.round_num=0
        self.in_finetune = False
        self.routine_steps = 2
        self.grad_params = [param[0] for param in self.ctx.model.named_parameters() if param[1].requires_grad]


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
        # print("model params: ")
        for key in model_params:
            # print(key)
            model_params[key] = param2tensor(model_params[key])
        # self.ctx.model.load_state_dict(self._param_filter(model_params),
        #                               strict=strict)

        trainable_parameters = self._param_filter(model_params)
        # print(f"trainable_parameters: \n{trainable_parameters.keys()}")
        # print(f"omega keys: \n{self.ctx.omega.keys()}")
        share_rate = 1.
        for key in trainable_parameters:

            # self.ctx.model.state_dict()[key].data.copy_(new_model_params[key])
            if key in self.ctx.omega:
                # print(f"shared: {key}")
                #old_val = self.ctx.model.state_dict()[key]
                #new_val =

                updated_val = trainable_parameters[key]

                old_omega = self.ctx.omega[key]
                new_omega_ = new_omega[key]

                updated_omega = new_omega_

                self.ctx.model.state_dict()[key].data.copy_(updated_val)
                self.ctx.omega[key] = copy.deepcopy(updated_omega)

        if self.first_round:
            self._align_global_local_parameters(self.ctx.model)
        # trainable_parameters = self._param_filter(model_parameters)
        # for key in trainable_parameters:
        #    self.ctx.model.state_dict()[key].data.copy_(trainable_parameters[key])s

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
        #if not self.first_round:
        #    print(f"last round mean difference loss: {ctx.aggr_diff_loss/ctx.batchnumbers}")
        ctx.aggr_diff_loss = 0
        ctx.batchnumbers = 0
        if ctx.cur_data_split == "train" and not self.in_finetune:
            self.round_num += 1

            self.in_finetune = True
        elif ctx.cur_data_split == "train" and self.in_finetune:
            self.in_finetune = False
        print(f"round number: {self.round_num}")
        ctx.log_ce_loss = 0
        ctx.log_csd_loss = 0
        new_omega = dict()
        new_mu = dict()
        server_model_state_dict = ctx.model.state_dict()
        i=0
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
        pred_global, pred_mixed, ctx.diff_loss = ctx.model(batch)
        csd_loss = CSDLoss(self._param_filter, ctx)
        # TODO: deal with the type of data within the dataloader or dataset
        if 'regression' in ctx.cfg.model.task.lower():
            label = batch.y
        else:
            label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.aggr_diff_loss += ctx.diff_loss.item()
        if ctx.routine_step == 0 or ctx.routine_step is None:
            ctx.loss_batch_ce = ctx.criterion(pred_global, label)
            ctx.loss_batch = ctx.loss_batch_ce
        else:
            ctx.loss_batch_ce = ctx.criterion(pred_mixed, label)
            ctx.loss_batch = ctx.loss_batch_ce
        #ctx.loss_batch_csd = self.get_csd_loss(ctx.model.state_dict(), ctx.new_mu, ctx.new_omega, ctx.cur_epoch_i + 1)
        ctx.loss_batch_csd = csd_loss(ctx.model.state_dict(), ctx.new_mu,
                                      ctx.new_omega, self.round_num)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = pred_global

        # record the index of the ${MODE} samples
        if hasattr(ctx.data_batch, 'data_index'):
            setattr(
                ctx,
                f'{ctx.cur_data_split}_y_inds',
                ctx.get(f'{ctx.cur_data_split}_y_inds') + [batch[_].data_index.item() for _ in range(len(label))]
            )

    def get_csd_loss(self, model_params, mu, omega, round_num):
        loss_set = []
        trainable_parameters = self._param_filter(model_params)
        for name in trainable_parameters:
            if name in omega:
                theta = self.ctx.model.state_dict()[name]

                # omega_dropout = torch.rand(omega[name].size()).cuda() if cuda else torch.rand(omega[name].size())
                # omega_dropout[omega_dropout>0.5] = 1.0
                # omega_dropout[omega_dropout <= 0.5] = 0.0

                loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())

        return sum(loss_set)

    def _hook_on_batch_backward(self, ctx):
        """
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()
        """
        ctx.optimizer.zero_grad()

        # disable weights to not train
        # if 1. routine step: freeze local weights. Else freeze all that are not local
        if self.ctx.routine_step == 0:
            for param in ctx.model.named_parameters():
                if param[0].startswith("local"):
                    param[1].requires_grad = False
        elif self.ctx.routine_step == 1:
            for param in ctx.model.named_parameters():
                if not param[0].startswith("local"):
                    param[1].requires_grad = False

        ctx.loss_batch_ce.backward(retain_graph=True)
        for name, param in ctx.model.named_parameters():
            if param.grad is not None:
                ctx.omega[name] += (len(ctx.data_batch.y) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2

        # Freeze Node Encoder such that it is not influenced by diff loss. Routine step does not matter.
        for param in ctx.model.named_parameters():
            if param[0].startswith("encoder") or param[0].startswith("encoder_atom"):
                param[1].requires_grad = False

        if self.ctx.routine_step == 0 or (self.ctx.routine_step == 1 and self.config.params.useDiffDuringLocal):
            loss = self.config.params.csd_importance * ctx.loss_batch_csd + self.config.params.diff_importance * ctx.diff_loss
            loss.backward(retain_graph=False)
        elif self.ctx.routine_step == 1 and self.config.params.useDiffDuringLocal:
            loss = self.config.params.diff_importance * ctx.diff_loss
            loss.backward(retain_graph=False)

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        # prox_loss_change = self.get_prox_loss_change()
        ctx.optimizer.step()
        """
        for key, param in prox_loss_change.items():
            curr_val = self.ctx.model.state_dict()[key]
            updated_val = curr_val + param
            self.ctx.model.state_dict()[key].data.copy_(updated_val)
        """
        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True

        #self.prox_loss_model_update()

    def _run_routine(self, mode, hooks_set, dataset_name=None):
        """Run the hooks_set and maintain the mode

        Arguments:
            mode (str): running mode of client, chosen from train/test
            hooks_set (dict): functions to be executed.
            dataset_name (str): which split.

        Note:
            Considering evaluation could be in ```hooks_set[
            "on_epoch_end"]```, there could be two data loaders in
        self.ctx, we must tell the running hooks which data_loader to call
        and which num_samples to count

        """
        if dataset_name is None:
            dataset_name = mode
        self.ctx.append_mode(mode)
        self.ctx.track_used_dataset(dataset_name)
        self.ctx.routine_step = None
        for hook in hooks_set["on_fit_start"]:
            hook(self.ctx)

        for epoch_i in range(self.ctx.get(
                "num_{}_epoch".format(dataset_name))):
            self.ctx.cur_epoch_i = epoch_i
            for hook in hooks_set["on_epoch_start"]:
                hook(self.ctx)

            for batch_i in range(
                    self.ctx.get("num_{}_batch".format(dataset_name))):
                self.ctx.cur_batch_i = batch_i
                for hook in hooks_set["on_batch_start"]:
                    hook(self.ctx)

                if self.ctx.cur_mode == 'train':
                    for routine_step in range(self.routine_steps):
                        self.ctx.routine_step = routine_step
                        for hook in hooks_set["on_batch_forward"]:
                            hook(self.ctx)
                        for hook in hooks_set["on_batch_backward"]:
                            hook(self.ctx)
                    self.ctx.routine_step = None
                else:
                    for hook in hooks_set["on_batch_forward"]:
                        hook(self.ctx)


                for hook in hooks_set["on_batch_end"]:
                    hook(self.ctx)

                # Break in the final epoch
                if self.ctx.cur_mode == 'train' and epoch_i == \
                        self.ctx.num_train_epoch - 1:
                    if batch_i >= self.ctx.num_train_batch_last_epoch - 1:
                        break

            for hook in hooks_set["on_epoch_end"]:
                hook(self.ctx)
        for hook in hooks_set["on_fit_end"]:
            hook(self.ctx)

        self.ctx.pop_mode()
        self.ctx.reset_used_dataset()
        # Avoid memory leak
        if not self.cfg.federate.share_local_model:
            if torch is None:
                pass
            else:
                self.ctx.model.to(torch.device("cpu"))


    @use_diff_laplacian
    def train(self, state: int, target_data_split_name="train", hooks_set=None):
        # state = round number
        self.ctx.state = state
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_data_split(target_data_split_name)

        self._run_routine(MODE.TRAIN, hooks_set, target_data_split_name)
        self.first_round = False

        return self.ctx.cfg.params.alpha, self.get_model_para(
        ), self.get_omega_para(), self.ctx.eval_metrics

    def get_omega_para(self):
        return self._param_filter(
            self.ctx.omega)

    def get_prox_loss_change(self):
        """Add the model update for the proximate loss term.

        Args:
            ctx:
            lam:

        Returns:

        """
        lam = self.config.params.lam
        lr = self.config.train.optimizer.lr
        model_params = copy.deepcopy(self.ctx.new_mu)
        # print("model params: ")
        for key in model_params:
            # print(key)
            model_params[key] = param2tensor(model_params[key])
        # self.ctx.model.load_state_dict(self._param_filter(model_params),
        #                               strict=strict)

        trainable_parameters = self._param_filter(model_params)
        # print(f"trainable_parameters: \n{trainable_parameters.keys()}")
        # print(f"omega keys: \n{self.ctx.omega.keys()}")
        prox_loss_change = {}
        for key in trainable_parameters:

            # self.ctx.model.state_dict()[key].data.copy_(new_model_params[key])
            if key.startswith('global'):
                stripped_key = key[len('global'):]
                local_key = 'local' + stripped_key
                local_val = self.ctx.model.state_dict()[local_key]
                server_val = model_params[key]
                prox_loss_change[local_key] = - lr * lam * (local_val - server_val)
                #updated_val = local_val - lr * lam * (local_val - server_val)
                #self.ctx.model.state_dict()[key].data.copy_(updated_val)
        return prox_loss_change


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
            if name in omega:
                theta = None
                for param in self.ctx.model.named_parameters():
                    if param[0] == name:
                        theta = param[1]
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
        trainer_builder = LaplacianDiffGlobalOutTrainer
        return trainer_builder
