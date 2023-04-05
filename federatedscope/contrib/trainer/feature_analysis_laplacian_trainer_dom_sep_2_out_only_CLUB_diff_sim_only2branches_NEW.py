import copy
import logging
import os
from copy import deepcopy
from typing import Callable

import torch
from torch_geometric.graphgym import optim
from federatedscope.core.auxiliaries.utils import param2tensor

from federatedscope.core.auxiliaries.decorators import use_diff, use_diff_laplacian
from federatedscope.core.auxiliaries.eunms import MODE
from federatedscope.gfl.trainer import GraphMiniBatchTrainer

logger = logging.getLogger(__name__)


class FeatureAnalysisLaplacianDomainSeparationVAE_2Out_OnlyCLUBDiffSim_only2branches_NEW_Trainer(GraphMiniBatchTrainer):
    def __init__(self,
                 model,
                 omega,
                 data,
                 device,
                 config,
                 client_ID,
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
        self.tmp = 0
        self.kld_imp = 0.
        self.routine_steps = 3
        self.client_ID = client_ID
        self.ctx.diff_loss_1_metric = []
        self.ctx.diff_loss_2_metric = []
        # Get all model parameters with reuqires_grad = True
        #for param in self.ctx.model.named_parameters():
        #    if param[0].startswith('fixed'):
        #        param[1].requires_grad = False

        self.grad_params = [param[0] for param in self.ctx.model.named_parameters() if param[1].requires_grad]

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

        #self._align_interm_parameters(self.ctx.model)
        #self._align_global_fixed_parameters(self.ctx.model)

        # trainable_parameters = self._param_filter(model_parameters)
        # for key in trainable_parameters:
        #    self.ctx.model.state_dict()[key].data.copy_(trainable_parameters[key])s

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
        model_path = f"{self.config.params.model_path}/model{self.client_ID}_0.pth"
        state_dict = torch.load(model_path)['model']
        ctx.model.load_state_dict(state_dict)
        self.ctx.diff_loss_1_metric = []
        self.ctx.diff_loss_2_metric = []


    def _hook_on_batch_forward(self, ctx):
        self.tmp += 1
        batch = ctx.data_batch.to(ctx.device)
        out, kld_loss_encoder, diff_local_global, MI, x_local, x_global = ctx.model(batch)

        #ctx.sim_interm_fixed = sim_interm_fixed




        if 'regression' in ctx.cfg.model.task.lower():
            label = batch.y
        else:
            label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.loss_batch = ctx.criterion(out, label)

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = out

        ctx.loss_out_local_interm = ctx.criterion(out, label)
        ctx.loss_out_local_interm_metric = ctx.loss_out_local_interm.detach().item()

        dataset_name = self.ctx.dataset_name
        cur_batch_i = self.ctx.cur_batch_i
        out_dir = self.config.outdir

        path = out_dir + '/features/client_' + str(self.client_ID)

        if not os.path.exists(path):
            os.makedirs(path)


        # save outputs
        # local
        #  x_local_1, x_local_2, x_global_1, x_global_2

        file_name = path + '/x_local_' + dataset_name + '_' + str(cur_batch_i) + '.pt'
        torch.save(x_local, file_name)

        file_name = path + '/x_global_' + dataset_name + '_' + str(cur_batch_i) + '.pt'
        torch.save(x_global, file_name)


        # save labels
        # local
        file_name = path + '/' + dataset_name + '_' + str(cur_batch_i) + '_labels' + '.pt'
        torch.save(label, file_name)


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


    @use_diff_laplacian
    def train(self, state: int, target_data_split_name="train", hooks_set=None):
        # state = round number
        self.ctx.state = state
        hooks_set = hooks_set or self.hooks_in_train

        self.ctx.check_data_split(target_data_split_name)

        self._run_routine(MODE.TRAIN, hooks_set, target_data_split_name)
        self.first_round = False

        return self.ctx.num_samples_train, self.get_model_para(
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
        for key in self.grad_params:
            # self.ctx.model.state_dict()[key].data.copy_(new_model_params[key])
            if key.startswith('local'):
                stripped_key = key[len('local'):]
                fixed_key = 'fixed' + stripped_key
                if fixed_key not in self.ctx.model.state_dict():
                    continue
                fixed_val = self.ctx.model.state_dict()[fixed_key]
                local_val = model_params[key]
                prox_loss_change[key] = - lr * lam * (local_val - fixed_val)
                #updated_val = local_val - lr * lam * (local_val - server_val)
                #self.ctx.model.state_dict()[key].data.copy_(updated_val)
        return prox_loss_change



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
        self.ctx.dataset_name = dataset_name
        self.ctx.dataset_mode = mode
        self.ctx.append_mode(mode)
        self.ctx.track_used_dataset(dataset_name)

        for hook in hooks_set["on_fit_start"]:
            hook(self.ctx)

        self.ctx.total_epochs = self.ctx.get("num_{}_epoch".format(dataset_name))

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

class DiffLoss(torch.nn.Module):
    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss





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
        trainer_builder = FeatureAnalysisLaplacianDomainSeparationVAE_2Out_OnlyCLUBDiffSim_only2branches_NEW_Trainer
        return trainer_builder