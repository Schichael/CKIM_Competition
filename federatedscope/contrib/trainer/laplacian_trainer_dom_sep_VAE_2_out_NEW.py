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


class LaplacianDomainSeparationVAE_2Out_NEW_Trainer(GraphMiniBatchTrainer):
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
        self.tmp = 0
        self.kld_imp = 0.
        self.routine_steps = 3
        # Get all model parameters with reuqires_grad = True
        #for param in self.ctx.model.named_parameters():
        #    if param[0].startswith('fixed'):
        #        param[1].requires_grad = False

        self.grad_params = [param[0] for param in self.ctx.model.named_parameters() if param[1].requires_grad]

    def _align_global_local_parameters(self, model):
        for name, param in deepcopy(model).named_parameters():
            if name.startswith('local'):
                stripped_name = name[len('local'):]
                global_name = 'global' + stripped_name
                model.state_dict()[name].data.copy_(copy.deepcopy(model.state_dict()[global_name].data))
                if name in self.ctx.omega:
                    self.ctx.omega[name] = deepcopy(self.ctx.omega[global_name])


    def _align_interm_parameters(self, model):
        for name, param in deepcopy(model).named_parameters():
            if name.startswith('interm'):
                stripped_name = name[len('interm'):]
                global_name = 'global' + stripped_name
                model.state_dict()[name].data.copy_(copy.deepcopy(model.state_dict()[global_name].data))
                if name in self.ctx.omega:
                    self.ctx.omega[name] = deepcopy(self.ctx.omega[global_name])


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

        if self.first_round:
            self._align_global_local_parameters(self.ctx.model)
            self._align_interm_parameters(self.ctx.model)

        # trainable_parameters = self._param_filter(model_parameters)
        # for key in trainable_parameters:
        #    self.ctx.model.state_dict()[key].data.copy_(trainable_parameters[key])s

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
        ctx.acc_rec_loss = 0.
        #if not self.first_round:
        #    print(f"last round mean difference loss: {ctx.aggr_diff_loss/ctx.batchnumbers}")

        if ctx.cur_data_split == "train" and not self.in_finetune:
            #print("in train")
            self.round_num += 1
            self.in_finetune = True
            """
            self.kld_imp = self.config.params.kld_importance
            if self.round_num > 30 and self.round_num <= 39:
                self.kld_imp = self.config.params.kld_importance - (self.config.params.kld_importance / 10) * (self.round_num - 30)
            elif self.round_num>39:
                self.kld_imp = 0.1 * self.config.params.kld_importance
            
            """


        elif ctx.cur_data_split == "train" and self.in_finetune:
            self.in_finetune = False

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
        self.tmp += 1
        batch = ctx.data_batch.to(ctx.device)
        out_global, out_local_interm, kld_loss_encoder, kld_global, kld_interm, kld_local, rec_loss, diff_local_interm, sim_global_interm = ctx.model(batch, ctx.routine_step)

        #ctx.sim_interm_fixed = sim_interm_fixed

        # TODO: Not in metrics yet
        #ctx.sim_interm_fixed_metric = sim_interm_fixed.detach().item()

        ctx.kld_loss_encoder = kld_loss_encoder
        ctx.kld_loss_encoder_metric = kld_loss_encoder.detach().item()

        ctx.kld_global = kld_global
        ctx.kld_global_metric = kld_global.detach().item()

        ctx.kld_interm = kld_interm
        ctx.kld_interm_metric = kld_interm.detach().item()

        ctx.kld_local = kld_local
        ctx.kld_local_metric = kld_local.detach().item()

        ctx.rec_loss = rec_loss
        ctx.rec_loss_metric = rec_loss.detach().item()

        ctx.diff_local_interm = diff_local_interm
        ctx.diff_local_interm_metric = diff_local_interm.detach().item()

        ctx.sim_global_interm = sim_global_interm
        ctx.sim_global_interm_metric = sim_global_interm.detach().item()


        #print(f"diff_local_global: {diff_local_global}")
        #print(f"mi_global_fixed: {sim_global_fixed}")
        #print(f"rec_loss: {rec_loss}")
        #print(f"kld_loss: {kld_loss}")

        # print(f"negative mi: {-ctx.mi}")
        csd_loss = CSDLoss(self._param_filter, ctx)
        # TODO: deal with the type of data within the dataloader or dataset
        if 'regression' in ctx.cfg.model.task.lower():
            label = batch.y
        else:
            label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)

        ctx.loss_batch = ctx.criterion(out_global, label)
        ctx.loss_out_global = ctx.loss_batch
        ctx.loss_out_local_interm = ctx.criterion(out_local_interm, label)
        ctx.loss_out_local_interm_metric = ctx.loss_out_local_interm.detach().item()

        #ctx.loss_batch_csd = self.get_csd_loss(ctx.model.state_dict(), ctx.new_mu, ctx.new_omega, ctx.cur_epoch_i + 1)
        ctx.loss_batch_csd = csd_loss(ctx.model.state_dict(), ctx.new_mu,
                                      ctx.new_omega, self.round_num)
        ctx.loss_batch_csd_metric = ctx.loss_batch_csd.detach().item()
        #print(f"loss_batch_csd: {ctx.loss_batch_csd}")

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = out_global

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

    def _backward_step1_all(self, ctx):
        """ Only the global branch and node encoder for testing

        Args:
            ctx:

        Returns:

        """
        ctx.optimizer.zero_grad()

        # backward through the global branch
        for param in ctx.model.named_parameters():
            if not (param[0].startswith("global") or param[0].startswith("clf")) and not (param[0].startswith("encoder") or param[0].startswith("encoder_atom")) and not (param[0].startswith("decoder")):
                param[1].requires_grad = False

        loss = ctx.loss_out_global + 0*ctx.rec_loss + self.config.params.kld_global_imp * ctx.kld_global# + ctx.loss_out_global # + self.config.params.sim_global_interm_imp * ctx.sim_global_interm
        # loss = ctx.loss_out_global

        loss.backward(retain_graph=False)
        # Compute omega
        for name, param in ctx.model.named_parameters():
            if param.grad is not None and param.requires_grad is True:
                ctx.omega[name] += (len(ctx.data_batch.y) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2

        # loss = self.config.params.csd_interm_imp * ctx.loss_batch_csd
        # loss.backward(retain_graph=False)

        # freeze everything that is not local
        """
        for param in ctx.model.named_parameters():
            if not param[0].startswith("local"):
                param[1].requires_grad = False


        loss = ctx.loss_out_local_interm + self.config.params.kld_local_imp * ctx.kld_local + self.config.params.diff_local_imp * ctx.diff_local_interm

        loss.backward()
        """
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True


    def _backward_step1(self, ctx):
        """ Only train Noder Encoder through the global task loss and the KLD loss

        Args:
            ctx:

        Returns:

        """
        ctx.optimizer.zero_grad()
        # freeze all but node encoder
        for param in ctx.model.named_parameters():
            if not (param[0].startswith("encoder") or param[0].startswith("encoder_atom")):
                param[1].requires_grad = False

        loss = ctx.loss_out_global + self.config.params.kld_ne_imp * ctx.kld_loss_encoder
        loss.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True


    def _backward_step_2(self, ctx):
        """ Train the intermediate branch with the interm Task loss, decoder loss and KLD loss

        Args:
            ctx:

        Returns:

        """
        ctx.optimizer.zero_grad()
        # only ther interm layers
        for param in ctx.model.named_parameters():
            if not (param[0].startswith("interm")):
                param[1].requires_grad = False

        loss = ctx.loss_out_local_interm + self.config.params.kld_interm_imp * ctx.kld_interm + self.config.params.recon_imp * ctx.rec_loss + \
               self.config.params.diff_interm_imp * ctx.diff_local_interm

        loss.backward(retain_graph=True)

        # Train decoder
        for param in ctx.model.named_parameters():
            if not (param[0].startswith("decoder")):
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True

        loss = ctx.rec_loss

        loss.backward(retain_graph=True)

        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True

        # Compute omega
        for name, param in ctx.model.named_parameters():
            if param.grad is not None and param.requires_grad is True:
                ctx.omega[name] += (len(ctx.data_batch.y) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2

        # L_prior
        # only the interm layers
        for param in ctx.model.named_parameters():
            if not (param[0].startswith("interm") or param[0].startswith("decoder")):
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True

        loss = self.config.params.csd_imp * ctx.loss_batch_csd
        loss.backward()

        #loss = self.config.params.csd_interm_imp * ctx.loss_batch_csd
        #loss.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True


    def _backward_step_3(self, ctx):
        """ Train the local and global branch. Freeze everything else

        Args:
            ctx:

        Returns:

        """
        ctx.optimizer.zero_grad()

        # backward through the global branch
        for param in ctx.model.named_parameters():
            if not (param[0].startswith("global") or param[0].startswith("clf") or param[0].startswith("local")):
                param[1].requires_grad = False

        loss = ctx.loss_out_global + self.config.params.sim_global_interm_imp * ctx.sim_global_interm
        # loss = ctx.loss_out_global

        loss.backward(retain_graph=False)
        # Compute omega
        for name, param in ctx.model.named_parameters():
            if param.grad is not None and param.requires_grad is True:
                ctx.omega[name] += (len(ctx.data_batch.y) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2

        # Prior loss for linear layer
        #for param in ctx.model.named_parameters():
        #    if not (param[0].startswith("global_linear_out1")):
        #        param[1].requires_grad = False

        #loss = self.config.params.csd_imp * ctx.loss_batch_csd
        #loss.backward(retain_graph=False)

        # freeze everything that is not local
        """
        for param in ctx.model.named_parameters():
            if not param[0].startswith("local"):
                param[1].requires_grad = False


        loss = ctx.loss_out_local_interm + self.config.params.kld_local_imp * ctx.kld_local + self.config.params.diff_local_imp * ctx.diff_local_interm

        loss.backward()
        """
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True

    def _hook_on_batch_backward(self, ctx):
        # Get all model parameters with reuqires_grad = True
        # grad_params = [param[0] for param in ctx.model.named_parameters() if param[1].requires_grad]

        ctx.optimizer.zero_grad()

        if ctx.routine_step == 0:
            #self._backward_step1_all(ctx)
            self._backward_step1(ctx)
        elif ctx.routine_step == 1:
            self._backward_step_2(ctx)
        else:
            self._backward_step_3(ctx)




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
                    self.ctx.routine_step = None
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
        trainer_builder = LaplacianDomainSeparationVAE_2Out_NEW_Trainer
        return trainer_builder
