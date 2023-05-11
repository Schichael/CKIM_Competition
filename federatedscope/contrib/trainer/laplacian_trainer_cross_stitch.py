import copy
import logging
import os
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
from torch_geometric.graphgym import optim
from federatedscope.core.auxiliaries.utils import param2tensor

from federatedscope.core.auxiliaries.decorators import use_diff, use_diff_laplacian
from federatedscope.core.auxiliaries.eunms import MODE
from federatedscope.gfl.trainer import GraphMiniBatchTrainer

logger = logging.getLogger(__name__)


class \
        Laplacian_Cross_Stitch_Trainer(
    GraphMiniBatchTrainer):
    def __init__(self,
                 model,
                 omega,
                 data,
                 device,
                 clientID,
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
        self.clientID = clientID
        self.device = device
        self.config = config
        self.first_round = True
        self.round_num=0
        self.in_finetune = False
        self.tmp = 0
        self.kld_imp = 0.
        self.routine_steps = 3
        self.cos_sim = torch.nn.CosineSimilarity()
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
                try:
                    model.state_dict()[name].data.copy_(
                        copy.deepcopy(model.state_dict()[global_name].data))
                except:
                    pass


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

        # trainable_parameters = self._param_filter(model_parameters)
        # for key in trainable_parameters:
        #    self.ctx.model.state_dict()[key].data.copy_(trainable_parameters[key])s

    def _hook_on_fit_start_init(self, ctx):
        super()._hook_on_fit_start_init(ctx)
        setattr(ctx, "{}_y_inds".format(ctx.cur_data_split), [])
        ctx.acc_rec_loss = 0.
        #if not self.first_round:
        #    print(f"last round mean difference loss: {ctx.aggr_diff_loss/ctx.batchnumbers}")
        ctx.kld_loss_encoder_metric = []
        ctx.loss_batch_csd_metric = []
        ctx.diff_local_global_metric = []
        ctx.diff_local_local_out_metric = []
        ctx.loss_global_clf_metric = []


        ##################################################################################
        ctx.num_local_features_not_0_1_metric = []
        ctx.avg_local_features_not_0_1_metric = []
        ctx.num_global_features_not_0_1_metric = []
        ctx.avg_global_features_not_0_1_metric = []
        ctx.num_global_combined_features_not_0_1_metric = []
        ctx.avg_global_combined_features_not_0_1_metric = []
        ctx.num_features_global_local_1_metric = []
        ctx.cos_sim_local_global_combined_1_metric = []
        ctx.cos_sim_global_global_combined_1_metric = []
        ctx.cos_sim_local_local_combined_1_metric = []
        ctx.cos_sim_global_local_combined_1_metric = []


        ctx.num_local_features_not_0_2_metric = []
        ctx.avg_local_features_not_0_2_metric = []
        ctx.num_global_features_not_0_2_metric = []
        ctx.avg_global_features_not_0_2_metric = []
        ctx.num_global_combined_features_not_0_2_metric = []
        ctx.avg_global_combined_features_not_0_2_metric = []
        ctx.num_features_global_local_2_metric = []
        ctx.cos_sim_local_global_combined_2_metric = []
        ctx.cos_sim_global_global_combined_2_metric = []
        ctx.cos_sim_local_local_combined_2_metric = []
        ctx.cos_sim_global_local_combined_2_metric = []

        ctx.num_local_features_not_0_3_metric = []
        ctx.avg_local_features_not_0_3_metric = []
        ctx.num_global_features_not_0_3_metric = []
        ctx.avg_global_features_not_0_3_metric = []
        ctx.num_global_combined_features_not_0_3_metric = []
        ctx.avg_global_combined_features_not_0_3_metric = []
        ctx.num_features_global_local_3_metric = []
        ctx.cos_sim_local_global_combined_3_metric = []
        ctx.cos_sim_global_global_combined_3_metric = []
        ctx.cos_sim_local_local_combined_3_metric = []
        ctx.cos_sim_global_local_combined_3_metric = []


        ###################################################################################


        ctx.diff_1_metric = []
        ctx.diff_2_metric = []
        ctx.diff_3_metric = []

        if ctx.cur_data_split == "train":
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

        if self.round_num < 2:
            self._align_global_local_parameters(ctx.model)

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


    def stats_calculator(self, local_features_orig, global_features_orig,
                         global_combined_features_orig, local_combined_features_orig):
        local_features = local_features_orig.cpu().detach().numpy()
        global_features = global_features_orig.cpu().detach().numpy()
        global_combined_features = global_combined_features_orig.cpu().detach().numpy()
        local_combined_features = local_combined_features_orig.cpu().detach().numpy()

        num_total_features_curr = local_features.shape[0] * local_features.shape[1]

        mult_local_global = local_features * global_features

        num_local_features_not_0 = np.sum(local_features != 0) / num_total_features_curr
        avg_local_features_not_0 = local_features.sum() / np.sum(local_features != 0)



        num_global_features_not_0 = np.sum(global_features != 0) / \
                                    num_total_features_curr

        #print(f"np.sum(local_out_features != 0): {np.sum(local_out_features != 0)}")

        #if x == 0:
            #print("sum of global features is 0")
        avg_global_features_not_0 = global_features.sum() / np.sum(global_features != 0)
        num_global_combined_features_not_0 = np.sum(global_combined_features != 0) / \
                                      num_total_features_curr
        avg_global_combined_features_not_0 = global_combined_features.sum() / np.sum(
            global_combined_features != 0)

        num_local_combined_features_not_0 = np.sum(local_combined_features != 0) / \
                                      num_total_features_curr
        avg_local_combined_features_not_0 = local_combined_features.sum() / np.sum(
            local_combined_features != 0)

        num_features_global_local = np.sum(mult_local_global != 0) / num_total_features_curr
        with torch.no_grad():
            cos_sim_local_global_combined = self.cos_sim(local_features_orig,
                                                         global_combined_features_orig).mean()
            cos_sim_global_global_combined = self.cos_sim(global_features_orig,
                                                          global_combined_features_orig).mean()
            cos_sim_local_local_combined = self.cos_sim(local_features_orig,
                                                         local_combined_features_orig).mean()
            cos_sim_global_local_combined = self.cos_sim(global_features_orig,
                                                          local_combined_features_orig).mean()

        return num_local_features_not_0, avg_local_features_not_0, \
            num_global_features_not_0, avg_global_features_not_0, \
            num_global_combined_features_not_0, avg_global_combined_features_not_0, \
            num_features_global_local, cos_sim_local_global_combined.cpu().detach(), \
            cos_sim_global_global_combined.cpu().detach(), cos_sim_local_local_combined.cpu().detach(), \
            cos_sim_global_local_combined.cpu().detach()

    def _hook_on_batch_forward(self, ctx):
        self.tmp += 1
        batch = ctx.data_batch.to(ctx.device)
        x_out, local_alpha_1, local_alpha_2, local_alpha_3, diff_1, \
            diff_2, diff_3, x_local_1, x_local_2, x_local_3, x_global_1, \
            x_global_2, x_global_3, x_global_1_cs, x_global_2_cs, x_global_3_cs, \
            x_local_1_cs, x_local_2_cs, x_local_3_cs = \
            ctx.model(batch)

        #out_global_local, out_local_interm, diff_local_interm, diff_local_local_out, sim_local_out_interm, x_local, \
        #x_interm, x_local_out

        local_alpha_1_global = torch.nn.functional.softmax(local_alpha_1[0])[0]
        local_alpha_1_local = torch.nn.functional.softmax(local_alpha_1[1], 0)[1]
        local_alpha_2_global = torch.nn.functional.softmax(local_alpha_2[0], 0)[0]
        local_alpha_2_local = torch.nn.functional.softmax(local_alpha_2[1], 0)[1]
        local_alpha_3_global = torch.nn.functional.softmax(local_alpha_3[0], 0)[0]
        local_alpha_3_local = torch.nn.functional.softmax(local_alpha_3[1], 0)[1]

        ctx.local_alpha_1_global_metric = local_alpha_1_global.detach().item()
        ctx.local_alpha_1_local_metric = local_alpha_1_local.detach().item()
        ctx.local_alpha_2_global_metric = local_alpha_2_global.detach().item()
        ctx.local_alpha_2_local_metric = local_alpha_2_local.detach().item()
        ctx.local_alpha_3_global_metric = local_alpha_3_global.detach().item()
        ctx.local_alpha_3_local_metric = local_alpha_3_local.detach().item()

        ctx.diff_1_metric.append(diff_1.detach().item())
        ctx.diff_2_metric.append(diff_2.detach().item())
        ctx.diff_3 = diff_3
        ctx.diff_3_metric.append(diff_3.detach().item())


        ctx.x_local_1 = x_local_1
        ctx.x_global_1 = x_global_1
        ctx.x_global_1_cs = x_global_1_cs



        #ctx.sim_interm_fixed = sim_interm_fixed

        # TODO: Not in metrics yet
        #ctx.sim_interm_fixed_metric = sim_interm_fixed.detach().item()


        """
        ctx.kld_loss_encoder_metric
        ctx.loss_batch_csd_metric
        ctx.diff_local_global_metric
        """
        #ctx.kld_global = kld_global
        #ctx.kld_global_metric = kld_global.detach().item()

        #ctx.kld_interm = kld_interm
        #ctx.kld_interm_metric = kld_interm.detach().item()

        #ctx.kld_local = kld_local
        #ctx.kld_local_metric = kld_local.detach().item()

        #ctx.rec_loss = rec_loss
        #ctx.rec_loss_metric = rec_loss.detach().item()


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

        ctx.loss_batch = ctx.criterion(x_out, label)


        ctx.loss_out = ctx.loss_batch

        #ctx.loss_batch_csd = self.get_csd_loss(ctx.model.state_dict(), ctx.new_mu, ctx.new_omega, ctx.cur_epoch_i + 1)
        ctx.loss_batch_csd = csd_loss(ctx.model.state_dict(), ctx.new_mu,
                                      ctx.new_omega, self.round_num)
        ctx.loss_batch_csd_metric.append(ctx.loss_batch_csd.detach().item())
        #print(f"loss_batch_csd: {ctx.loss_batch_csd}")

        ctx.batch_size = len(label)
        ctx.y_true = label
        ctx.y_prob = x_out

        """
        ctx.x_local = x_local
        ctx.x_interm = x_interm
        ctx.x_local_out = x_local_out
        """



        # stats calc
        num_local_features_not_0, avg_local_features_not_0, \
            num_global_features_not_0, avg_global_features_not_0, \
            num_global_combined_features_not_0, avg_global_combined_features_not_0, \
            num_features_global_local, cos_sim_local_global_combined, \
            cos_sim_global_global_combined, cos_sim_local_local_combined, \
            cos_sim_global_local_combined = \
            self.stats_calculator(x_local_1, x_global_1, x_global_1_cs, x_local_1_cs)




        ctx.num_local_features_not_0_1_metric.append(num_local_features_not_0)
        ctx.avg_local_features_not_0_1_metric.append(avg_local_features_not_0)
        ctx.num_global_features_not_0_1_metric.append(num_global_features_not_0)
        ctx.avg_global_features_not_0_1_metric.append(avg_global_features_not_0)
        ctx.num_global_combined_features_not_0_1_metric.append(num_global_combined_features_not_0)
        ctx.avg_global_combined_features_not_0_1_metric.append(avg_global_combined_features_not_0)
        ctx.num_features_global_local_1_metric.append(num_features_global_local)
        ctx.cos_sim_local_global_combined_1_metric.append(cos_sim_local_global_combined)
        ctx.cos_sim_global_global_combined_1_metric.append(
            cos_sim_global_global_combined)
        ctx.cos_sim_local_local_combined_1_metric.append(
            cos_sim_local_local_combined)
        ctx.cos_sim_global_local_combined_1_metric.append(
            cos_sim_global_local_combined)

        num_local_features_not_0, avg_local_features_not_0, \
            num_global_features_not_0, avg_global_features_not_0, \
            num_global_combined_features_not_0, avg_global_combined_features_not_0, \
            num_features_global_local, cos_sim_local_global_combined, \
            cos_sim_global_global_combined, cos_sim_local_local_combined, \
            cos_sim_global_local_combined = \
            self.stats_calculator(x_local_2, x_global_2, x_global_2_cs, x_local_2_cs)

        ctx.num_local_features_not_0_2_metric.append(num_local_features_not_0)
        ctx.avg_local_features_not_0_2_metric.append(avg_local_features_not_0)
        ctx.num_global_features_not_0_2_metric.append(num_global_features_not_0)
        ctx.avg_global_features_not_0_2_metric.append(avg_global_features_not_0)
        ctx.num_global_combined_features_not_0_2_metric.append(
            num_global_combined_features_not_0)
        ctx.avg_global_combined_features_not_0_2_metric.append(
            avg_global_combined_features_not_0)
        ctx.num_features_global_local_2_metric.append(num_features_global_local)
        ctx.cos_sim_local_global_combined_2_metric.append(cos_sim_local_global_combined)
        ctx.cos_sim_global_global_combined_2_metric.append(
            cos_sim_global_global_combined)
        ctx.cos_sim_local_local_combined_2_metric.append(
            cos_sim_local_local_combined)
        ctx.cos_sim_global_local_combined_2_metric.append(
            cos_sim_global_local_combined)

        num_local_features_not_0, avg_local_features_not_0, \
            num_global_features_not_0, avg_global_features_not_0, \
            num_global_combined_features_not_0, avg_global_combined_features_not_0, \
            num_features_global_local, cos_sim_local_global_combined, \
            cos_sim_global_global_combined, cos_sim_local_local_combined, \
            cos_sim_global_local_combined = \
            self.stats_calculator(x_local_3, x_global_3, x_global_3_cs, x_local_3_cs)

        ctx.num_local_features_not_0_3_metric.append(num_local_features_not_0)
        ctx.avg_local_features_not_0_3_metric.append(avg_local_features_not_0)
        ctx.num_global_features_not_0_3_metric.append(num_global_features_not_0)
        ctx.avg_global_features_not_0_3_metric.append(avg_global_features_not_0)
        ctx.num_global_combined_features_not_0_3_metric.append(
            num_global_combined_features_not_0)
        ctx.avg_global_combined_features_not_0_3_metric.append(
            avg_global_combined_features_not_0)
        ctx.num_features_global_local_3_metric.append(num_features_global_local)
        ctx.cos_sim_local_global_combined_3_metric.append(cos_sim_local_global_combined)
        ctx.cos_sim_global_global_combined_3_metric.append(
            cos_sim_global_global_combined)
        ctx.cos_sim_local_local_combined_3_metric.append(
            cos_sim_local_local_combined)
        ctx.cos_sim_global_local_combined_3_metric.append(
            cos_sim_global_local_combined)


        if self.round_num == 0 or self.round_num == 1 or self.round_num == 498 or \
                self.round_num == 998:
            dataset_name = self.ctx.dataset_name
            cur_batch_i = self.ctx.cur_batch_i
            out_dir = self.config.outdir

            path = out_dir + '/features/client_' + str(self.clientID)

            if not os.path.exists(path):
                os.makedirs(path)


            # local
                # local
                file_name = path + '/local_1' + dataset_name + '_' + str(cur_batch_i) + \
                            '.pt'
                torch.save(x_local_1, file_name)

                # global
                file_name = path + '/global_1' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_global_1, file_name)

                # global_cs
                file_name = path + '/global_1_cs_' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_global_1_cs, file_name)

                file_name = path + '/local_1_cs_' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_local_1_cs, file_name)


                file_name = path + '/local_2' + dataset_name + '_' + str(cur_batch_i) + \
                            '.pt'
                torch.save(x_local_2, file_name)

                # global
                file_name = path + '/global_2' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_global_2, file_name)

                file_name = path + '/global_2_cs_' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_global_2_cs, file_name)

                file_name = path + '/local_2_cs_' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_local_2_cs, file_name)


                file_name = path + '/local_3' + dataset_name + '_' + str(cur_batch_i) + \
                            '.pt'
                torch.save(x_local_3, file_name)

                # global
                file_name = path + '/global_3' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_global_3, file_name)

                file_name = path + '/global_3_cs_' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_global_3_cs, file_name)

                file_name = path + '/local_3_cs_' + dataset_name + '_' + str(
                    cur_batch_i) + '.pt'
                torch.save(x_local_3_cs, file_name)


            # alphas
            file_name = path + '/local_alpha_1_' + dataset_name + '_' + str(
                cur_batch_i) + '.pt'
            torch.save(local_alpha_1, file_name)

            file_name = path + '/local_alpha_2_' + dataset_name + '_' + str(
                cur_batch_i) + '.pt'
            torch.save(local_alpha_2, file_name)

            file_name = path + '/local_alpha_3_' + dataset_name + '_' + str(
                cur_batch_i) + '.pt'
            torch.save(local_alpha_3, file_name)

            # save labels
            # local
            file_name = path + '/' + dataset_name + '_' + str(
                cur_batch_i) + '_labels' + '.pt'
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




    def _backward_step(self, ctx):
        """ Train the local and global branch. Freeze everything else

        Args:
            ctx:

        Returns:

        """
        ctx.optimizer.zero_grad()

        """
        ctx.loss_out_local_interm = ctx.criterion(out_local_interm, label)
        ctx.loss_out_interm = ctx.criterion(out_interm, label)
        """

        # loss_out_local_interm loss_out_local_local_out

        # loss for output and KLD
        loss = ctx.loss_out + self.config.params.diff_imp * \
               ctx.diff_3
        loss.backward(retain_graph=True)
        try:
            ctx.model.local_alpha_3.grad = ctx.model.local_alpha_3.grad * 10
            ctx.model.local_alpha_2.grad = ctx.model.local_alpha_2.grad * 10
            ctx.model.local_alpha_1.grad = ctx.model.local_alpha_1.grad * 100
        except:
            pass

        # Reset requires_grad
        for param in ctx.model.named_parameters():
            if param[0] in self.grad_params:
                param[1].requires_grad = True

        # Compute omega
        for name, param in ctx.model.named_parameters():
            if param.grad is not None and param.requires_grad is True:
                ctx.omega[name] += (len(ctx.data_batch.y) / len(
                    ctx.data['train'].dataset)) * param.grad.data.clone() ** 2

        loss = self.config.params.csd_imp * ctx.loss_batch_csd
        loss.backward()


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

        self._backward_step(ctx)


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
        trainer_builder = Laplacian_Cross_Stitch_Trainer
        return trainer_builder
