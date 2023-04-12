import copy
import logging
from copy import deepcopy

import torch

from federatedscope.contrib.trainer\
    .laplacian_trainer_dom_sep_1_out_only_CLUB_diff_no_global_NEW_further import \
    LaplacianDomainSeparation_1Out_OnlyCLUBDiff_noGlobal_Further_NEW_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_1_out_only_CLUB_diff_no_global_NEW import \
    LaplacianDomainSeparation_1Out_OnlyCLUBDiff_noGlobal_NEW_Trainer
from federatedscope.contrib.trainer\
    .laplacian_trainer_dom_sep_1_out_only_diff_no_global_NEW import \
    LaplacianDomainSeparationVAE_1Out_OnlyDiff_noGlobal_NEW_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_1_out_only_diff_proxLoss_NEW import \
    LaplacianDomainSeparationVAE_1Out_OnlyDiffProxLoss_NEW_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_2_out_only_diff_proxLoss_NEW import \
    LaplacianDomainSeparationVAE_2Out_OnlyDiffProxLoss_NEW_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_2_out_only_diff_sim_NEW import \
    LaplacianDomainSeparationVAE_2Out_OnlyDiffSim_NEW_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_VAE_1_out import LaplacianDomainSeparationVAE_1Out_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_VAE_2_out import LaplacianDomainSeparationVAE_2Out_Trainer
from federatedscope.contrib.trainer.laplacian_trainer_dom_sep_VAE_2_out_NEW import \
    LaplacianDomainSeparationVAE_2Out_NEW_Trainer

from federatedscope.contrib.trainer.laplacian_trainer_with_domain_separation_with_summation import \
    LaplacianDomainSeparationWithSummationTrainer
from federatedscope.contrib.workers.client import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class LaplacianDomainSeparationVAE_1_out_onlyCLUB_Diff_noGlobalFurther_NEW_Client(
    Client):
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        self.alpha = config.params.alpha
        self.omega_set = self._set_init_omega(model, device)
        #self._align_global_local_parameters(model)

        trainer = LaplacianDomainSeparation_1Out_OnlyCLUBDiff_noGlobal_Further_NEW_Trainer(
            model=model,
            omega=self.omega_set,
            data=data,
            device=device,
            config=config,
            client_ID=ID,
            only_for_eval=False,
            monitor=None
        )

        super().__init__(ID=ID,
                 server_id=server_id,
                 state=state,
                 config=config,
                 data=data,
                 model=model,
                 device=device,
                 strategy=strategy,
                 is_unseen_client=is_unseen_client,
                 trainer=trainer,
                 *args,
                 **kwargs)
        #self._test_align_global_local_parameters(self.model)
        trainer.monitor = self._monitor


    def _set_init_omega(self, model, device):

        omega_set = {}
        for name, param in deepcopy(model).named_parameters():
            omega_set[name] = torch.zeros_like(param.data).to(device)
        return omega_set

    def _align_global_local_parameters(self, model):
        for name, param in deepcopy(model).named_parameters():
            if name.startswith('local'):
                stripped_name = name[len('local'):]
                global_name = 'global' + stripped_name
                model.state_dict()[name].data.copy_(copy.deepcopy(model.state_dict()[global_name].data))

    def _test_align_global_local_parameters(self, model):
        for name, param in deepcopy(model).named_parameters():
            if name.startswith('local'):
                stripped_name = name[len('local'):]
                global_name = 'global' + stripped_name
                local_params = model.state_dict()[name].data
                global_params = model.state_dict()[global_name].data
                print(f"local data: {param}")
                print(f"global data: {model.state_dict()[global_name].data}")
                break

    def callback_funcs_for_model_para(self, message: Message):
        """
        The handling function for receiving model parameters,
        which triggers the local training process.
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message, which includes sender, receiver,
            state, and content.
                More detail can be found in federatedscope.core.message
        """


        round, sender, content = message.state, message.sender, \
                                 message.content
        # When clients share the local model, we must set strict=True to
        # ensure all the model params (which might be updated by other
        # clients in the previous local training process) are overwritten
        # and synchronized with the received model
        with torch.no_grad():
            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model)
        self.state = round
        skip_train_isolated_or_global_mode = \
            self.early_stopper.early_stopped and \
            self._cfg.federate.method in ["local", "global"]
        if self.is_unseen_client or skip_train_isolated_or_global_mode:
            # for these cases (1) unseen client (2) isolated_global_mode,
            # we do not local train and upload local model
            sample_size, model_para_all, results = \
                0, self.trainer.get_model_para(), {}
            if skip_train_isolated_or_global_mode:
                logger.info(
                    f"[Local/Global mode] Client #{self.ID} has been "
                    f"early stopped, we will skip the local training")
                self._monitor.local_converged()
        else:
            if self.early_stopper.early_stopped and \
                    self._monitor.local_convergence_round == 0:
                logger.info(
                    f"[Normal FL Mode] Client #{self.ID} has been locally "
                    f"early stopped. "
                    f"The next FL update may result in negative effect")
                self._monitor.local_converged()
            sample_size, model_para_all, omega_set, results = self.trainer.train(self.state)
            if self._cfg.federate.share_local_model and not \
                    self._cfg.federate.online_aggr:
                model_para_all = copy.deepcopy(model_para_all)
            train_log_res = self._monitor.format_eval_res(
                results,
                rnd=self.state,
                role='Client #{}'.format(self.ID),
                return_raw=True)
            logger.info(train_log_res)
            if self._cfg.wandb.use and self._cfg.wandb.client_train_info:
                self._monitor.save_formatted_results(train_log_res,
                                                     save_file_name="")

        # Return the feedbacks to the server after local update
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=(sample_size, model_para_all, omega_set)))

