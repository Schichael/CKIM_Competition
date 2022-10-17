import copy
import logging
from copy import deepcopy

import torch
from federatedscope.core import aggregator as agg
from federatedscope.contrib.aggregator.laplacian_aggregator import LaplacianAggregator
from federatedscope.contrib.trainer.laplacian_trainer import LaplacianTrainer
from federatedscope.contrib.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_dict
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)

class LaplacianServer(Server):
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 unseen_clients_id=None,
                 **kwargs):

        self.omega_set = self._set_init_omega(model, device)
        self.eps = config.params.eps
        self.p = config.params.p
        #aggregator = agg.ClientsAvgAggregator(model=model,
        #                                 device=device,
        #                                 config=config,
        #                                 )
        aggregator = LaplacianAggregator(model=model,
                                         omega=self.omega_set,
                                         device=device,
                                         config=config,
                                         )
        """
        trainer = LaplacianTrainer(
            model=model,
            omega=self.omega_set,
            data=data,
            device=device,
            config=config,
            only_for_eval=False,
            monitor=None
        )
        """
        trainer = None

        super().__init__(
            ID=ID,
            state=state,
            config=config,
            data=data,
            model=model,
            client_num=client_num,
            total_round_num=total_round_num,
            device=device,
            strategy=strategy,
            unseen_clients_id=unseen_clients_id,
            trainer=trainer,
            aggregator=aggregator,
            **kwargs
        )



    def _set_init_omega(self, model, device):
        omega_set = {}
        print("server params: ")
        for name, param in deepcopy(model).named_parameters():
            print(name)
            omega_set[name] = torch.zeros_like(param.data).to(device)
        return omega_set

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer. When enough messages are receiving,
        some events (such as perform aggregation, evaluation, and move to
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for
            evaluation; and check the message buffer for training otherwise.
        """
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone":
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result):
            if not check_eval_result:  # in the training process
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                for model_idx in range(self.model_num):
                    model = self.models[model_idx]
                    aggregator = self.aggregators[model_idx]
                    msg_list = list()
                    for client_id in train_msg_buffer:
                        if self.model_num == 1:
                            msg_list.append(train_msg_buffer[client_id])
                        else:
                            train_data_size, model_para_multiple = \
                                train_msg_buffer[client_id]
                            msg_list.append((train_data_size,
                                             model_para_multiple[model_idx]))

                    # Trigger the monitor here (for training)
                    if 'dissim' in self._cfg.eval.monitoring:
                        B_val = self._monitor.calc_blocal_dissim(
                            model.load_state_dict(strict=False), msg_list)
                        formatted_eval_res = self._monitor.format_eval_res(
                            B_val, rnd=self.state, role='Server #')
                        logger.info(formatted_eval_res)

                    # Aggregate
                    agg_info = {
                        'client_feedback': msg_list,
                        'recover_fun': self.recover_fun,
                        'eps': self.eps,
                        'p': self.p,
                        'server_omega': self.omega_set
                    }
                    with torch.no_grad():
                        new_param, new_omega = aggregator.aggregate(agg_info)
                        # for key in result:
                        #    model.state_dict()[key].data.copy_(result[key])
                        # model.load_state_dict(new_param, strict=False)

                        #print("load state dict")
                        #for name, param in model.state_dict:
                        #    print(name)
                        #for key in new_param:
                        #    print(key)
                        #for name, param in self.model.named_parameters():
                        #    self.omega_set[name] = copy.deepcopy(new_omega[name])

                        for name, param in self.model.named_parameters():
                            self.model.state_dict()[name].data.copy_(new_param[name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638
                            self.omega_set[name] = copy.deepcopy(new_omega[name])

                self.state += 1
                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    logger.info(
                        f'Server #{self.ID}: Starting evaluation at the end '
                        f'of round {self.state - 1}.')
                    self.eval()

                if self.state < self.total_round_num:
                    # Move to next round of training
                    logger.info(
                        f'----------- Starting a new training round (Round '
                        f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    # Final Evaluate
                    logger.info('Server #{:d}: Training is finished! Starting '
                                'evaluation.'.format(self.ID))
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict(self.history_results,
                                                  formatted_eval_res)
                if self.mode == 'standalone' and \
                        self._monitor.wandb_online_track and \
                        self._monitor.use_wandb:
                    self._monitor.merge_system_metrics_simulation_mode(
                        file_io=False, from_global_monitors=True)
                self.check_and_save()

        else:
            move_on_flag = False

        return move_on_flag


    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast
                behavior. And sample_client_num = -1 denotes to broadcast to
                all the clients.
            filter_unseen_clients: whether filter out the unseen clients that
                do not contribute to FL process by training on their local
                data and uploading their local model update. The splitting is
                useful to check participation generalization gap in [ICLR'22,
                What Do We Mean by Generalization in Federated Learning?]
                You may want to set it to be False when in evaluation stage
        """
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        skip_broadcast = self._cfg.federate.method in ["local", "global"]
        model_para = {} if skip_broadcast else self.model.state_dict()
        omega_set = {} if skip_broadcast else self.omega_set

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(self.state, self.total_round_num),
                    content=[model_para, omega_set]))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def terminate(self, msg_type='finish'):
        """
        To terminate the FL course
        """
        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.model.state_dict()
            omega = self.omega_set
        self._monitor.finish_fl()

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    content=[model_para, omega]))
































