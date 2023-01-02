import logging
from copy import deepcopy
import copy

import torch

from federatedscope.contrib.trainer.FedAvg_VAE_trainer import FedAvg_VAE_trainer
from federatedscope.contrib.trainer.laplacian_trainer import LaplacianTrainer
from federatedscope.contrib.workers.client import Client
from federatedscope.core.message import Message

logger = logging.getLogger(__name__)


class Fedavg_VAE_client(Client):
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
        trainer = FedAvg_VAE_trainer(
            model=model,
            data=data,
            device=device,
            config=config,
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


        trainer.monitor = self._monitor



