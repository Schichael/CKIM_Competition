import torch

from federatedscope.gfl.trainer import GraphMiniBatchTrainer


class FedAvg_VAE_trainer(GraphMiniBatchTrainer):
    def __init__(self,
                 model,
                 data,
                 device,
                 config,
                 only_for_eval=False,
                 monitor=None):
        super().__init__(model,
                         data,
                         device,
                         config,
                         only_for_eval,
                         monitor)

    def _hook_on_batch_forward(self, ctx):
        batch = ctx.data_batch.to(ctx.device)

        pred, vae_loss = ctx.model(batch)
        # print(f"vae_loss: {vae_loss}")
        # TODO: deal with the type of data within the dataloader or dataset
        if 'regression' in ctx.cfg.model.task.lower():
            label = batch.y
        else:
            label = batch.y.squeeze(-1).long()
        if len(label.size()) == 0:
            label = label.unsqueeze(0)
        ctx.vae_loss = vae_loss
        ctx.loss_batch_ce = ctx.criterion(pred, label)
        ctx.loss_batch = ctx.loss_batch_ce

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
        """
        ctx.optimizer.zero_grad()
        ctx.loss_task.backward()
        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()
        """

        ctx.optimizer.zero_grad()
        loss = ctx.loss_batch_ce + ctx.vae_loss
        loss.backward(retain_graph=False)

        if ctx.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(ctx.model.parameters(),
                                           ctx.grad_clip)
        ctx.optimizer.step()

def call_fedavg_VAE_trainer(trainer_type):
    if trainer_type == 'FedAvg_VAE_trainer':
        trainer_builder = FedAvg_VAE_trainer
        return trainer_builder