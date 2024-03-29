import os
import shutil
import time

import torch

from federatedscope.core.aggregator import Aggregator
import copy

class LaplacianAggregator(Aggregator):
    def __init__(self, model=None, omega=None, device='cpu', config=None):
        super(LaplacianAggregator, self).__init__()
        self.model = copy.deepcopy(model)
        self.omega=copy.deepcopy(omega)
        self.device = device
        self.cfg = config

    def load_model(self, path):
        assert self.model is not None

        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt['model'])
            return ckpt['cur_round']
        else:
            raise ValueError("The file {} does NOT exist".format(path))
    def save_model(self, path, cur_round=-1):
        assert self.model is not None

        ckpt = {'cur_round': cur_round, 'model': self.model.state_dict()}
        torch.save(ckpt, self.cfg.outdir + '/aggregated_model.pt')


        if self.cfg.params.save_client_always and cur_round >= 3:
            for c in range(1, self.cfg.federate.sample_client_num+1):
                #path = self.cfg.outdir + f'/model{str(c)}.pth'
                path = self.cfg.outdir + f'/model{str(c)}_{cur_round % 2}.pth'
                new_path = self.cfg.outdir + f'/best_aggr_model{str(c)}.pth'
                shutil.copyfile(path, new_path)

    def aggregate(self, agg_info):
        # alpha, model_params, omega
        st = time.time()
        client_feedback = agg_info["client_feedback"]
        server_omega = agg_info["server_omega"]
        eps = agg_info['eps']
        p = agg_info['p']
        alpha_sum = 0
        for c in client_feedback:
            alpha_sum += c[0]
        new_param = {}
        new_omega = {}

        for name, param in self.model.named_parameters():
            new_param[name] = param.data.zero_().to(self.device)
            new_omega[name] = server_omega[name].data.zero_().to(self.device)


            #    print(f"old param: {param[0][:3]}")
            for c in client_feedback:
                alpha, model_params, omega = c
                if name in omega:
                    model_params[name] = model_params[name].to(self.device)
                    new_param[name] += (alpha / alpha_sum) * omega[name] * model_params[name]

                    #new_param[name] += (alpha / alpha_sum) * omega[name] * model_params[name]
                    #if i == 35:
                    #    print(f"client param: {model_params[name][0][:3]}")
                    #    print(f"client omega: {omega[name][0][:3]}")
                    new_omega[name] += (alpha / alpha_sum) * omega[name]
            #print(f"new_omega name: {name}: {new_omega[name]}")
            new_param[name] /= (new_omega[name] + eps)
            #print(f"new_param name: {name}: {new_param[name]}")
            #new_param[name] = new_param[name] / new_param[name]


        # Pruning
        if p>0:
            self.pruning_server(p, new_param, new_omega)
        et1 = time.time()
        for key in new_param.keys():
            self.model.state_dict()[key].data.copy_(new_param[key])
            self.omega[key] = copy.deepcopy(new_omega[key])
        et2 = time.time()

        return new_param, new_omega

    def pruning_server(self, p, new_param, new_omega):
        for name in new_param.keys():
            unpruing_omega = new_omega[name]
            threshold = self.get_threshold(p, unpruing_omega)
            new_omega[unpruing_omega < threshold] = unpruing_omega[unpruing_omega < threshold].mean().item()
            # unpruing_param = deepcopy(server_model.state_dict()[name].data.detach())
            # new_param[name][unpruing_omega < threshold] = unpruing_param[unpruing_omega < threshold]

    def get_threshold(self, p, matrix):
        rank_matrix, _ = matrix.view(-1).sort(descending=True)
        threshold = rank_matrix[int(p * len(rank_matrix))]
        return threshold