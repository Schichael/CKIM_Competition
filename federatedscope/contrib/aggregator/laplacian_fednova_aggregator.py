from federatedscope.core.aggregator import Aggregator
import copy
import torch

class LaplacianNovaAggregator(Aggregator):
    def __init__(self, model=None, omega=None, device='cpu', config=None):
        super(LaplacianNovaAggregator, self).__init__()

        self.omega=copy.deepcopy(omega)
        self.device = device
        self.cfg = config
        self.model = self._set_init_model(model, device)

    def _set_init_model(self, model, device):
        model_set = {}
        print("server params: ")
        for name, param in copy.deepcopy(model).named_parameters():
            print(name)
            model_set[name] = param.to(self.device)
        return model_set

    def aggregate(self, agg_info):
        # alpha, model_params, omega
        client_feedback = agg_info["client_feedback"]
        server_omega = agg_info["server_omega"]
        eps = agg_info['eps']
        p = agg_info['p']
        alpha_sum = 0
        tau_effs = []
        for c in client_feedback:
            alpha, model_params, omega, num_rounds = c
            alpha_sum += c[0]

        for c in client_feedback:
            alpha, model_params, omega, num_rounds = c

            tau_eff_cuda = torch.tensor(
                num_rounds * alpha/alpha_sum).cuda()
            tau_effs.append(tau_eff_cuda)

        tau_eff = tau_effs[0]
        for i, el in enumerate(tau_effs):
            if i == 0:
                continue
            tau_eff = tau_eff + el

        tau_eff = tau_eff.item()

        scales = []


        for c in client_feedback:
            alpha, model_params, omega, num_rounds = c
            scale = tau_eff / num_rounds
            scales.append(scale)

        print(scales)

        new_param = {}
        new_omega = {}
        for name, param in self.model.items():
            new_param[name] = param.detach().clone().data.zero_().to(self.device)
            new_omega[name] = server_omega[name].data.zero_().to(self.device)


            #    print(f"old param: {param[0][:3]}")

            for i, c in enumerate(client_feedback):
                alpha, model_params, omega, num_rounds = c

                if name in omega:
                    model_grad = model_params[name].to(self.device) - self.model[name].to(self.device)
                    norm_model_grad = model_grad * scales[i]
                    model_new = self.model[name] + norm_model_grad
                    model_new = model_new.to(self.device)

                    # model_params[name] = model_params[name].to(self.device)

                    # Not sure about the division
                    new_param[name] += omega[name] * model_new * (alpha/alpha_sum)

                    #new_param[name] += (alpha / alpha_sum) * omega[name] * model_params[name]
                    #if i == 35:
                    #    print(f"client param: {model_params[name][0][:3]}")
                    #    print(f"client omega: {omega[name][0][:3]}")
                    new_omega[name] += omega[name] * (alpha/alpha_sum)
            #print(f"new_omega name: {name}: {new_omega[name]}")
            new_param[name] /= (new_omega[name] + eps)
            #print(f"new_param name: {name}: {new_param[name]}")
            #new_param[name] = new_param[name] / new_param[name]


        # Pruning
        if p>0:
            self.pruning_server(p, new_param, new_omega)

        for key in new_param.keys():
            self.model[key] = copy.deepcopy(new_param[key])
            self.omega[key] = copy.deepcopy(new_omega[key])

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