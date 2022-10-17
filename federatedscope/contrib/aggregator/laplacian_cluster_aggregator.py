from federatedscope.core.aggregator import Aggregator
import copy

class LaplacianClusterAggregator(Aggregator):
    def __init__(self, model=None, omega=None, device='cpu', config=None):
        super(Aggregator, self).__init__()
        self.model = copy.deepcopy(model)
        self.omega=copy.deepcopy(omega)
        self.device = device
        self.cfg = config
        self.clusters = []



    def aggregate(self, agg_info):
        # alpha, model_params, omega
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

        for key in new_param.keys():
            self.model.state_dict()[key].data.copy_(new_param[key])
            self.omega[key] = copy.deepcopy(new_omega[key])
        return new_param, new_omega

    def subtract_(target, minuend, subtrahend):
        for name in target:
            target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()

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