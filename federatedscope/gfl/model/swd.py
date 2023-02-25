# sliced wasserstein distance
# https://github.com/koshian2/swd-pytorch/blob/master/swd.py
import torch


def swd(x1, x2, n_repeat_projection=128, proj_per_repeat=4, device='cuda:0'):
    # n_repeat_projectton * proj_per_repeat = 512
    # Please change these values according to memory usage.
    # original = n_repeat_projection=4, proj_per_repeat=128

    distances = []
    for j in range(n_repeat_projection):
        # random
        rand = torch.randn(x1.size(1), proj_per_repeat).to(device)  # (slice_size**2*ch)
        rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize
        # projection
        proj1 = torch.matmul(x1, rand)
        proj2 = torch.matmul(x2, rand)
        proj1, _ = torch.sort(proj1, dim=0)
        proj2, _ = torch.sort(proj2, dim=0)
        d = torch.abs(proj1 - proj2)
        distances.append(torch.mean(d))

    # swd
    result = torch.mean(torch.stack(distances))

    return result
