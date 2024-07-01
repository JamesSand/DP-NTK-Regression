import torch

from gen_data import gen_random_multi_gaussian

n = 100

# generate m neuron projectors
m = 10
d = 5

def gen_h_dis(m, x_data):
    # h^{dis}_{ij} = 1/m sum^m <<w_r, x_i> x_i, <w_r, x_j> x_j>
    # h^{dis}_{ij} = <x_i, x_j> 1/m sum^m <w_r, x_i> <w_r, x_j>
    # where <x_i, x_j>: n * n
    # <w_r, x_i> <w_r, x_j>: m * n * n
    # 1/m sum^m <w_r, x_i> <w_r, x_j>: n * n

    # generate neurons: m * d shape
    w_r = torch.randn(m, d)

    # n * n
    inner_xi_xj = x_data @ x_data.t()

    # m * n
    inner_wr_xi = w_r @ x_data.t()

    # m * n * n
    inner_wr_xi_inner_wr_xj = torch.empty(m, n, n)
    
    for iter_m in range(m):
        inner_wr_xi_inner_wr_xj[iter_m] = inner_wr_xi[iter_m][..., None] @ inner_wr_xi[iter_m][None, ...]

    # n * n
    avg_inner_wr_xi_inner_wr_xj = inner_wr_xi_inner_wr_xj.mean(dim=0)

    # n * n
    h_dis = avg_inner_wr_xi_inner_wr_xj * inner_xi_xj

    return h_dis

    


