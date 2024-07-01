import torch

from gen_data import gen_random_multi_gaussian, gen_sanitiy_check_data

def gen_h_dis(w_r, x_data):
    # h^{dis}_{ij} = 1/m sum_r^m <<w_r, x_i> x_i, <w_r, x_j> x_j>
    # h^{dis}_{ij} = <x_i, x_j> 1/m sum_r^m <w_r, x_i> <w_r, x_j>
    # where 
    # <x_i, x_j>: n * n
    # <w_r, x_i> <w_r, x_j>: m * n * n
    # 1/m sum_r^m <w_r, x_i> <w_r, x_j>: n * n

    n = x_data.shape[0]
    m = w_r.shape[0]

    # # generate neurons: m * d shape
    # w_r = torch.randn((m, d), dtype=torch.float32)

    # n * n
    inner_xi_xj = x_data @ x_data.t()

    # m * n
    inner_wr_xi = w_r @ x_data.t()

    # m * n * n
    inner_wr_xi_inner_wr_xj = torch.empty((m, n, n), dtype=torch.float32)
    
    for iter_m in range(m):
        # n * n = n * 1 @ 1 * n
        inner_wr_xi_inner_wr_xj[iter_m] = inner_wr_xi[iter_m][..., None] @ inner_wr_xi[iter_m][None, ...]

    # n * n
    avg_inner_wr_xi_inner_wr_xj = inner_wr_xi_inner_wr_xj.mean(dim=0)

    # n * n
    h_dis = avg_inner_wr_xi_inner_wr_xj * inner_xi_xj

    return h_dis

def gen_alpha(h_dis, reg_lambda, y_data):
    # h_dis: n * n
    # reg_lambda: float
    # y_data: n * 1
    # return: alpha: n * 1

    # https://pytorch.org/docs/stable/generated/torch.linalg.inv.html
    # linalg.solve(A, B) == linalg.inv(A) @ B  # When B is a matrix

    n = h_dis.shape[0]

    # n * n
    k_plus_lambda = h_dis + reg_lambda * torch.eye(n)

    # n * 1
    alpha = torch.linalg.solve(k_plus_lambda, y_data)

    return alpha


def gen_z_embed(z, x_data, w_r):
    # z: 1 * d
    # x_data: n * d
    # w_r: m * d

    # 1/m sum_r^m <<w_r, z> z, <w_r, x_i> x_i>
    # = <z, x_i> 1/m sum_r^m <w_r, z> <w_r, x_i>
    # where 
    # <z, x_i>: 1 * n
    # <w_r, x_i>: m * n
    # <w_r, z> <w_r, x_i>: m * 1 * n
    # 1/m sum_r^m <w_r, z> <w_r, x_i>: 1 * n

    # return: z_embed: 1 * n

    m = w_r.shape[0]
    n = x_data.shape[0]
    nz = z.shape[0]

    # m * n
    inner_wr_xi = w_r @ x_data.t()

    # m * 1
    inner_wr_z = w_r @ z.t()

    # 1 * n
    inner_z_xi = z @ x_data.t()

    inner_wr_z_wr_xi = torch.empty((m, nz, n), dtype=torch.float32)
    # breakpoint()
    for iter_m in range(m):
        inner_wr_z_wr_xi[iter_m] = inner_wr_z[iter_m][..., None] @ inner_wr_xi[iter_m][None, ...]
        # try:
        #     inner_wr_z_wr_xi[iter_m] = inner_wr_z[iter_m][..., None] @ inner_wr_xi[iter_m][None, ...]
        # except Exception as e:
        #     print(e)
        #     breakpoint()
        #     print()

    # 1 * n
    avg_inner_wr_z_wr_xi = inner_wr_z_wr_xi.mean(dim=0)

    z_embed = inner_z_xi * avg_inner_wr_z_wr_xi

    return z_embed


if __name__ == "__main__":

    m = 20
    reg_lambda = 1.0

    positive_data, negative_data = gen_sanitiy_check_data()
    print(positive_data.shape)
    print(positive_data[:10])
    print(negative_data.shape)
    print(negative_data[:10])

    # gen label here
    n, d = positive_data.shape

    # generate w_r
    w_r = torch.randn((m, d), dtype=torch.float32)
    
    positive_label = torch.full((n, ), 1, dtype=torch.float32)
    negative_label = torch.full((n, ), -1, dtype=torch.float32)

    # concate them
    # x_data: 100 * d
    x_data = torch.cat((positive_data, negative_data), dim=0)

    # 100 * 1
    y_data = torch.cat((positive_label, negative_label), dim=0)

    # 100 * 100
    h_dis = gen_h_dis(w_r, x_data)

    alpha = gen_alpha(h_dis, reg_lambda, y_data)

    # nz * n
    sanity_query_embed = gen_z_embed(positive_data, x_data, w_r)

    # nz * 1
    sanity_pred = sanity_query_embed @ alpha
    
    # set to 1 and -1
    sanity_pred[sanity_pred >= 0] = 1
    sanity_pred[sanity_pred < 0] = -1

    # correct = sanity_pred.sum()
    count_ones = torch.sum(sanity_pred == 1)
    nz = sanity_pred.shape[0]

    accuracy = count_ones / nz

    print(sanity_pred)
    print(accuracy)

    breakpoint()
    print()







