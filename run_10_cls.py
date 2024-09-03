# %%
import os
import torch
# import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
# from PIL import Image
import torchvision
# from torch.utils.data import Subset
from tqdm import tqdm
# from torch.distributions.laplace import Laplace
import math
from ntk_utils import gen_h_dis, gen_alpha, gen_z_embed, process_query
from truncated_laplace_utils import add_truncated_laplace_noise, add_laplace_noise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# the following code is adapted from
# https://github.com/josharnoldjosh/Resnet-Extract-Image-Feature-Pytorch-Python 

# Load the pretrained resnet18 model
model = models.resnet18(pretrained=True)
model = model.to(device)

# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

# Image transforms
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(img):
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512).to(device)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        # my_embedding.copy_(o.data)
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    t_img = t_img.to(device)
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()

    return my_embedding

# download cifar10 dataset
ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)



# %%

# mapping from class name to index
class_to_idx_dict = ds.class_to_idx
# mapping from index to class name
idx_to_cls_dict = {}

# collect all class name
class_name_list = []
# collect all class index
class_idx_list = []
for key, value in list(class_to_idx_dict.items()):
    class_name_list.append(key)
    class_idx_list.append(value)
    idx_to_cls_dict[value] = key

# collect data according to its class
ds_train_idx_by_cls_list = [] 
for cls_idx in class_idx_list:
    ds_train_idx_by_cls_list.append([])

# start collecting
for i in tqdm(range(len(ds))):
    cur_cls_idx = ds[i][1]
    ds_train_idx_by_cls_list[cur_cls_idx].append(i)


# %%
######### normalization x data start ###########
def calculate_norm(input_data):
    # input_data: n * d
    square_data = input_data * input_data
    norm_data = square_data.sum(dim=1)
    norm_data = torch.sqrt(norm_data)
    return norm_data

def data_normalization(input_data):
    # input data: n * d
    # print("data normalized")
    x_norm = calculate_norm(input_data)
    x_norm = x_norm[..., None]
    ball_data = input_data / x_norm

    return ball_data
######### normalization x data end ###########


train_num = 1000
test_num = 100
label_num = 10

total_train_idx_list = []
total_test_idx_list = []
for train_idx_list in ds_train_idx_by_cls_list:
    total_train_idx_list += train_idx_list[:train_num]
    total_test_idx_list += train_idx_list[-test_num:]

print("total train idx len", len(total_train_idx_list))
print("total test idx len", len(total_test_idx_list))

############ get img and label tensor according to idx start ###############
def get_img_label_tensor(idx_list):
    img_ts_list = []
    label_list = []
    for idx in tqdm(idx_list):
        image, label = ds[idx]
        img_ts = get_vector(image)
        img_ts_list.append(img_ts)
        label_list.append(label)

    # concat image tensor
    img_ts = torch.stack(img_ts_list, dim=0)

    label_ts = torch.zeros((len(idx_list), label_num), dtype=torch.float32)
    # set the negative label to -1
    label_ts -= 1.0
    for i, label in enumerate(label_list):
        # set corresponding label to 1
        label_ts[i][label] = 1.0

    cls_index_label_ts = torch.tensor(label_list, dtype=torch.int64)

    return img_ts, label_ts, cls_index_label_ts
############ get img and label tensor according to idx end ###############

# img_ts: n * 512
# label_ts: n * 10
# cls_index_label_ts: n * 1
train_img_ts, train_label_ts, train_cls_index_label_ts = get_img_label_tensor(total_train_idx_list)
test_img_ts, test_label_ts, test_cls_index_label_ts = get_img_label_tensor(total_test_idx_list)

train_img_ts = data_normalization(train_img_ts)
test_img_ts = data_normalization(test_img_ts)

m = 256
reg_lambda = 10.0


cpu_device = torch.device("cpu")

x_data = train_img_ts.to(cpu_device)
y_data = train_label_ts.to(cpu_device)
n, d = x_data.shape

# generate w_r
w_r = torch.randn((m, d), dtype=torch.float32).to(cpu_device)
# h_dis: n * n
h_dis = gen_h_dis(w_r, x_data)

print("hdis shape", h_dis.shape)

# calculate NTK Regression alpha
alpha = gen_alpha(h_dis, reg_lambda, y_data)

print("alpha shape", alpha.shape)
    

################# test ntk regression accuracy start #####################
def process_10_cls_query(z, w_r, x_data, alpha):
    # z denote the query, nz denote the query num
    # z: nz * d
    # w_r: m * d
    # x_data: n * d
    # alpha: n * 10
    # return: pred: nz * 1

    # nz * n
    query_embed = gen_z_embed(z, x_data, w_r)

    # nz * 10
    query_pred = query_embed @ alpha

    query_result = torch.argmax(query_pred, dim=1)

    # nz * 1
    return query_result


def test_accuracy_for_10_cls(test_dataset, gt_label, w_r, x_data, alpha):
    # test_dataset: n * 512
    # gt_label: n * 1
    
    # pred: nz * 1
    pred = process_10_cls_query(test_dataset, w_r, x_data, alpha)
    nz = pred.shape[0]
    succ_cnt = torch.sum(pred == gt_label)
    test_acc = succ_cnt / nz
    return test_acc
################# test ntk regression accuracy end #####################


unprivate_train_acc = test_accuracy_for_10_cls(train_img_ts, train_cls_index_label_ts, w_r, train_img_ts, alpha)
unprivate_test_acc = test_accuracy_for_10_cls(test_img_ts, test_cls_index_label_ts, w_r, train_img_ts, alpha)

print(unprivate_train_acc, unprivate_test_acc)

with open(os.path.join("logs", "unprivate_log.txt"), "w") as fw:
    fw.write(f"unprivate_train_acc {unprivate_train_acc}\n")
    fw.write(f"unprivate_test_acc {unprivate_test_acc}\n")


def cal_k(eps, delta, beta):
    eta = 7e-3
    n = 1e3
    k_bound = (eps * eps * eta * eta) / (8 * math.log(1 / delta) * n * n * beta * beta)
    k = int(math.floor(k_bound))
    return k


private_repeat_time = 5


def gaussain_sampling_on_k(h_dis, y_data, reg_lambda, wt_train_img_ts=None, k=None):
    
    assert wt_train_img_ts is not None

    test_acc_list = []
    train_acc_list = []

    # repeat_time = 10
    for _ in tqdm(range(private_repeat_time)):
        if k is None:
            wt_h_dis = h_dis
        else:
            # setup gaussian sampler
            gaussian_sampler = torch.distributions.MultivariateNormal(
                loc=torch.zeros(n).to(device), covariance_matrix=h_dis
            )

            # gausian sampling
            wt_h_dis = torch.empty(k, n, n).to(device)

            for i in range(k):
                sample_vec = gaussian_sampler.sample()
                wt_h_dis[i] = sample_vec[..., None] @ sample_vec[None, ...]

            # take mean over dim k
            # n * n
            wt_h_dis = wt_h_dis.mean(dim=0)

        alpha = gen_alpha(wt_h_dis, reg_lambda, y_data)
        alpha = alpha / n

        cur_train_acc = test_accuracy_for_10_cls(train_img_ts, train_cls_index_label_ts, w_r, wt_train_img_ts, alpha)
        cur_test_acc = test_accuracy_for_10_cls(test_img_ts, test_cls_index_label_ts, w_r, wt_train_img_ts, alpha)

        train_acc_list.append(cur_train_acc)
        test_acc_list.append(cur_test_acc)

        cur_train_mean = sum(train_acc_list) / len(train_acc_list)
        cur_test_mean = sum(test_acc_list) / len(test_acc_list)
        print(cur_train_mean, cur_test_mean)

    final_train_acc = sum(train_acc_list) / len(train_acc_list)
    final_test_acc = sum(test_acc_list) / len(test_acc_list)

    return final_train_acc, final_test_acc


# fix beta and delta
beta = 1e-6
delta = 1e-3

# we run different eps exponent, from (0.5, 1.5)
# eps_exponent_list = [0.5 + i * 0.1 for i in range(11)]
eps_exponent_list = [1.4, 1.5]

for eps_exponent in eps_exponent_list:
    eps = 10 ** eps_exponent

    # calculate number of Gaussian Samples according to eps
    k = cal_k(eps, delta, beta)

    print("-" * 50)

    print(f"eps exponent {eps_exponent}, k {k}")

    # add truncated laplace noise on X
    # wt_train_img_ts = add_truncated_laplace_noise(beta, eps, delta, train_img_ts)

    wt_train_img_ts = add_laplace_noise(beta, eps, x_data)

    print("truncated laplace noise added")

    # get private test acc and train acc
    final_test_acc, final_train_acc = gaussain_sampling_on_k(h_dis, y_data, reg_lambda, wt_train_img_ts, k)
    
    print("test acc", final_test_acc)
    print("train acc", final_train_acc)

    # with open("log.txt", "a+") as f

    cur_log_path = os.path.join("logs", f"eps_{eps_exponent}.txt")
    with open(cur_log_path, "w") as fw:
        # fw.write(f"eps_exponent {str(eps_exponent)}\n")
        fw.write(f"test_acc {str(final_test_acc)}\n")
        fw.write(f"train_acc {str(final_train_acc)}\n")
    

    print("-" * 50)

        


