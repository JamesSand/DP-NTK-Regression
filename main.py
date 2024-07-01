import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import torchvision
from torch.utils.data import Subset
from tqdm import tqdm
from torch.distributions.laplace import Laplace


from ntk_utils import gen_h_dis, gen_alpha, gen_z_embed, process_query

# device =
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pretrained model
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
    # 1. Load the image with Pillow library
    # img = Image.open(image_name)
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
    # origin_device = t_img.device
    t_img = t_img.to(device)
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()

    return my_embedding

    # my_embedding = my_embedding.to(origin_device)
    # # 8. Return the feature vector
    # return my_embedding.numpy()


ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=False)

def gen_2classes_indices(cls1_name, cls2_name):
    cls1_indices, cls2_indices, other_indices = [], [], []
    cls1_idx, cls2_idx = ds.class_to_idx[cls1_name], ds.class_to_idx[cls2_name]

    for i in range(len(ds)):
        current_class = ds[i][1]
        if current_class == cls1_idx:
            cls1_indices.append(i)
        elif current_class == cls2_idx:
            cls2_indices.append(i)
        else:
            other_indices.append(i)

    return cls1_indices, cls2_indices

def gen_feature_tensor(idx_list):
    img_ts_list = []
    for idx in tqdm(idx_list):
        image, label = ds[idx]
        img_ts = get_vector(image)
        # img_ts = torch.from_numpy(img_np)
        img_ts_list.append(img_ts)

    # concat
    ret_ts = torch.stack(img_ts_list, dim=0)
    return ret_ts


def test_accuracy(test_dataset, gt_label, w_r, x_data, alpha):
    pred = process_query(test_dataset, w_r, x_data, alpha)
    succ_cnt = torch.sum(pred == gt_label)
    nz = pred.shape[0]
    accuracy = succ_cnt / nz
    # print("accuracy", accuracy)
    return accuracy


def add_laplace_on_alpha(cls1_test_ts, cls2_test_ts, alpha, reg_lambda, w_r, x_data, eps=None):
    if eps is None:
        print("Not adding any noise on alpha")
        return alpha
    n = alpha.shape[0]
    Delta = 3 / (n * reg_lambda)
    dp_lambda = Delta / eps

    laplace_sampler = Laplace(torch.tensor([0.0]), torch.tensor([dp_lambda]))

    # for loop here
    test_acc_list = []
    train_acc_list = []

    for i in tqdm(range(10)):
      laplace_noise = laplace_sampler.sample((n, )).to(alpha.device)
      laplace_noise = laplace_noise[..., 0]
      # print(alpha.shape)
      # print(laplace_noise.shape)

      wt_alpha = alpha + laplace_noise

      cls1_accuracy = test_accuracy(cls1_test_ts, 1, w_r, x_data, wt_alpha)
      cls2_accuracy = test_accuracy(cls2_test_ts, -1, w_r, x_data, wt_alpha)

      cls1_train_acc = test_accuracy(cls1_train_ts, 1, w_r, x_data, wt_alpha)
      cls2_train_acc = test_accuracy(cls2_train_ts, -1, w_r, x_data, wt_alpha)

      test_acc_list.append((cls1_accuracy + cls2_accuracy) / 2)
      train_acc_list.append((cls1_train_acc + cls2_train_acc) / 2)

    final_test_acc = sum(test_acc_list) / len(test_acc_list)
    final_train_acc = sum(train_acc_list) / len(train_acc_list)

    return final_test_acc, final_train_acc


    # return wt_alpha


if __name__ == "__main__":

    # there are 5k images for 1 class
    train_num = 1000
    test_num = 100
    label_scale = 100.0

    cls1_name = "airplane"
    cls2_name = "cat"

    cls1_indices, cls2_indices = gen_2classes_indices(cls1_name, cls2_name)

    cls1_train_ts = gen_feature_tensor(cls1_indices[:train_num]).to(device)
    cls2_train_ts = gen_feature_tensor(cls2_indices[:train_num]).to(device)

    cls1_label = torch.full((train_num, ), label_scale, dtype=torch.float32)
    cls2_label = torch.full((train_num, ), -1 * label_scale, dtype=torch.float32)

    cls1_test_ts = gen_feature_tensor(cls1_indices[-test_num:]).to(device)
    cls2_test_ts = gen_feature_tensor(cls2_indices[-test_num:]).to(device)


    ############# test on NTK Regression start #################

    m = 256
    reg_lambda = 10.0

    x_data = torch.cat((cls1_train_ts, cls2_train_ts), dim=0).to(device)
    y_data = torch.cat((cls1_label, cls2_label), dim=0).to(device)

    n, d = x_data.shape

    # generate w_r
    w_r = torch.randn((m, d), dtype=torch.float32).to(device)

    h_dis = gen_h_dis(w_r, x_data)

    alpha = gen_alpha(h_dis, reg_lambda, y_data)

    # may scale down alpha here
    alpha = alpha / (n * n)

    eps = 1e10

    # add lapalce nosie on alpha
    wt_alpha = add_laplace_on_alpha(alpha, reg_lambda, eps)

    cls1_accuracy = test_accuracy(cls1_test_ts, 1, w_r, x_data, wt_alpha)
    cls2_accuracy = test_accuracy(cls2_test_ts, -1, w_r, x_data, wt_alpha)

    cls1_train_acc = test_accuracy(cls1_train_ts, 1, w_r, x_data, wt_alpha)
    cls2_train_acc = test_accuracy(cls2_train_ts, -1, w_r, x_data, wt_alpha)

    final_accuracy = (cls1_accuracy + cls2_accuracy) / 2

    print("cls1 test acc", cls1_accuracy)
    print("cls2 test acc", cls2_accuracy)
    print("final test acc", final_accuracy)

    print("cls1 train acc", cls1_train_acc)
    print("cls2 train acc", cls2_train_acc)







