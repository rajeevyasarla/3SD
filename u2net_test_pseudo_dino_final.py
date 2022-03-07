import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from functools import wraps, partial

import pdb
import numpy as np
from PIL import Image
import glob
import random

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB
import smoothness

# ------- util tool functions ----------
def exists(val):
    return val is not None
def default(val, default):
    return val if exists(val) else default
def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

def get_module_device(module):
    return next(module.parameters()).device

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class L2Norm(nn.Module):
    def forward(self, x, eps = 1e-6):
        norm = x.norm(dim = 1, keepdim = True).clamp(min = eps)
        return x / norm




def dino_loss_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):

    teacher_logits = teacher_logits.detach()
    student_probs = (student_logits / student_temp).softmax(dim = -1)
    teacher_probs = ((teacher_logits-centers) / teacher_temp).softmax(dim = -1)
    return - (teacher_probs * torch.log(student_probs + eps)).sum(dim = -1).mean()

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


# ------- 3. dino model and pseudo label generation --------
class Dino(nn.Module):
    def __init__(
        self,
        net,
        image_size,
        patch_size = 16,
        num_classes_K = 200,
        student_temp = 0.9,
        teacher_temp = 0.04,
        local_upper_crop_scale = 0.4,
        global_lower_crop_scale = 0.5,
        moving_average_decay = 0.9,
        center_moving_average_decay = 0.9,
        augment_fn = None,
        augment_fn2 = None
    ):
        super().__init__()
        self.net = net

        # default BYOL augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
        )

        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn2, DEFAULT_AUG)

        DEFAULT_AUG_BAG = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p=0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p=0.2
            ),
        )
        self.augment_bag = default(None, DEFAULT_AUG_BAG)

        # local and global crops

        self.local_crop = T.RandomResizedCrop((image_size[0], image_size[0]), scale = (0.05, local_upper_crop_scale))
        self.local_crop_bag = T.RandomResizedCrop((image_size[0], image_size[0]), scale = (0.3, 0.6))
        self.global_crop = T.RandomResizedCrop((image_size[0], image_size[0]), scale = (global_lower_crop_scale, 1.))

        self.student_encoder =  U2NET(3, 1,image_size,patch_size) if (self.net=='u2net') else  U2NETP(3, 1)
        self.teacher_encoder = U2NET(3, 1,image_size,patch_size) if (self.net=='u2net') else  U2NETP(3, 1)

        if torch.cuda.is_available():
            self.student_encoder = torch.nn.DataParallel(self.student_encoder)
            self.teacher_encoder = torch.nn.DataParallel(self.teacher_encoder)
        self.teacher_ema_updater = EMA(moving_average_decay)

        self.register_buffer('teacher_centers', torch.zeros(1, num_classes_K))
        self.register_buffer('last_teacher_centers',  torch.zeros(1, num_classes_K))

        self.register_buffer('teacher_centers_bag', torch.zeros(1,num_classes_K,image_size[0]//patch_size,image_size[0]//patch_size))
        self.register_buffer('last_teacher_centers_bag', torch.zeros(1, num_classes_K,image_size[0]//patch_size,image_size[0]//patch_size))

        self.teacher_centering_ema_updater = EMA(center_moving_average_decay)

        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

        # get device of network and make wrapper same device
        #device = get_module_device(net)
        if torch.cuda.is_available():
            self.cuda()

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(2, 3, 320,320).cuda())

    @singleton('teacher_encoder')
    def _get_teacher_encoder(self):
        teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(teacher_encoder, False)
        return teacher_encoder

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

        new_teacher_centers = self.teacher_centering_ema_updater.update_average(self.teacher_centers, self.last_teacher_centers)
        self.teacher_centers.copy_(new_teacher_centers)
        #pdb.set_trace()
        new_teacher_centers_bag = self.teacher_centering_ema_updater.update_average(self.teacher_centers_bag,self.last_teacher_centers_bag)
        self.teacher_centers_bag.copy_(new_teacher_centers_bag)

    def forward(
        self,
        x,
        return_embedding = False,
        return_projection = True,
        student_temp = None,
        teacher_temp = None
    ):
        if return_embedding:
            return self.student_encoder(x, return_projection = return_projection)

        image_one, image_two = self.augment1(x), self.augment2(x)

        local_image_one, local_image_two   = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_proj_one = self.student_encoder(local_image_one)[-1]
        student_proj_two = self.student_encoder(local_image_two)[-1]

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one = teacher_encoder(global_image_one)[-1]
            teacher_proj_two = teacher_encoder(global_image_two)[-1]
            #print(teacher_proj_one.shape)

        loss_fn_ = partial(
            dino_loss_fn,
            student_temp = default(student_temp, self.student_temp),
            teacher_temp = default(teacher_temp, self.teacher_temp),
            centers = self.teacher_centers
        )

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim = 0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss


def main():

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp
    test_datasets = ['DUTS_Test','HKU-IS','DUT','THUR']
    for dataset in test_datasets:
        image_dir = os.path.join(os.getcwd(), './../testing/', 'img',dataset)
        folder_pred = os.path.join(os.getcwd(), '../testing/','output_' + model_name + '_results' + os.sep)
        prediction_dir = os.path.join(os.getcwd(), '../testing/', 'output_' + model_name + '_results' , dataset+ os.sep)
        model_dir = os.path.join(os.getcwd(), 'saved_models', 'final_patch32_pseudo_dino_edge_pre_trans_' + model_name, model_name + '_bce_epoch_139_train_fulldino.pth')

        if (os.path.exists(folder_pred) == False):
            os.mkdir(folder_pred)
        if (os.path.exists(prediction_dir)==False):
            os.mkdir(prediction_dir)
        img_name_list = list(glob.glob(image_dir + '/*'+'.jpg')) + list(glob.glob(image_dir + '/*'+'.png'))
        #print(img_name_list)

        # --------- 2. dataloader ---------
        #1. dataloader
        test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                            lbl_name_list = [],
                                            transform=transforms.Compose([RescaleT(320),
                                                                          ToTensorLab(flag=0)])
                                            )
        test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1)

        # --------- 3. model define ---------

        dino = Dino(model_name, [320],32)
        if torch.cuda.is_available():

            dino.load_state_dict(torch.load(model_dir))
            dino.cuda()
        else:
            net.load_state_dict(torch.load(model_dir, map_location='cpu'))
        dino.train()


        # --------- 4. inference for each image ---------
        for i_test, data_test in enumerate(test_salobj_dataloader):

            #print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

            inputs_test = data_test['image']
            inputs_test = inputs_test.type(torch.FloatTensor)

            if torch.cuda.is_available():
                inputs_test = Variable(inputs_test.cuda())
            else:
                inputs_test = Variable(inputs_test)
            with torch.no_grad():
                #loss = dino(inputs_test)
                d1,d2,d3,d4,d5,d6,d7,edge,cam_map,bag_map,pred_class = dino.student_encoder(inputs_test)
            #pdb.set_trace()
            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            # save results to test_results folder
            if not os.path.exists(prediction_dir):
                os.makedirs(prediction_dir, exist_ok=True)
            save_output(img_name_list[i_test],pred,prediction_dir)
            print("inferencing:",img_name_list[i_test].split(os.sep)[-1])
            del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
