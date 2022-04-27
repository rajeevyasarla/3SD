import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import random
import glob
import os
import copy

from new_data_loader import Rescale
from new_data_loader import RescaleT
from new_data_loader import RandomCrop
from new_data_loader import ToTensor
from new_data_loader import ToTensorLab
from new_data_loader import SalObjDataset
from functools import wraps, partial
import smoothness

from model import U2NET
from model import U2NETP
import pdb

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

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

#normalize camp map
def norm_cam_map(input_cam,bag_map,pred_class):
    B, C, H, W = input_cam.shape
    bag_map = F.upsample(bag_map, size=[H,W], mode='bilinear')
    cam_map = torch.zeros(B,1,H,W).cuda()
    probs = pred_class.softmax(dim = -1)
    
    for idx in range(B):
        tmp_cam_vec = input_cam[idx,:,:,:].view( C, H * W).softmax(dim = -1)
        tmp_cam_vec = tmp_cam_vec[torch.argmax(probs[idx,:]),:]
        tmp_cam_vec = tmp_cam_vec - tmp_cam_vec.min()
        tmp_cam_vec = tmp_cam_vec / (tmp_cam_vec.max())

        tmp_vec = tmp_cam_vec
        tmp_vec = tmp_vec.view(1, H, W)
        cam_map[idx,:,:,:] = tmp_vec
    cam_map = F.upsample(cam_map, size=[320,320], mode='bilinear')
    return cam_map


# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    eps = 0.000000001
    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

    return loss0, loss

def gated_edge(pred,edge):
    kernel = np.ones((11, 11)) / 121.0
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))  # size: (1, 1, 11,11)
    if torch.cuda.is_available():
        kernel_tensor = Variable(kernel_tensor.type(torch.FloatTensor).cuda(), requires_grad=False)
    dilated_pred = torch.clamp(torch.nn.functional.conv2d(pred, kernel_tensor, padding=(5, 5)), 0, 1) # performing dilation
    gated_edge_out = edge *dilated_pred
    '''B, C, H, W = gated_edge_out.shape
    gated_edge_out = gated_edge_out.view(B, C * H * W)
    gated_edge_out = gated_edge_out / (gated_edge_out.max(dim=1)[0].view(B, 1))
    gated_edge_out = gated_edge_out.view(B, C, H, W)'''
    return gated_edge_out

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

def dino_loss_bag_fn(
    teacher_logits,
    student_logits,
    teacher_temp,
    student_temp,
    centers,
    eps = 1e-20
):

    
    teacher_logits = teacher_logits.detach()
    student_probs = student_logits 
    teacher_probs = ((teacher_logits-centers))
    # creating positive and negative pairs
    student_global = F.upsample(student_logits, size=[1,1], mode='bilinear')
    B,C,H,W = student_logits.shape
    student_probs = student_probs.view(B,C,H*W).transpose(1,2)
    student_global = student_global.view(B,C,1)
    student_global = student_global/student_global.norm(dim=1).view(B,1,1)
    student_probs = student_probs/student_probs.norm(dim=-1).view(B,H*W,1)
    sim_student = torch.bmm(student_probs,student_global)
    pos_student_mask = Variable(torch.zeros(sim_student.shape).cuda(),requires_grad=False)
    pos_student_mask[sim_student>0.95*sim_student.data.detach().max()] = 1
    neg_student_mask = Variable(torch.zeros(sim_student.shape).cuda(),requires_grad=False)
    neg_student_mask[sim_student<1.1*sim_student.data.detach().min()] = 1
    neg_student_mask = torch.bmm(pos_student_mask,neg_student_mask.transpose(1,2))
    
    teacher_global = F.upsample(teacher_probs, size=[1,1], mode='bilinear')
    teacher_probs = teacher_probs.view(B,C,H*W).transpose(1,2)
    teacher_global = teacher_global.view(B,C,1)
    teacher_global = teacher_global/teacher_global.norm(dim=1).view(B,1,1)
    teacher_probs = teacher_probs/teacher_probs.norm(dim=-1).view(B,H*W,1)
    sim_teacher = torch.bmm(teacher_probs,teacher_global)
    pos_teacher_mask = Variable(torch.zeros(sim_teacher.shape).cuda(),requires_grad=False)
    pos_teacher_mask[sim_teacher>0.95*sim_teacher.data.detach().max()] = 1
    pos_teacher_mask = torch.bmm(pos_student_mask,pos_teacher_mask.transpose(1,2))
    neg_teacher_mask = Variable(torch.zeros(sim_teacher.shape).cuda(),requires_grad=False)
    neg_teacher_mask[sim_teacher<1.1*sim_teacher.data.detach().min()] = 1
    neg_teacher_mask = torch.bmm(pos_student_mask,neg_teacher_mask.transpose(1,2))
    pos_student_mask = torch.bmm(pos_student_mask,pos_student_mask.transpose(1,2))
    
    sim_student = torch.exp(torch.bmm(student_probs,student_probs.transpose(1,2))/student_temp)
    sim_teacher = torch.exp(torch.bmm(student_probs,teacher_probs.transpose(1,2))/teacher_temp)
    denom = (pos_student_mask+neg_student_mask)*sim_student + (pos_teacher_mask+neg_teacher_mask)*sim_teacher
    denom = denom.sum(dim=-1).view(B,H*W,1) +0.000001
    loss = pos_student_mask*sim_student/denom + (1-pos_student_mask)
    loss = -1*pos_student_mask*torch.log(loss) -1*pos_teacher_mask*torch.log(pos_teacher_mask*sim_teacher/denom + (1-pos_teacher_mask))
   
    return 0.003*loss.mean()

# ------- 2. set the directory of training dataset --------

model_name = 'u2net' #'u2netp'

data_dir = './data/training/DUTS/'#os.path.join(os.getcwd(), 'train_data' + os.sep)
tra_image_dir = 'img/'#os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
tra_label_dir = 'gt/'#os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)
tra_edge_dir = 'edge/'#os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)

syn_data_dir = './data/training/DUTS-TR/'#os.path.join(os.getcwd(), 'train_data' + os.sep)
syn_tra_image_dir = 'img/'#os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'im_aug' + os.sep)
syn_tra_label_dir = 'gt/'#os.path.join('DUTS', 'DUTS-TR', 'DUTS-TR', 'gt_aug' + os.sep)



image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', 'fullysup_patch32_' + model_name + os.sep)
if (os.path.isdir(model_dir)==False):
    os.mkdir(model_dir)
epoch_num = 100000
batch_size_train = 10
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = list(glob.glob(data_dir + tra_image_dir + '*' + image_ext))

tra_lbl_name_list = []
tra_edge_name_list = []
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
    tra_edge_name_list.append(data_dir + tra_edge_dir + imidx + label_ext)

syn_tra_img_name_list = list(glob.glob(syn_data_dir + syn_tra_image_dir + '*' + label_ext))
#pdb.set_trace()
syn_tra_lbl_name_list = []
for img_path in syn_tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]
    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]
    syn_tra_lbl_name_list.append(syn_data_dir + syn_tra_label_dir + imidx + label_ext)
#pdb.set_trace()
tra_img_name_list += syn_tra_img_name_list
tra_lbl_name_list += syn_tra_lbl_name_list
print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("train edges: ", len(tra_edge_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    edge_name_list=tra_edge_name_list,
    transform=transforms.Compose([
        RescaleT(352),
        RandomCrop(320),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)


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
        #print(self.teacher_centers_bag.shape)
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

    def bag_loss(self, x, return_embedding = False,return_projection = True,student_temp = None,teacher_temp = None):
        if return_embedding:
            return self.student_encoder(x, return_projection=return_projection)

        image_one, image_two = self.augment_bag(x), self.augment_bag(x)

        local_image_one, local_image_two = self.local_crop_bag(image_one), self.local_crop_bag(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_proj_one = self.student_encoder(local_image_one)[-2]
        student_proj_two = self.student_encoder(local_image_two)[-2]

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one = teacher_encoder(global_image_one)
            teacher_proj_two = teacher_encoder(global_image_two)
        #pdb.set_trace()
        teacher_logits_avg = torch.cat((teacher_proj_one[-2], teacher_proj_two[-2])).mean(dim=0)
        self.last_teacher_centers_bag.copy_(teacher_logits_avg)
        student_proj_two_glb = student_proj_two.mean(dim=-1).mean(dim=-1)
        student_proj_one_glb = student_proj_one.mean(dim=-1).mean(dim=-1)

        loss_fn_bag = partial(
            dino_loss_bag_fn,
            student_temp=default(student_temp, self.student_temp),
            teacher_temp=default(teacher_temp, self.teacher_temp),
            centers=self.teacher_centers_bag
        )
        loss_fn_ = partial(
            dino_loss_fn,
            student_temp=default(student_temp, self.student_temp),
            teacher_temp=default(teacher_temp, self.teacher_temp),
            centers=self.teacher_centers
        )
        loss = (loss_fn_bag(teacher_proj_one[-2], student_proj_two) + loss_fn_bag(teacher_proj_two[-2],
                                                                                  student_proj_one)) / 4
        loss += (loss_fn_(teacher_proj_one[-1], student_proj_two_glb) + loss_fn_(teacher_proj_two[-1],
                                                                                 student_proj_one_glb)) / 4
        return loss



# ------- 4. define model --------
# define the net
'''if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)'''
dino = Dino(model_name,[320],32)
if torch.cuda.is_available():
    dino.cuda()
    #dino = torch.nn.DataParallel(dino)


# ------- 5. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(dino.parameters(), lr=0.0006, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
dino_optimizer = optim.Adam(dino.parameters(), lr=0.0003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)




# ------- 6. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 10000 # save the model every 10000 iterations
sm_loss_weight = 0.3
smooth_loss = smoothness.smoothness_loss(size_average=True)





for epoch in range(0,epoch_num):
    #net.train()
    dino.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels, edges = data['image'], data['label'], data['edge']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        edges = edges.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v, edges_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),requires_grad=False), Variable(edges.cuda(),requires_grad=False)
        else:
            inputs_v, labels_v, edges_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False), Variable(edges, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        
        loss = 0
        loss2 = 0
        pseudo_label_gts = 0
        d0, d1, d2, d3, d4, d5, d6,  pred_edges, cam_map, bag_map, pred_class = dino.student_encoder(inputs_v)
        
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6 , labels_v)
        
        smoothLoss_cur1 = sm_loss_weight * smooth_loss(d0, T.Grayscale()(inputs_v))
        edge_loss = bce_loss(gated_edge(labels_v,pred_edges), gated_edge(labels_v,edges_v))
        loss += edge_loss + smoothLoss_cur1 
        if loss == loss:
            loss.backward()
        optimizer.step()

        # # print statistics
        if loss == loss:
            running_loss += loss.data.item()
            if loss2 >0:
                running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss, cam_map, pred_edges, edge_loss, pseudo_label_gts, pred_class, dino_loss, dino_bag_loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
        

        if ite_num % save_frq == 0:

            torch.save(dino.student_encoder.state_dict(), model_dir + model_name+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            dino.train()  # resume train
            ite_num4val = 0
    if (epoch+1) % 10 ==0:
        torch.save(dino.student_encoder.state_dict(), model_dir + model_name+"_bce_epoch_%d_train.pth" % (epoch))
        torch.save(dino.state_dict(), model_dir + model_name+"_bce_epoch_%d_train_fulldino.pth" % (epoch))
