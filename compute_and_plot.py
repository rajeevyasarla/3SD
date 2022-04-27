import os
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
'''from sklearn.metrics import (precision_recall_curve,  PrecisionRecallDisplay)
from sklearn.metrics import precision_recall_curve'''
import cv2
import pdb
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob




def Sobel_op(img):
    kernel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    kernel_x_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel_x, 0), 0))  # size: (1, 1, 11,11)
    kernel_x_tensor = Variable(kernel_x_tensor.type(torch.FloatTensor), requires_grad=False)
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernel_y_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel_y, 0), 0))  # size: (1, 1, 11,11)
    kernel_y_tensor = Variable(kernel_y_tensor.type(torch.FloatTensor), requires_grad=False)
    Gx = torch.nn.functional.conv2d(img, kernel_x_tensor, padding=(1, 1))
    Gy = torch.nn.functional.conv2d(img, kernel_y_tensor, padding=(1, 1))
    G = torch.sqrt(Gx*Gx + Gy*Gy)
    G = F.tanh(G)
    kernel = np.ones((3, 3)) / 9.0
    kernel_tensor = torch.Tensor(np.expand_dims(np.expand_dims(kernel, 0), 0))  # size: (1, 1, 11,11)
    kernel_tensor = Variable(kernel_tensor.type(torch.FloatTensor), requires_grad=False)
    dilated_G = torch.clamp(torch.nn.functional.conv2d(G, kernel_tensor, padding=(1,1)), 0, 1)
    return dilated_G

def B_measure(gt,target):
    h, w = gt.shape
    gt = gt.astype(np.float32)
    target = target.astype(np.float32)
    gt = torch.from_numpy(gt)
    target = torch.from_numpy(target)
    G_gt = Sobel_op(gt.view(1,1,h,w))
    G_target = Sobel_op(target.view(1, 1, h, w))
    B = 1 - (2*(torch.sum(G_gt*G_target))/(torch.sum(G_target*G_target)+torch.sum(G_gt*G_gt)))

    return B

def E_measure(gt,target):
    gt=gt
    target=target
    #pdb.set_trace()
    phi_gt = np.subtract(gt, gt.mean())
    phi_target = np.subtract(target, target.mean())
    numerator = 2*phi_gt*phi_target
    deno = phi_gt*phi_gt + phi_target*phi_target
    phi = numerator/deno
    Enhance_phi = 0.25*(1+phi)**2
    Em = Enhance_phi.mean()
    return Em
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path,file)):
            yield file

def object_s(pred, gt):
    temp = pred[gt == 1]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    return score

def S_object(pred, gt):
    fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
    bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
    o_fg = object_s(fg, gt)
    o_bg = object_s(bg, 1 - gt)
    u = gt.mean()
    Q = u * o_fg + (1 - u) * o_bg

    return Q

def centroid( gt):
    rows, cols = gt.size()[-2:]
    gt = gt.view(rows, cols)
    cuda = False
    if gt.sum() == 0:
        if cuda:
            X = torch.eye(1).cuda() * round(cols / 2)
            Y = torch.eye(1).cuda() * round(rows / 2)
        else:
            X = torch.eye(1) * round(cols / 2)
            Y = torch.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        if cuda:
            i = torch.from_numpy(np.arange(0, cols)).cuda().float()
            j = torch.from_numpy(np.arange(0, rows)).cuda().float()
        else:
            i = torch.from_numpy(np.arange(0, cols)).float()
            j = torch.from_numpy(np.arange(0, rows)).float()
        X = torch.round((gt.sum(dim=0) * i).sum() / total + 1e-20)
        Y = torch.round((gt.sum(dim=1) * j).sum() / total + 1e-20)
    return X.long(), Y.long()

def divideGT( gt, X, Y):
    h, w = gt.size()[-2:]
    area = h * w
    gt = gt.view(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.float()
    Y = Y.float()
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def dividePrediction( pred, X, Y):
    h, w = pred.size()[-2:]
    pred = pred.view(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def ssim( pred, gt):
    gt = gt.float()
    h, w = pred.size()[-2:]
    N = h * w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

    aplha = 4 * x * y * sigma_xy
    beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q
def S_region(pred, gt):
    X, Y = centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = divideGT(gt, X, Y)
    p1, p2, p3, p4 = dividePrediction(pred, X, Y)
    Q1 = ssim(p1, gt1)
    Q2 = ssim(p2, gt2)
    Q3 = ssim(p3, gt3)
    Q4 = ssim(p4, gt4)
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4

    return Q
def S_measure(target,gt):
    alpha = 0.5
    h, w = gt.shape
    gt = torch.from_numpy(gt).type(torch.FloatTensor)
    target = torch.from_numpy(target).type(torch.FloatTensor)
    gt = gt.view(1,1,h,w)
    target = target.view(1,1,h,w)
    Q = alpha * S_object(target, gt) + (1 - alpha) * S_region(target, gt)
    return Q

gt_path = './testing/gt/'







target_path = './testing/output_u2net_results/'#'/media/rajeev/HD2/saliency_comparemethods/saliency maps_mfnnet/densenet169-salmap/'#'#'./../compare_methods/EDNS/' #'./testing_old/final_patch32_pseudo_dino_edge_pre_trans_u2net_results/'

test_datasets = ['DUTS']
output_dir = './plots/'



Num_th = 20
Threshold = 0.5
Flag_figs = 0

for dataset in test_datasets:
    name = 'exp' + '_' + dataset
    precision_list = np.zeros((Num_th, 1))
    recall_list = np.zeros((Num_th, 1))
    F_score = np.zeros((Num_th, 1))
    f1_score_list = []
    MAE_list = []
    Emeasure_list = []
    Bmeasure_list = []
    Smeasure_list = []
    count = 0
    print("----------------------------------------------------------------------------------------")
    img_name_list = list(glob.glob(gt_path + dataset + '/*' + '.jpg')) + list(glob.glob(gt_path + dataset + '/*' + '.png'))
    print("{} dataset starting, Total image : {} ".format(name,len(img_name_list)))
    for file in files(gt_path + dataset):
        gt_name = os.path.join(gt_path,dataset,file)
        target_name = os.path.join(target_path,dataset,file)
        # pdb.set_trace()
        # print(target_name)#,precision_list,recall_list)
        Gt = cv2.imread(gt_name,0)
        pred = cv2.imread(target_name,0)
        h, w = Gt.shape
        # print(w,h,pred.shape)
        pred = cv2.resize(pred,(w,h))
        Gt = Gt.astype(np.float32)
        pred = pred.astype(np.float32)


        Bmeasure_list.append(B_measure(Gt, pred))

        gt = np.zeros(Gt.shape)
        target = np.zeros(pred.shape)
        gt[Gt<Threshold] = 0
        gt[Gt>=Threshold] = 1
        target[pred<Threshold] = 0
        target[pred>=Threshold] = 1

        Emeasure_list.append(E_measure(gt, target))
        MAE_list.append(np.absolute(np.subtract(gt, target)).mean())
        Smeasure_list.append(S_measure(target, gt))
        f1_score_list.append(f1_score(gt.reshape(h*w),target.reshape(h*w),labels='binary'))

        if Flag_figs == 1:
            t_count = 0
            for th in np.linspace(0.001, 0.99, Num_th):
                gt = np.zeros(Gt.shape)
                target = np.zeros(pred.shape)
                gt[Gt < th] = 0
                gt[Gt >= th] = 1
                target[pred < th] = 0
                target[pred >= th] = 1
                precision_list[t_count] += precision_score(gt.reshape(h*w),target.reshape(h*w))
                recall_list[t_count] += recall_score(gt.reshape(h*w),target.reshape(h*w))
                #F_score[t_count] += f1_score(gt.reshape(h*w),target.reshape(h*w),labels='binary')
                t_count +=1
        count +=1
        if count%500==0:
            print(count)
        # print("{} : F1_score : {} gtsum : {} pred sum : {} ".format(file,f1_score_list[-1],gt.sum(),target.sum()))
        # pdb.set_trace()
    precision_list = precision_list/count
    recall_list = recall_list/count
    F_score = F_score/count
    MAE = sum(MAE_list)/len(MAE_list)
    F_mu = sum(f1_score_list)/len(f1_score_list)
    E_mu = sum(Emeasure_list)/len(Emeasure_list)
    B_mu = sum(Bmeasure_list)/len(Bmeasure_list)
    S_mu = sum(Smeasure_list) / len(Smeasure_list)
    np.savez('%s/%s.npz' % (output_dir, name), precision_list=precision_list, recall_list=recall_list, F_score=F_score, MAE=MAE, F_mu=F_mu, E_mu=E_mu, B_mu=B_mu, S_mu=S_mu)
    print("Dataset:{} Mean F1_Score : {}".format(dataset,F_mu))
    print("Dataset:{} Mean MAE : {}".format(dataset,MAE))
    print("Dataset:{} Mean E_measure : {}".format(dataset,E_mu))
    print("Dataset:{} Mean B_measure : {}".format(dataset,B_mu))
    print("Dataset:{} Mean S_measure : {}".format(dataset, S_mu))
    print("{} dataset done".format(dataset))
    print("----------------------------------------------------------------------------------------")
    #print("Mean precision_Score : {}".format(sum(precision_list)/len(precision_list)))
    #print("Mean recall_Score : {}".format(sum(recall_list)/len(recall_list)))
    #pr_display = PrecisionRecallDisplay(precision=precision_list, recall=recall_list).plot()
    #mpl.use('tkagg')
    plt.plot(recall_list,precision_list)
    plt.savefig(output_dir + name+'_'+'Precision_recall.png')
    plt.clf()
    plt.plot(np.linspace(0, 255, Num_th), F_score)
    plt.savefig(output_dir + name+'_'+'Fscore.png')
    plt.clf()

