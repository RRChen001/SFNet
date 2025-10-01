import cv2, torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as FF


def BGR2HSV(img, pred_img, batch):
    img_hsv = []
    pred_img_hsv = []
    for n in range(batch):
        img_ = img[n, ...] * 255
        img_ = img_.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        hsv = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
        hsv = FF.to_tensor(hsv).float()
        img_hsv.append(hsv)

        pred_img_ = pred_img[n, ...] * 255
        pred_img_ = pred_img_.permute(1, 2, 0).cpu().detach().numpy().astype(np.uint8)
        pred_hsv = cv2.cvtColor(pred_img_, cv2.COLOR_BGR2HSV)
        pred_hsv = FF.to_tensor(pred_hsv).float()
        pred_img_hsv.append(pred_hsv)
    img_hsv = torch.stack(img_hsv, dim=0)
    pred_img_hsv = torch.stack(pred_img_hsv, dim=0)
    return img_hsv, pred_img_hsv


def HSV(img, pred_img):
    batch = img.shape[0]
    hsv, pre_hsv = BGR2HSV(img, pred_img, batch)
    loss = F.mse_loss(hsv, pre_hsv, reduction='none')
    #loss = loss * (1 - edge) + 10 * loss * edge
    losses_H = loss[0].mean()
    losses_S = loss[1].mean()
    losses_V = loss[2].mean()
    losses = (losses_H + losses_S + losses_V)

    return losses_H,losses_S,losses_V,losses