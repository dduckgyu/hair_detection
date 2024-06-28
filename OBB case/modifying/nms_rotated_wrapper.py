import numpy as np
import torch
import math
from Rotated_IoU.oriented_iou_loss import cal_iou
from tqdm import tqdm
import nms_rotated_ext
from multiprocessing import Process

def obb_soft_nms(dets, scores, iou_thr, sigma=0.5, device_id=None):
    """
    RIoU NMS - iou_thr.
    Args:
        dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        scores (tensor/array): (num)
        iou_thr (float): (1)
    Returns:
        dets (tensor): (n_nms, [cx cy w h θ])
        inds (tensor): (n_nms), nms index of dets
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.numel() == 0: # len(dets)
        inds = dets_th.new_zeros(0, dtype=torch.int64)
    else:
        # same bug will happen when bboxes is too small
        too_small = dets_th[:, [2, 3]].min(1)[0] < 0.001 # [n]
        if too_small.all(): # all the bboxes is too small
            inds = dets_th.new_zeros(0, dtype=torch.int64)
        else:
            ori_inds = torch.arange(dets_th.size(0)) # 0 ~ n-1
            ori_inds = ori_inds[~too_small]
            dets_th = dets_th[~too_small] # (n_filter, 5)
            scores = scores[~too_small]

            N = dets_th.shape[0]
            if N > 1:
                scores, inds = scores.sort(descending=True)
                inds.to(device_id).view(N, 1).float()
                dets_th = dets_th[inds]

                # inds = torch.arange(0, N, dtype=torch.float).to(device_id).view(N, 1)
                
                # IoU calculate
                for i in tqdm(range(N)):
                # intermediate parameters for later parameters exchange
                    pos = i + 1

                    if i != N - 1:
                        ovr = []
                        for j in range(pos,N):
                            iou = cal_iou(dets_th[i],dets_th[j])
                            iou = iou.squeeze(0)
                            ovr.append(iou)
                                
                        # Gaussian decay
                        ovr = torch.cat(ovr, dim=0)
                        weight = torch.exp(-(ovr * ovr) / sigma)
                        scores[pos:] = weight * scores[pos:]

                # select the boxes and keep the corresponding indexes
                inds = inds[scores > iou_thr].to('cpu').numpy()
                inds = ori_inds[inds].to(device_id)

            else:
                inds = nms_rotated_ext.nms_rotated(dets_th, scores, iou_thr)
                inds = ori_inds[inds]

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds

def obb_nms(dets, scores, iou_thr, device_id=None):
    """
    RIoU NMS - iou_thr.
    Args:
        dets (tensor/array): (num, [cx cy w h θ]) θ∈[-pi/2, pi/2)
        scores (tensor/array): (num)
        iou_thr (float): (1)
    Returns:
        dets (tensor): (n_nms, [cx cy w h θ])
        inds (tensor): (n_nms), nms index of dets
    """
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.numel() == 0: # len(dets)
        inds = dets_th.new_zeros(0, dtype=torch.int64)
    else:
        # same bug will happen when bboxes is too small
        too_small = dets_th[:, [2, 3]].min(1)[0] < 0.001 # [n]
        if too_small.all(): # all the bboxes is too small
            inds = dets_th.new_zeros(0, dtype=torch.int64)
        else:
            ori_inds = torch.arange(dets_th.size(0)) # 0 ~ n-1
            ori_inds = ori_inds[~too_small]
            dets_th = dets_th[~too_small] # (n_filter, 5)
            scores = scores[~too_small]

            inds = nms_rotated_ext.nms_rotated(dets_th, scores, iou_thr)
            inds = ori_inds[inds]

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def poly_nms(dets, iou_thr, device_id=None):
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else f'cuda:{device_id}'
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError('dets must be eithr a Tensor or numpy array, '
                        f'but got {type(dets)}')

    if dets_th.device == torch.device('cpu'):
        raise NotImplementedError
    inds = nms_rotated_ext.nms_poly(dets_th.float(), iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds

if __name__ == '__main__':
    rboxes_opencv = torch.tensor(([136.6, 111.6, 200, 100, -60],
                                  [136.6, 111.6, 100, 200, -30],
                                  [100, 100, 141.4, 141.4, -45],
                                  [100, 100, 141.4, 141.4, -45]))
    rboxes_longedge = torch.tensor(([136.6, 111.6, 200, 100, -60],
                                    [136.6, 111.6, 200, 100, 120],
                                    [100, 100, 141.4, 141.4, 45],
                                    [100, 100, 141.4, 141.4, 135]))
    