import torch
import torch.nn as nn
from torchvision.ops import roi_align

def get_landmark_bbox(lm, scale=1, target_resolution = 0):
    l_eye_id = [36, 42]
    r_eye_id = [42, 48]
    nose_id = [27, 36]
    mouth_id = [48, 68]
    bbox = []
    for _i, box_id in enumerate([mouth_id, l_eye_id, r_eye_id, nose_id]):
        box_lm = lm[:, box_id[0]:box_id[1]]
        ly, ry = torch.min(box_lm[:, :, 0], dim=1)[0], torch.max(box_lm[:, :, 0], dim=1)[0]
        lx, rx = torch.min(box_lm[:, :, 1], dim=1)[0], torch.max(box_lm[:, :, 1], dim=1)[0]  # shape: [b]
        lx, rx, ly, ry = (lx * scale).long(), (rx * scale).long(), (ly * scale).long(), (ry * scale).long()
        if _i == 1 or _i == 2:
            p = int(20 / 512 * target_resolution)
        else:
            p = int(10 / 512 * target_resolution)
        lx, rx, ly, ry = lx - p, rx + p, ly - p, ry + p
        lx, rx, ly, ry = lx.unsqueeze(1), rx.unsqueeze(1), ly.unsqueeze(1), ry.unsqueeze(1)
        bbox.append(torch.cat([ly, lx, ry, rx], dim=1).detach())
    return bbox


def local_loss(predict_images, target_images, target_keypoint, loss_fn = torch.nn.L1Loss(reduction='mean')):
    device = predict_images.device
    B, _, H, W = predict_images.shape
    assert H == W

    gt_bboxs_512 = get_landmark_bbox(target_keypoint, scale = 1, target_resolution = H)
    _i = torch.arange(B).unsqueeze(1).to(device)
    # mouth region
    gt_mouth_bbox = torch.cat([_i, gt_bboxs_512[0]], dim=1).float().to(device)
    gt_mouth = roi_align(target_images, boxes = gt_mouth_bbox, output_size=120)
    pred_mouth = roi_align(predict_images, boxes = gt_mouth_bbox, output_size=120)
    
    #left eye region
    gt_l_eye_bbox = torch.cat([_i, gt_bboxs_512[1]], dim=1).float().to(device)
    gt_l_eye = roi_align(target_images, boxes=gt_l_eye_bbox, output_size=80)
    pred_l_eye = roi_align(target_images, boxes=gt_l_eye_bbox, output_size=80)
    
    # right eye region
    gt_r_eye_bbox = torch.cat([_i, gt_bboxs_512[2]], dim=1).float().to(device)
    gt_r_eye = roi_align(target_images, boxes=gt_r_eye_bbox, output_size=80)
    pred_r_eye = roi_align(target_images, boxes=gt_r_eye_bbox, output_size=80)

    # from PIL import Image; import numpy as np
    # Image.fromarray(((gt_mouth[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/mouth.png')
    # Image.fromarray(((gt_r_eye[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/r_eye.png')
    

    local_loss = loss_fn(gt_mouth, pred_mouth) + loss_fn(gt_l_eye, pred_l_eye) + loss_fn(gt_r_eye, pred_r_eye)
    return local_loss

def lmk2mask(images, keypoint, non_bbox_rate):
    _, _, H, W = images.shape
    assert H == W
    mask = torch.ones_like(images[:, :1]) * non_bbox_rate
    gt_bboxs_512 = get_landmark_bbox(keypoint, scale = 1, target_resolution = H)
    for box in gt_bboxs_512:
        for _i in range(box.shape[0]):
            lx, rx, ly, ry = box[_i]
            mask[_i, :, rx:ry, lx:ly] = 1

    # # debug
    # from PIL import Image; import numpy as np
    # Image.fromarray(((images[0].permute(1,2,0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)).save('wasted/target.png')
    # Image.fromarray((mask[0].repeat(3,1,1).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)).save('wasted/mask_lmks.png')
    # Image.fromarray(((mask[0].repeat(3,1,1) * images[0]).permute(1,2,0).detach().cpu().numpy() * 255).astype(np.uint8)).save('wasted/mask_overlap.png')
            
    return mask

class LocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._cached = {}

    def forward(self, predict_images, target_images, target_keypoint, loss_fn = torch.nn.L1Loss(reduction='mean'), use_cache = False):
        device = predict_images.device
        B, _, H, W = predict_images.shape
        assert H == W

        if use_cache:
            gt_bboxs_512 = self._cached[H]
        else:
            gt_bboxs_512 = get_landmark_bbox(target_keypoint, scale = 1, target_resolution = H)
            self._cached[H] = gt_bboxs_512 # the cache of defferent resolutions should be saved seperately
        
        _i = torch.arange(B).unsqueeze(1).to(device)
        # mouth region
        gt_mouth_bbox = torch.cat([_i, gt_bboxs_512[0]], dim=1).float().to(device)
        gt_mouth = roi_align(target_images, boxes = gt_mouth_bbox, output_size=120)
        pred_mouth = roi_align(predict_images, boxes = gt_mouth_bbox, output_size=120)
        
        #left eye region
        gt_l_eye_bbox = torch.cat([_i, gt_bboxs_512[1]], dim=1).float().to(device)
        gt_l_eye = roi_align(target_images, boxes=gt_l_eye_bbox, output_size=80)
        pred_l_eye = roi_align(target_images, boxes=gt_l_eye_bbox, output_size=80)
        
        # right eye region
        gt_r_eye_bbox = torch.cat([_i, gt_bboxs_512[2]], dim=1).float().to(device)
        gt_r_eye = roi_align(target_images, boxes=gt_r_eye_bbox, output_size=80)
        pred_r_eye = roi_align(target_images, boxes=gt_r_eye_bbox, output_size=80)        

        local_loss = loss_fn(gt_mouth, pred_mouth) + loss_fn(gt_l_eye, pred_l_eye) + loss_fn(gt_r_eye, pred_r_eye)
        return local_loss