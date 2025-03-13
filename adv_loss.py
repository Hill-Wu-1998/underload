# Licensed under the Apache License, Version 2.0 (the "License");
import torch
import torchvision
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.metrics import bbox_iou, box_iou
from yolov5.utils.general import xyxy2xywh, non_max_suppression, xywh2xyxy
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


def daedalus_loss(adv_output, T_conf=0.25):
    # develop the f_3(maximize the objectness score and minimize the box area) in daedalus
    p_obj = adv_output[:,:, 4:5]    # objectness score
    p_cls = adv_output[:,:, 5:]     # class score
    bbox_wh = adv_output[:,:, 2:4]      # bounding box
    candidates = p_obj > T_conf
    num_objects = candidates.sum().item()
    # target attack class is the class with the highest score
    box_scores, _ = (p_obj * p_cls).max(2, keepdim=True)

    # Make the objectness of all detections to be 1.
    loss_1 = torch.square(box_scores-1).mean()
    # Minimising the size of all bounding box.
    loss_3 = torch.square(torch.mul(bbox_wh[..., 0], bbox_wh[..., 1])).mean()
    adv_loss = loss_1 + loss_3
    return adv_loss, num_objects


def overload_loss(adv_output, T_conf=0.25):
    # F_conf
    p_obj = adv_output[:,:, 4:5]    # objectness score, 1*25200*1 while 640*640 input
    p_cls = adv_output[:,:, 5:]     # class score, 1*25200*20
    candidates = p_obj > T_conf
    num_objects = candidates.sum().item()
    attack_conf = torch.zeros(p_obj.shape).to(p_obj.device)
    # attack_conf = torch.zeros(p_obj.squeeze(-1).shape).to(p_obj.device)
    
    cp_i, _ = (p_obj * p_cls).max(2, keepdim=True)
    c = cp_i > T_conf
    other = ~c
    
    # if p_obj*p_cls > T_conf, then F_conf item = p_obj
    attack_conf[c] = p_obj[c]
    # else p_obj*p_cls < T_conf, then F_conf item = p_obj * p_cls
    attack_conf[other] = cp_i[other]
    # F_l(·) is a monotonic increasing function. log(x), tanh(x), x2/2 and − log(1−x) are selected in OVERLOAD
    # defalut is tanh(x)
    attack_loss = torch.tanh(attack_conf)
    attack_loss = attack_loss.sum()
    return attack_loss, num_objects


def max_objects(adv_output, conf_thres=0.25, target_class=8):
    # targeted attack from phantom sponges
    p_obj = adv_output[:, :, 4:5]
    p_cls = adv_output[:, :, 5:]
    candidates = p_obj > conf_thres
    num_objects = candidates.sum().item()
    x2 = p_obj * p_cls
    conf, j = x2.max(2, keepdim=False)
    all_target_conf = x2[:, :, target_class]
    under_thr_target_conf = all_target_conf[conf < conf_thres]
    conf_avg = len(conf.view(-1)[conf.view(-1) > conf_thres]) / len(adv_output)
    # print(f"pass to NMS: {conf_avg}")
    zeros = torch.zeros(under_thr_target_conf.size()).to(adv_output.device)
    zeros.requires_grad = True
    # x3 = torch.maximum(-under_thr_target_conf + conf_thres, zeros)
    x3 = torch.clamp(-under_thr_target_conf + conf_thres, min=0)
    mean_conf = torch.sum(x3, dim=0) / (adv_output.size()[0] * adv_output.size()[1])
    return num_objects, mean_conf


def bboxes_area(output_patch, imgs, conf_thres=0.25):
    # phantom sponges loss function
    t_loss = 0.0
    preds_num = 0
    patch_size = [imgs.shape[2], imgs.shape[3]]
    xc_patch = output_patch[..., 4] > conf_thres
    not_nan_count = 0
    # For each img in the batch
    for (xi, x) in enumerate(output_patch):  # image index, image inference
        x1 = x[xc_patch[xi]].clone()
        x2 = x1[:, 5:] * x1[:, 4:5]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box_x1 = xywh2xyxy(x1[:, :4])
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height, 4096=64*64
        agnostic = True
        conf_x1, j_x1 = x2.max(1, keepdim=True)
        x1_full = torch.cat((box_x1, conf_x1, j_x1.float()), 1)[conf_x1.view(-1) > conf_thres]
        c_x1 = x1_full[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes_x1, scores_x1 = x1_full[:, :4] + c_x1, x1_full[:, 4]  # boxes (offset by class), scores
        final_preds_num = len(torchvision.ops.nms(boxes_x1, scores_x1, conf_thres)) # 使用非极大值抑制（NMS）来去除重叠的预测框，然后计算剩下的预测框的数量。
        preds_num += final_preds_num
        # calculate bboxes' area avg
        bboxes_x1_wh = xyxy2xywh(boxes_x1)[:, 2:]
        bboxes_x1_area = bboxes_x1_wh[:, 0] * bboxes_x1_wh[:, 1]
        img_loss = bboxes_x1_area.mean() / (patch_size[0] * patch_size[1])
        if not torch.isnan(img_loss):
            t_loss += img_loss
            not_nan_count += 1
    if not_nan_count == 0:
        t_loss_f = torch.tensor(torch.nan, requires_grad=True)
    else:
        t_loss_f = t_loss / not_nan_count
    return t_loss_f


def compute_iou(clean_output, adv_output, imgs):
    # phantom sponges loss function, we consider the worst case which leads to the ignore this part (l3=0)
    conf_thres = 0.25
    iou_thres = 0.45
    img_size = [imgs.shape[2], imgs.shape[3]]
    device = adv_output.device
    batch_loss = []
    gn = torch.tensor(img_size)[[1, 0, 1, 0]]
    gn = gn.to(device)
    clean_box = non_max_suppression(clean_output, conf_thres, iou_thres, classes=None)
    adv_box = non_max_suppression(adv_output, 0.001, iou_thres, classes=None)
    for (img_clean_preds, img_patch_preds) in zip(clean_box, adv_box):  # per image
        clean_preds = img_clean_preds.clone()
        adv_preds = img_patch_preds.clone()
        for clean_det in clean_preds:
            clean_clss = clean_det[5]
            clean_xyxy = torch.stack([clean_det])  # .clone()
            clean_xyxy_out = (clean_xyxy[..., :4] / gn).to(device)
            img_patch_preds_out = adv_preds[adv_preds[:, 5].view(-1) == clean_clss]
            patch_xyxy_out = (img_patch_preds_out[..., :4] / gn).to(device)
            if len(clean_xyxy_out) != 0:
                # target = box_iou(patch_xyxy_out, clean_xyxy_out)
                target = get_iou(patch_xyxy_out, clean_xyxy_out)
                if len(target) != 0:
                    target_m, _ = target.max(dim=0)
                else:
                    target_m = torch.zeros(1).to(device)
                batch_loss.append(target_m)
    one = torch.tensor(1.0).to(device)
    if len(batch_loss) == 0:
        return one
    return (one - torch.stack(batch_loss).mean())


def get_iou(bbox1, bbox2):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        bbox1: (tensor) Ground truth bounding boxes, Shape: [num_objects, 4]
        bbox2: (tensor) Prior boxes from priorbox layers, Shape: [num_priors, 4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(bbox1, bbox2)
    area_a = ((bbox1[:, 2] - bbox1[:, 0]) *
                (bbox1[:, 3] - bbox1[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((bbox2[:, 2] - bbox2[:, 0]) *
                (bbox2[:, 3] - bbox2[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
        box_a: (tensor) bounding boxes, Shape: [A,4].
        box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
        (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                        box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                        box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


class Yololoss(ComputeLoss):
    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp * iou
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)    # pi[..., 4] is the objectness score
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, (lbox, lobj, lcls)


class Yololoss_v8(v8DetectionLoss):
    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)
        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE
        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        # we need compute perturbation through the loss
        return loss.sum() * batch_size, loss  # loss(box, cls, dfl)
    
    