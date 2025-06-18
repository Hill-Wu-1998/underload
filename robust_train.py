import torch
import copy
import random
import math
import argparse
import torch.nn as nn
from yolov5.train import main as yolov5_train
from yolov5.train import parse_opt
from yolov5.utils.callbacks import Callbacks
from adv_loss import Yololoss, Yololoss_v8
from ultralytics.models.yolo.detect import DetectionTrainer

Robust = "Underload"
attack = (1/255.0, 4)
clip_eps = 8/255.0

class Pert_Gen(Callbacks):
    # perturbation generator for yolov3 and yolov5
    pass

class Adv_Trainer_v8(DetectionTrainer):
    def preprocess_batch(self, batch):
        infer_model = copy.deepcopy(self.model)
        adv_criterion = Yololoss_v8(infer_model)
        infer_model.criterion = adv_criterion
        # reimplement preprocess_batch for adversarial training
        """Preprocesses a batch of images by scaling, converting to float and generate perturbation."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255  # keep fixed before cumpute final perturbation
        batch["img"].requires_grad_(True)
        K = range(attack[1])
        step_size = attack[0]
        pert = torch.zeros([batch["img"].shape[0], 3, batch["img"].shape[2], batch["img"].shape[3]]).to(self.device)
        pert.requires_grad_(True)
        if Robust == "MTD":
            adv_batch = batch.copy()
            for _ in K:
                adv_batch["img"] = batch["img"].clone() + pert
                _, loss = infer_model(adv_batch)
                lbox, lcls, ldfl = loss[0], loss[1], loss[2]
                lcls_grad = torch.autograd.grad(lcls, batch["img"], retain_graph=True)[0]
                cls_pert = fgsm(lcls_grad, step_size)
                lbox_grad = torch.autograd.grad(lbox, batch["img"], retain_graph=True)[0]
                box_pert = fgsm(lbox_grad, step_size)
                with torch.no_grad():
                    loc_adv_input, cls_adv_input = adv_batch.copy(), adv_batch.copy()
                    loc_adv_input["img"].add_(box_pert).clamp_(0, 1)
                    cls_adv_input["img"].add_(cls_pert).clamp_(0, 1)
                    if infer_model(loc_adv_input)[0] > infer_model(cls_adv_input)[0]:
                        pert.add_(box_pert).clamp_(-clip_eps, clip_eps)
                    else:
                        pert.add_(cls_pert).clamp_(-clip_eps, clip_eps)
            del loc_adv_input, cls_adv_input, lbox_grad, lcls_grad, box_pert, cls_pert, loss, lbox, lcls, ldfl, adv_batch
        
        elif Robust == "ODD":
            adv_batch = batch.copy()
            for _ in K:
                adv_batch["img"] = batch["img"].clone() + pert
                _, loss = infer_model(adv_batch)
                lbox, lcls, ldfl = loss[0], loss[1], loss[2]
                lcls_grad = torch.autograd.grad(lcls, batch["img"], retain_graph=True)[0]
                cls_pert = fgsm(lcls_grad, step_size)
                with torch.no_grad():
                    pert.add_(cls_pert).clamp_(-clip_eps, clip_eps)
            del lcls_grad, cls_pert, lbox, lcls, ldfl, adv_batch
            
        elif Robust == "Underload":
            adv_batch = batch.copy()
            ratio = 1.0
            targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            for _ in K:
                adv_batch["img"] = batch["img"].clone() + pert
                _, loss = infer_model(adv_batch)
                lbox, lcls, ldfl = loss[0], loss[1], loss[2]
                lcls_grad = torch.autograd.grad(lcls, batch["img"], retain_graph=True)[0]
                cls_pert = fgsm(lcls_grad, step_size)
                with torch.no_grad():
                    pert.add_(cls_pert).clamp_(-clip_eps, clip_eps)
            del lcls_grad, cls_pert, lbox, lcls, ldfl, adv_batch
            target_list = []
            for id in range(batch["img"].shape[0]):
                current_target = [target for target in targets if target[0] == id]
                if current_target:  # Check if current_target is not empty
                    target_list.append(torch.stack(current_target))
                else:
                    # Add None target for the image
                    target_list.append(None)
            
            with torch.no_grad():
                # get the mask for object perturbation
                for idx, t in enumerate(target_list):
                    if t is not None:
                        for box in t:
                            x, y, w, h = box[2:6]
                            w, h = w * ratio, h * ratio
                            x1 = max(int((x - w/2) * batch["img"].shape[2]), 0)
                            y1 = max(int((y - h/2) * batch["img"].shape[3]), 0)
                            x2 = min(int((x + w/2) * batch["img"].shape[2]), batch["img"].shape[2])
                            y2 = min(int((y + h/2) * batch["img"].shape[3]), batch["img"].shape[3])
                            pert[idx, :, y1:y2, x1:x2].fill_(0)
                    else:
                        continue
        
        elif Robust == "None":
            pass
        
        else:
            raise ValueError(f"Robust training method {Robust} is not supported.")
        
        if self.args.multi_scale:
            # code from DetectionTrainer
            imgs = batch["img"]
            sz = (
                random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride)
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        batch["img"] = batch["img"].clone().add(pert.data).clamp_(0, 1).detach()
        return batch


def fgsm(grad, step_size):
    return step_size * grad.sign()


def Underload(model, imgs, targets, device, alpha=1.0):
    # only apply object perturbation on image background
    image_width = imgs.shape[2]
    image_height = imgs.shape[3]
    model_clone = copy.deepcopy(model)  # Clone the model
    clean_input = imgs.detach().clone()
    clean_input.requires_grad_(True)
    targets_clone = targets.detach().clone()
    targets_clone = targets_clone.to(device)
    targets_clone.requires_grad_(True)
    pert = torch.zeros([clean_input.shape[0], 3, clean_input.shape[2], clean_input.shape[3]]).to(device)
    pert.requires_grad_(True)
    K = range(attack[1])
    step_size = attack[0]
    criterion = Yololoss(model)
    for _ in K:
        with torch.cuda.amp.autocast(enabled=False):
            adv_input = clean_input.clone() + pert
            adv_input.clamp_(0, 1)
            adv_output = model_clone(adv_input) # Use the cloned model for inference
            _, loss_items = criterion(adv_output, targets_clone)
            lbox, lobj, lcls = loss_items
            # compute grad of object loss
            obj_grad = torch.autograd.grad(lobj, clean_input)[0]
            obj_pert = fgsm(obj_grad, step_size)
        # compute the pert
        with torch.no_grad():
            pert.add_(obj_pert).clamp_(-clip_eps, clip_eps)
    del clean_input, adv_input, model_clone, obj_grad, obj_pert, targets_clone  # Delete the cloned model to free memory
    
    target_list = []
    for id in range(imgs.shape[0]):
        current_target = [target for target in targets if target[0] == id]
        if current_target:  # Check if current_target is not empty
            target_list.append(torch.stack(current_target))
        else:
            # Add None target for the image
            target_list.append(None)
    
    with torch.no_grad():
        for idx, t in enumerate(target_list):
            if t is not None:
                for box in t:
                    x, y, w, h = box[2:6]
                    w, h = w * alpha, h * alpha
                    x1 = max(int((x - w/2) * image_width), 0)
                    y1 = max(int((y - h/2) * image_height), 0)
                    x2 = min(int((x + w/2) * image_width), image_width)
                    y2 = min(int((y + h/2) * image_height), image_height)
                    pert[idx, :, y1:y2, x1:x2].fill_(0)
            else:
                continue
    imgs.add_(pert.data).clamp_(0, 1)


def ODD(model, imgs, targets, device):
    model_clone = copy.deepcopy(model)  # Clone the model
    clean_input = imgs.detach().clone()
    clean_input.requires_grad_(True)
    targets_clone = targets.detach().clone()
    targets_clone = targets_clone.to(device)
    targets_clone.requires_grad_(True)
    pert = torch.zeros([clean_input.shape[0], 3, clean_input.shape[2], clean_input.shape[3]]).to(device)
    pert.requires_grad_(True)
    K = range(attack[1])
    step_size = attack[0]
    criterion = Yololoss(model)
    for _ in K:
        with torch.cuda.amp.autocast(enabled=False):
            adv_input = clean_input.clone() + pert
            adv_input.clamp_(0, 1)
            adv_output = model_clone(adv_input) # Use the cloned model for inference
            _, loss_items = criterion(adv_output, targets_clone)
            lbox, lobj, lcls = loss_items
            # compute grad of object loss
            obj_grad = torch.autograd.grad(lobj, clean_input)[0]
            obj_pert = fgsm(obj_grad, step_size)
        # compute the pert
        with torch.no_grad():
            pert.add_(obj_pert).clamp_(-clip_eps, clip_eps)
    del clean_input, adv_input, targets_clone, model_clone  # Delete the cloned model to free memory
    
    imgs.add_(pert.data).clamp_(0, 1)


def MTD(model, imgs, targets, device):
    model_clone = copy.deepcopy(model)  # Clone the model
    clean_input = imgs.detach().clone()
    clean_input.requires_grad_(True)
    targets_clone = targets.detach().clone()
    targets_clone = targets_clone.to(device)
    targets_clone.requires_grad_(True)
    pert = torch.zeros([clean_input.shape[0], 3, clean_input.shape[2], clean_input.shape[3]]).to(device)
    pert.requires_grad_(True)
    K = range(attack[1])
    step_size = attack[0]
    criterion = Yololoss(model)
    for _ in K:
        with torch.cuda.amp.autocast(enabled=False):
            adv_input = clean_input.clone() + pert
            adv_input.clamp_(0, 1)
            adv_output = model_clone(adv_input) # Use the cloned model for inference
            _, loss_items = criterion(adv_output, targets_clone)
            lbox, lobj, lcls = loss_items
            # compute attacks in the location task domain
            loc_grad = torch.autograd.grad(lbox, clean_input, retain_graph=True)[0]
            loc_pert = fgsm(loc_grad, step_size)
            # compute attacks in the classification task domain
            cls_grad = torch.autograd.grad(lcls, clean_input, retain_graph=True)[0]
            cls_pert = fgsm(cls_grad, step_size)
        # compute the final attack
        with torch.no_grad():
            loc_adv_input, cls_adv_input = adv_input.clone(), adv_input.clone()
            loc_adv_input.add_(loc_pert).clamp_(0, 1)
            cls_adv_input.add_(cls_pert).clamp_(0, 1)
            if criterion(model_clone.forward(loc_adv_input), targets_clone)[0] > criterion(model_clone.forward(cls_adv_input), targets_clone)[0]:
                pert.add_(loc_pert).clamp_(-clip_eps, clip_eps)
            else:
                pert.add_(cls_pert).clamp_(-clip_eps, clip_eps)
    del clean_input, adv_input, targets_clone, model_clone
    imgs.add_(pert.data).clamp_(0, 1)


def main(opts):
    global Robust, attack, clip_eps
    
    Robust = opts.robust
    step_size_val = opts.attack_step_size / 255.0  # 转换为 0-1 范围
    attack = (step_size_val, opts.attack_iterations)
    clip_eps = opts.clip_eps / 255.0

    if Robust in ["MTD", "ODD", "Underload"]:
        dir_name = f"{Robust}_step{opts.attack_step_size}_iter{opts.attack_iterations}"
    elif Robust == "None":
        dir_name = "baseline"
    else:
        dir_name = f"{Robust}_step{opts.attack_step_size}_iter{opts.attack_iterations}"

    project_path = f"{opts.project_dir}_{dir_name}"
    
    print(f"--- Starting Training ---")
    print(f"Model: {opts.model}")
    print(f"Robust Method: {Robust}")
    print(f"Attack Params: step_size={attack[0]:.6f}, iterations={attack[1]}, clip_eps={clip_eps:.6f}")
    print(f"Saving results to: {project_path}")
    print("-------------------------")

    if opts.model in ["v3tiny", "v5"]:
        # train robust yolov5 or yolov3tiny
        opt = parse_opt(True)
        opt.data = opts.data
        opt.imgsz = opts.imgsz
        opt.cfg = opts.cfg
        opt.hyp = opts.hyp
        opt.weights = opts.weights
        opt.optimizer = opts.optimizer
        opt.device = opts.device
        opt.batch_size = opts.batch_size
        opt.epochs = opts.epochs
        opt.workers = opts.workers
        opt.project = project_path

        callbacks = Pert_Gen()
        if Robust == "MTD":
            callbacks._callbacks['on_train_batch_start'].append({'callback': MTD})
        elif Robust == "ODD":
            callbacks._callbacks['on_train_batch_start'].append({'callback': ODD})
        elif Robust == "Underload":
            callbacks._callbacks['on_train_batch_start'].append({'callback': Underload})
        elif Robust == "None":
            pass
        else:
            raise ValueError(f"Robust training method {Robust} is not supported for yolov5.")
        
        yolov5_train(opt, callbacks=callbacks)

    elif opts.model == "v8":
        train_args = dict(
            model=opts.weights, 
            data=opts.data, 
            epochs=opts.epochs, 
            imgsz=opts.imgsz, 
            optimizer=opts.optimizer,
            device=opts.device, 
            batch=opts.batch_size, 
            project=project_path,
            workers=opts.workers
        )
        adv_trainer = Adv_Trainer_v8(overrides=train_args)
        adv_trainer.train()

    else:
        raise ValueError(f"Model type '{opts.model}' is not supported. Choose 'v5' or 'v8'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Adversarial Training Script")

    parser.add_argument('--model', type=str, default='v5', choices=['v5', 'v3tiny', 'v8'], help='Model version to train (v5/v3tiny or v8)')
    parser.add_argument('--data', type=str, help='Path to data.yaml')
    parser.add_argument('--weights', type=str, help='Initial pretrained weights path')
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--imgsz', type=int, default=512, help='Train, val image size (pixels)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer to use (e.g., SGD, Adam, AdamW, auto)')
    parser.add_argument('--device', default='0', help='Cuda device, e.g. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=16, help='Max dataloader workers')
    parser.add_argument('--project-dir', type=str, default='../underload', help='Base directory to save results')
    parser.add_argument('--cfg', type=str, default='./model/yolov5s_robust.yaml', help='(YOLOv5 only) model.yaml path')
    parser.add_argument('--hyp', type=str, default='./hyps/hyp_scratch_low.yaml', help='(YOLOv5 only) hyperparameters.yaml path')
    parser.add_argument('--robust', type=str, default='Underload', choices=['None', 'MTD', 'ODD', 'Underload'], help='Robust training method')
    parser.add_argument('--attack-step-size', type=float, default=1.0, help='PGD attack step size (in 0-255 scale)')
    parser.add_argument('--attack-iterations', type=int, default=4, help='Number of PGD attack iterations')
    parser.add_argument('--clip-eps', type=float, default=4.0, help='Epsilon for clipping perturbation (in 0-255 scale)')

    args = parser.parse_args()
    main(args)
