import torch
import cv2
import numpy as np
import shutil
import os
from torchvision import transforms
from yolov5.models.experimental import attempt_load
from yolov5.utils.dataloaders import LoadImages
from adv_loss import max_objects, bboxes_area, overload_loss, daedalus_loss
from yolov5.utils.general import non_max_suppression
from tqdm import tqdm

transp = transforms.ToPILImage()
# progress bar format
half_terminal_size = shutil.get_terminal_size((80, 20)).columns // 3
bar_format = '{l_bar}{bar:' + str(half_terminal_size) + '}{r_bar}'


def fgsm(grad, step_size):
    return step_size * grad.sign()


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """debug function, visualizes results"""
    img = img[0]
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # color is 0-1
    color = (0, 0, 255) if color is None else color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    img = img.permute(1, 2, 0).cpu().numpy()  # CHW to HWC
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB to BGR
    img = (img * 255).astype(np.uint8)  # Convert to np.uint8
    cv2.rectangle(img, c1, c2, color, lineType=cv2.LINE_AA)
    # to tensor
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
    img = torch.from_numpy(img).permute(2, 0, 1)  # HWC to CHW
    transp(img).save("demo_img.png", 'PNG')
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def daedalus(model, dataset, imgsz, patch_save_path):
    pert = torch.zeros([3, imgsz, imgsz], device=device)
    epochs = 1
    # train the patch
    for epoch in range(epochs):
        print(f"Training Patch Epoch [{epoch+1}/{epochs}]")
        pbar = tqdm(enumerate(dataset), total=dataset.nf, bar_format=bar_format)
        for idx, (path, im, im0s, vid_cap, s) in pbar:
            im = torch.from_numpy(im).to(device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            
            clean_input = im.clone()
            clean_input.requires_grad_(True)
            # model inference
            current_pert = pert.clone()
            current_pert.to(device)
            current_pert.requires_grad_(True)
            adv_input = clean_input[:] + current_pert
            adv_input.clamp_(0, 1.0)
            adv_output = model(adv_input)[0]
            adv_loss, num_objects = daedalus_loss(adv_output)  # function 3 in the paper
            # we ignore the L2 norm component to maximize the attack strength
            # and control the perturbation norm by computing the L2 norm of all pixels
            adv_loss_grad = torch.autograd.grad(adv_loss, current_pert)[0]
            next_pert = current_pert + fgsm(adv_loss_grad, 1/255.0)
            norm = torch.sum(torch.square(next_pert))
            norm = torch.sqrt(norm)
            factor = min(1, 70 / norm.item())
            next_pert *= factor
            # projection and update
            next_pert.clamp_(0, 1)
            with torch.no_grad():
                pert = next_pert.clone()
            # save patch
            if epoch == 0 and idx == 0:
                best_patch = num_objects
            else:
                if num_objects > best_patch:
                    best_patch = num_objects
                else:
                    pass
            del clean_input, adv_input, adv_output, adv_loss, num_objects, adv_loss_grad, next_pert
            pbar.set_description(f"Candidate Number: {best_patch}, Current Norm: {format(norm.item(), '.2f')}")
            pbar.update()
    pert = pert.squeeze(0)
    transp(pert).save(patch_save_path + "daedalus_patch" + str(best_patch) + ".png", 'PNG')


def phantom_sponges(model, dataset, imgsz, l1, l2, l3=0, save_path="./patch_baseline/"):
    UAP_patch = torch.zeros([3, imgsz, imgsz])
    epochs = 3
    patch_save_path = save_path
    if not os.path.exists(patch_save_path):
        os.makedirs(patch_save_path)
    else:
        # clear the patch folder
        shutil.rmtree(patch_save_path)
        os.makedirs(patch_save_path)
    # train the patch
    for epoch in range(epochs):
        print(f"Training Patch Epoch [{epoch+1}/{epochs}]")
        pbar = tqdm(enumerate(dataset), total=dataset.nf, bar_format=bar_format)
        # pbar = tqdm(enumerate(dataset), total=dataset.nf)
        for idx, (path, im, im0s, vid_cap, s) in pbar:
            im = torch.from_numpy(im).to(device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            clean_input = im.clone()
            clean_input.requires_grad_(True)
            current_patch = UAP_patch.clone()
            current_patch = current_patch.to(device)
            current_patch.requires_grad_(True)
            adv_input = clean_input[:] + current_patch
            adv_input.requires_grad_(True)
            adv_input.clamp_(0, 1.0)
            # interface
            clean_output = model(clean_input)[0]
            adv_output = model(adv_input)[0]
            # compuate attack loss
            object_num, loss_max_obj = max_objects(adv_output)
            loss_box_area = bboxes_area(adv_output, im)
            adv_loss = loss_max_obj * l1
            if not torch.isnan(loss_box_area):
                adv_loss += loss_box_area * l2
            # fix l3=0
            patch_loss_grad = torch.autograd.grad(adv_loss, current_patch)[0]
            next_patch = current_patch - fgsm(patch_loss_grad, 1/255.0)
            norm = torch.sum(torch.square(next_patch))
            norm = torch.sqrt(norm)
            factor = min(1, 70 / norm.item())
            next_patch *= factor
            # projection
            next_patch.clamp_(0, 1)
            with torch.no_grad():
                UAP_patch = next_patch.clone()
            # save patch
            if epoch == 0 and idx == 0:
                best_patch = object_num
            else:
                if object_num > best_patch:
                    best_patch = object_num
                else:
                    pass
            del clean_input, adv_input, clean_output, adv_output, loss_max_obj, loss_box_area, adv_loss, next_patch
            pbar.set_description(f"Candidate Number: {best_patch}, Current Norm: {format(norm.item(), '.2f')}")
            pbar.update()
    transp(UAP_patch).save(patch_save_path+ "ps_patch"  + str(best_patch) + ".png", 'PNG')


def overload(model, dataset, imgsz, patch_save_path):
    conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.25, 0.45, None, False, 30000
    pert = torch.zeros([3, imgsz, imgsz], device=device)
    epochs = 1
    # train the patch
    for epoch in range(epochs):
        print(f"Training Patch Epoch [{epoch+1}/{epochs}]")
        pbar = tqdm(enumerate(dataset), total=dataset.nf, bar_format=bar_format)
        for idx, (path, im, im0s, vid_cap, s) in pbar:
            im = torch.from_numpy(im).to(device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            imgs_center = (im.shape[1] // 2, im.shape[2] // 2)
            ratio = max(im0s.shape[0], im0s.shape[1]) / imgsz
            if im0s.shape[0] > im0s.shape[1]:
                img_x0 = imgs_center[0] - im0s.shape[1] / (ratio * 2)
                img_y0 = 0
            elif im0s.shape[0] < im0s.shape[1]:
                img_x0 = 0
                img_y0 = imgs_center[1] - im0s.shape[0] / (ratio * 2)
            else:
                img_x0, img_y0 = 0, 0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            clean_input = im.clone()
            clean_input.requires_grad_(True)
            
            # interface and NMS
            with torch.no_grad():
                clean_output = model(clean_input)[0]
                nms_output = non_max_suppression(clean_output, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
                num_target = nms_output[0].shape[0]
                if num_target == 0:
                    if epoch == 0 and idx == 0:
                        best_patch = 0
                    continue
                else:
                    # the number of grids is 5*5
                    grid = 5
                # different grids have different perturbation weights
                pert_grid_W = {}
                # compute the bounding box area in different grids
                for i in range(grid):
                    for j in range(grid):
                        x1, y1, x2, y2 = img_x0 + i * im0s.shape[1] / (ratio * grid), img_y0 + j * im0s.shape[0] / (ratio * grid), \
                                            img_x0 + (i + 1) * im0s.shape[1] / (ratio * grid), img_y0 + (j + 1) * im0s.shape[0] / (ratio * grid)
                        # to pytorch tensor
                        x1, y1, x2, y2 = torch.tensor([x1, y1, x2, y2]).to(device)
                        # plot_one_box([x1, y1, x2, y2], im)
                        grid_target = 0
                        # check the grid contain the target
                        for target in nms_output[0]:
                            target_x1, target_y1, target_x2, target_y2 = target[0], target[1], target[2], target[3]
                            # plot_one_box([target_x1, target_y1, target_x2, target_y2], im)
                            # if has intersection
                            inter = (torch.min(x2, target_x2) - torch.max(x1, target_x1)).clamp(0) * \
                                    (torch.min(y2, target_y2) - torch.max(y1, target_y1)).clamp(0)
                            if inter > 0:
                                grid_target += 1
                        pert_grid_W[(i, j)] = grid_target
                # compute the perturbation weights
                for region in pert_grid_W:
                    pert_grid_W[region] = 1 - (0.05 * num_target * pert_grid_W[region]) / (grid * grid)
            
            # compute the spatial attention perturbation
            with torch.set_grad_enabled(True):
                current_pert = pert.clone()
                current_pert.to(device)
                current_pert.requires_grad_(True)
                adv_input = clean_input[:] + current_pert
                adv_input.clamp_(0, 1.0)
                adv_output = model(adv_input)[0]
                attack_loss, num_objects = overload_loss(adv_output, T_conf=0.25)
                loss_box_area = bboxes_area(adv_output, im) # overload + L_area
                attack_loss += loss_box_area
                loss_grad = torch.autograd.grad(attack_loss, clean_input)[0]
                next_pert = current_pert + fgsm(loss_grad, 1/255.0)
                norm = torch.sum(torch.square(next_pert))
                norm = torch.sqrt(norm)
                factor = min(1, 70 / norm.item())
                next_pert *= factor
                next_pert.clamp_(0, 1)
            
            with torch.no_grad():
                pert = next_pert.clone()
                # apply the perturbation weights
                for region in pert_grid_W:
                    x1, y1, x2, y2 = img_x0 + region[0] * im0s.shape[1] / (ratio * grid), img_y0 + region[1] * im0s.shape[0] / (ratio * grid), \
                                    img_x0 + (region[0] + 1) * im0s.shape[1] / (ratio * grid), img_y0 + (region[1] + 1) * im0s.shape[0] / (ratio * grid)
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    pert[:, y1:y2, x1:x2] *= pert_grid_W[region]
            # save patch
            if epoch == 0 and idx == 0:
                best_patch = num_objects
            else:
                if num_objects > best_patch:
                    best_patch = num_objects
                else:
                    pass
            del clean_input, adv_input, clean_output, adv_output, attack_loss, num_objects, loss_grad, next_pert
            pbar.set_description(f"Candidate Number: {best_patch}, Current Norm: {format(norm.item(), '.2f')}")
            pbar.update()
    pert = pert.squeeze(0)
    transp(pert).save(patch_save_path + "overload_patch" + str(best_patch) + ".png", 'PNG')


if __name__=="__main__":
    imgsz = 640
    # dataset path
    img_path = ""
    l1, l2, l3 = 1, 10, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights_dir = "./weights"
    weights_list = os.listdir(weights_dir)
    dataset = LoadImages(img_path, img_size=imgsz, stride=32, auto=False)
    
    for weights in weights_list:
        if os.path.splitext(weights)[1] == '.pt':
            patch_save_path = "./patch_" + weights + "/"
            target = os.path.join(weights_dir, weights)
            if not os.path.exists(patch_save_path):
                os.makedirs(patch_save_path)
            target_model = attempt_load(target, device, False).eval()
            phantom_sponges(target_model, dataset, imgsz, l1, l2, l3, patch_save_path)
            # overload(target_model, dataset, imgsz, patch_save_path)
            # daedalus(target_model, dataset, imgsz, patch_save_path)
            print(f"Finish {weights}!")
