# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import copy, deepcopy

import numpy as np
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.utils.loss import Yololoss_v8


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
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
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=batch["batch_idx"],
            cls=batch["cls"].squeeze(-1),
            bboxes=batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)



class Adv_Trainer_v8(DetectionTrainer):
    def preprocess_batch(self, batch):
        attack = (1/255.0, 4)  # PGD: step_size, K iterations
        clip_eps = attack[0] * attack[1]
        import torch
        def fgsm(grad, step_size):
            return step_size * grad.sign()
        
        # infer_model = deepcopy(self.model)
        infer_model = deepcopy(self.model.module)
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
        
        adv_batch = batch.copy()
        ratio = 0.5
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
        # transp(pert[0]).save("tmp_pert_obj.png", 'PNG')
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
        # transp(pert[0]).save("tmp_pert_underload.png", 'PNG')
        
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
