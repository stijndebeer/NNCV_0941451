import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from torchvision.datasets import Cityscapes, wrap_dataset_for_transforms_v2
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    RandomHorizontalFlip,
    RandomRotation,
    RandomApply,
    ColorJitter,
    RandomCrop,
    RandomResizedCrop,
    GaussianBlur
)

from model import Model

class CombinedLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=1.0, use_focal=False, gamma=2.0, num_classes=19, ignore_index=255):
        super().__init__()
        if use_focal:
            self.cross_entropy = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        else:
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(n_classes=num_classes, ignore_index=ignore_index)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

    def forward(self, main_output, targets):
        ce_loss = self.cross_entropy(main_output, targets)
        dice_loss = self.dice_loss(main_output, targets)

        total_loss = (self.weight_ce * ce_loss + self.weight_dice * dice_loss)
        return total_loss
    
class CombinedOCRLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_dice=1.0,  weight_aux=0.5, use_focal=False, gamma=2.0, num_classes=19, ignore_index=255):
        super().__init__()
        if use_focal:
            self.cross_entropy = FocalLoss(gamma=gamma, ignore_index=ignore_index)
        else:
            self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss(n_classes=num_classes, ignore_index=ignore_index)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.weight_aux = weight_aux

    def forward(self, main_output, aux_output, targets):
        ce_loss = self.cross_entropy(main_output, targets)
        dice_loss = self.dice_loss(main_output, targets)
        aux_loss = self.cross_entropy(aux_output, targets)

        total_loss = (self.weight_ce * ce_loss + self.weight_dice * dice_loss + self.weight_aux * aux_loss)
        return total_loss
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes=19, ignore_index=255, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply softmax to get class probabilities for Dice loss
        preds_softmax = F.softmax(preds, dim=1)  # [B, C, H, W]

        # Create one-hot encoding of targets, ignoring ignored pixels
        targets_onehot = F.one_hot(targets.clamp(0, self.n_classes - 1), num_classes=self.n_classes)  # [B, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Mask out ignored pixels
        valid_mask = (targets != self.ignore_index).float()  # [B, H, W]
        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]

        # Compute Dice loss
        intersection = (preds_softmax * targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        union = (preds_softmax * valid_mask).sum(dim=(2, 3)) + (targets_onehot * valid_mask).sum(dim=(2, 3))  # [B, C]
        dice_loss = 1 - ((2. * intersection + self.smooth) / (union + self.smooth)).mean()  # Scalar

        return dice_loss
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, input, target):
        logpt = -self.ce(input, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * (-logpt)
        return loss.mean()

# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")

    # ReduceLROnPlateau
    parser.add_argument("--lr-patience", type=int, default=8, help="Epochs with no improvement before reducing LR")
    parser.add_argument("--lr-factor", type=float, default=0.75, help="Factor to reduce LR by")
    parser.add_argument("--lr-min", type=float, default=0.0003, help="Minimum learning rate")

    #accumulating gradients
    parser.add_argument("--accumulation-steps", type=int, default=8, help="Number of steps to accumulate gradients")
    return parser

def dice_score(preds, labels, num_classes, epsilon=1e-6):
    """Computes the Dice Score for multiple classes"""
    dice_per_class = []
    
    for class_id in range(num_classes):
        pred_mask = (preds == class_id).float()
        label_mask = (labels == class_id).float()

        intersection = (pred_mask * label_mask).sum()
        union = pred_mask.sum() + label_mask.sum()

        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_per_class.append(dice.item())

    mean_dice = sum(dice_per_class) / num_classes
    return dice_per_class, mean_dice

def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean=[0.2869, 0.3251, 0.2839] #cityscapes
    std=[0.1867, 0.1908, 0.1871] #cityscapes

    # Define transforms for training (with augmentations)
    train_transform = Compose([
        ToImage(),
        # RandomCrop((256, 256), pad_if_needed=True),
        RandomResizedCrop((256, 256), scale=(0.5, 4.0)),
        RandomHorizontalFlip(p=0.5),
        RandomRotation(degrees=10),
        RandomApply([
            GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ], p=0.3),
        RandomApply([
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        ], p=0.3),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=mean, std=std),
    ])

    # Define transforms for validation (no augmentations)
    transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=mean,std=std),
    ])

    # Load datasets
    train_dataset = Cityscapes(
        args.data_dir, 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transforms=train_transform
    )
    valid_dataset = Cityscapes(
        args.data_dir, 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transforms=transform
    )

    train_dataset = wrap_dataset_for_transforms_v2(train_dataset)
    valid_dataset = wrap_dataset_for_transforms_v2(valid_dataset)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device).cuda()

    # Load pre-trained weights
    weights_path = os.path.join("weights", "modelname.pth")
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"Warning: Weights file not found at {weights_path}. Training from scratch.")

    # Define the loss function
    criterion = CombinedLoss(weight_ce=0.5, weight_dice=1.0, use_focal=False, gamma=2.0)
    # OCR
    # criterion = CombinedOCRLoss(weight_ce=0.5, weight_dice=1.0, weight_aux=0.5, use_focal=False, gamma=2.0)

    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Define the scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.lr_min
    )

    # Initialize GradScaler for mixed precision training
    scaler = torch.amp.GradScaler() #remove when not using MAP

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        optimizer.zero_grad() #gradient accumulation
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # Remove channel dimension

            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                main_out = model(images)
                loss = criterion(main_out, labels)
                #if OCR
                # main_out, aux_out = model(images)
                # loss = criterion(main_out, aux_out, labels)
            # Backward pass with scaled
            scaler.scale(loss).backward()
            # gradient accumulation
            if (i + 1) % args.accumulation_steps == 0 or (i+1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            all_dice_scores = [] #dice
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)  # Remove channel dimension

                # output, ocr_output = model(images)
                output= model(images)
                loss = criterion(output, labels)

                losses.append(loss.item())
                
                # Compute Dice Score
                dice_scores, mean_dice = dice_score(output.softmax(1).argmax(1), labels, 19)
                all_dice_scores.append(dice_scores)
                #
            
                if i == 0:
                    predictions = output.softmax(1).argmax(1)
                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)
                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            mean_dice_scores = torch.tensor(all_dice_scores).mean(dim=0).tolist() #dice
            overall_mean_dice = sum(mean_dice_scores) / 19 #dice
            print(f"Logging to W&B: valid_loss={valid_loss}, mean_dice_score={mean_dice_scores}, overall_mean_dice={overall_mean_dice}")
            wandb.log({
                "valid_loss": valid_loss,
                "mean_dice_score": overall_mean_dice, #dice
                # **{f"dice_class_{i}": score for i, score in enumerate(mean_dice_scores)} #dice per class
            }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            scheduler.step(valid_loss)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
        
    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
