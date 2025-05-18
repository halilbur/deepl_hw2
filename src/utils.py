import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

def save_confusion_matrix_image(cm, path, filename="confusion_matrix.png", class_names=None):
    if class_names is None:
        class_names = ['Background', 'Polyp'] # For binary case
    
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    try:
        sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues") # fmt="d" for integer counts
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.savefig(os.path.join(path, filename))
        plt.close()
        print(f"Confusion matrix saved to {os.path.join(path, filename)}")
    except Exception as e:
        print(f"Could not save confusion matrix: {e}")

def save_evaluation_images(original_img_tensor, pred_mask_tensor, target_mask_tensor, img_idx, results_dir):
    """Saves original, predicted mask, and target mask for a single sample."""
    original_pil = torchvision.transforms.ToPILImage()(denormalize(original_img_tensor.cpu())[0])
    pred_pil = torchvision.transforms.ToPILImage()(pred_mask_tensor[0].float().cpu()) # Ensure mask is float (0 or 1)
    target_pil = torchvision.transforms.ToPILImage()(target_mask_tensor[0].float().cpu())



    # Create a combined image
    width, height = original_pil.size
    combined_img = Image.new('RGB', (width * 3, height + 30)) # +30 for titles
    
    # Add titles
    try:
        font = ImageFont.truetype("arial.ttf", 15) # Adjust font if needed
    except IOError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(combined_img)
    draw.text((10, 5), "Original Image", fill="white", font=font)
    draw.text((width + 10, 5), "Predicted Mask", fill="white", font=font)
    draw.text((width * 2 + 10, 5), "Target Mask", fill="white", font=font)

    combined_img.paste(original_pil, (0, 30))
    combined_img.paste(pred_pil.convert('RGB'), (width, 30)) # Convert mask to RGB for pasting
    combined_img.paste(target_pil.convert('RGB'), (width * 2, 30))
    
    combined_filename = f"eval_sample_{img_idx}.png"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    combined_img.save(os.path.join(results_dir, combined_filename))


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # Assuming tensor is (C, H, W) or (N, C, H, W)
    # This is a common ImageNet denormalization
    if tensor.ndim == 3:
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    elif tensor.ndim == 4:
        for i in range(tensor.size(0)):
            for t, m, s in zip(tensor[i], mean, std):
                t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer=None):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))

def calculate_metrics(preds, targets):
    """
    Segmentasyon metriklerini hesaplar: Accuracy, Precision, Recall, F1, IoU, Confusion Matrix.
    preds: Model tahminleri (Sigmoid sonrası > 0.5 yapılmış, 0 veya 1 içeren tensor).
    targets: Gerçek maskeler (0 veya 1 içeren tensor).
    """
    # Flatten tensors
    preds_flat = preds.view(-1).cpu().numpy().astype(int) # Should already be 0/1
    targets_flat = targets.view(-1).cpu().numpy().astype(bool).astype(int) # Ensure strictly 0/1

    # Calculate metrics
    accuracy = accuracy_score(targets_flat, preds_flat)
    precision = precision_score(targets_flat, preds_flat, average='binary', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='binary', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='binary', zero_division=0)
    # IoU (Jaccard Score)
    iou = jaccard_score(targets_flat, preds_flat, average='binary', zero_division=0)
    # Confusion Matrix
    cm = confusion_matrix(targets_flat, preds_flat, labels=[0, 1]) # Ensure labels are 0 and 1

    # Dice Coefficient is the same as F1 Score for binary classification
    dice = f1

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice, # <<< Add dice coefficient
        "confusion_matrix": cm
    }

def log_images_to_tensorboard(loader, model, writer, epoch, device, num_images=5, threshold=0.5, phase='Validation'):
    """
    Giriş (RGB), hedef ve tahmin görüntülerini TensorBoard'a loglar.
    Args:
        loader: DataLoader'dan bir batch almak için.
        model: Tahminleri üretmek için model.
        writer: TensorBoard yazıcısı.
        epoch: Mevcut epoch numarası.
        device: Veri ve modelin taşınacağı cihaz.
        num_images: Loglanacak örnek sayısı.
        threshold: Tahminleri binary hale getirmek için eşik değeri.
        phase: 'Validation' veya 'Test'.
    """
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        # Get a batch of data
        try:
            images, targets = next(iter(loader))
        except StopIteration:
            print("DataLoader is empty, cannot log images.")
            model.train() # Set model back to training mode
            return

        images = images.to(device)
        targets = targets.to(device) # Ensure targets are also on the correct device

        # Get predictions
        with torch.amp.autocast(device_type=device if device != "cpu" else "cuda"): # handle cpu case for autocast
            preds_raw = model(images)

    model.train() # Set model back to training mode

    # Görüntüleri CPU'ya alıp detach et
    images_cpu = images.cpu().detach()
    targets_cpu = targets.cpu().detach()
    # Tahminleri Sigmoid'den geçirip threshold uygula
    preds_processed = (torch.sigmoid(preds_raw).cpu().detach() > threshold).int().float()

    # Sadece ilk birkaç örneği (örn. ilk num_images) grid olarak logla
    actual_num_samples = min(num_images, images_cpu.shape[0])
    if actual_num_samples == 0: return

    # Giriş görüntülerini (3 kanal) ve maskeleri (1 kanal) birleştirmek için
    # Maskeleri 3 kanala kopyalayarak gri tonlamalı hale getirelim
    targets_rgb = targets_cpu[:actual_num_samples].repeat(1, 3, 1, 1)
    preds_rgb = preds_processed[:actual_num_samples].repeat(1, 3, 1, 1)

    # Grid oluştur: Input | Prediction | Target
    img_grid = torchvision.utils.make_grid(
        torch.cat((images_cpu[:actual_num_samples], preds_rgb, targets_rgb), dim=0),
        nrow=actual_num_samples # Her satırda input, pred, target yan yana olacak şekilde ayarla
    )

    writer.add_image(f'{phase}/Input_Pred_Target_Grid', img_grid, global_step=epoch)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6): # smooth factor to prevent division by zero and stabilize training
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculates Dice Loss.
        Args:
            logits: Raw, unnormalized output from the model (before sigmoid).
                    Shape: (N, C, H, W) where C is number of classes (1 for binary).
            targets: Ground truth labels.
                     Shape: (N, C, H, W), same as logits. Values should be 0 or 1.
        Returns:
            torch.Tensor: Scalar dice loss.
        """
        # Apply sigmoid to logits to get probabilities (0 to 1 range)
        probs = torch.sigmoid(logits)

        # Flatten label and prediction tensors for dot product calculation
        # For binary segmentation (C=1), view(-1) works directly.
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_coefficient = (2. * intersection + self.smooth) / \
                           (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # We want to minimize (1 - Dice Coefficient)
        return 1 - dice_coefficient