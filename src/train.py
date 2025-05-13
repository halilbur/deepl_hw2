import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np # <<< Add numpy import
import datetime # <<< Add datetime import
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Güncellenmiş importlar
from model import CustomUNet # <<< DEĞİŞİKLİK
from dataset import get_loaders
from utils import save_checkpoint, load_checkpoint, calculate_metrics, log_images_to_tensorboard, DiceLoss

# --- Hiperparametreler ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 150
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
# --- Directory Settings (will be modified by timestamp) ---
BASE_SAVE_DIR = "saved_models"
BASE_LOG_DIR = "runs"
MODEL_NAME_BASE = "kvasir_custom_unet_bce_dice_bs8" # Updated model name

# --- train_one_epoch function ---
def train_one_epoch(loader, model, optimizer, bce_loss_fn, dice_loss_fn, loss_alpha, loss_beta, scaler, writer, epoch): # Added loss_alpha, loss_beta
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]")
    running_loss = 0.0
    running_bce_loss = 0.0 # Optional: for logging
    running_dice_loss = 0.0 # Optional: for logging

    model.train() # Ensure model is in training mode

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward
        with torch.amp.autocast(device_type=DEVICE):
            predictions = model(data) # These are raw logits
            
            # Calculate individual losses
            loss_b = bce_loss_fn(predictions, targets)
            loss_d = dice_loss_fn(predictions, targets)
            
            # Combine losses
            # You can choose weights, e.g., alpha=0.5, beta=0.5 or alpha=1.0, beta=1.0
            combined_loss = (loss_alpha * loss_b) + (loss_beta * loss_d)

        # Backward
        optimizer.zero_grad()
        scaler.scale(combined_loss).backward()
        # Optional: Gradient Clipping (from previous suggestion, if needed)
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += combined_loss.item()
        running_bce_loss += loss_b.item() # Optional
        running_dice_loss += loss_d.item() # Optional
        loop.set_postfix(loss=combined_loss.item(), bce=loss_b.item(), dice=loss_d.item())

    avg_loss = running_loss / len(loader)
    avg_bce_loss = running_bce_loss / len(loader) # Optional
    avg_dice_loss = running_dice_loss / len(loader) # Optional

    writer.add_scalar("Loss/train_combined", avg_loss, epoch)
    writer.add_scalar("Loss/train_bce", avg_bce_loss, epoch) # Optional
    writer.add_scalar("Loss/train_dice", avg_dice_loss, epoch) # Optional
    print(f"Epoch {epoch+1} - Training Combined Loss: {avg_loss:.4f} (BCE: {avg_bce_loss:.4f}, Dice: {avg_dice_loss:.4f})")
    return avg_loss

# --- validate_model function ---
def validate_model(loader, model, bce_loss_fn, dice_loss_fn, loss_alpha, loss_beta, writer, epoch): # Added loss_alpha, loss_beta
    model.eval()
    running_val_loss = 0.0
    running_val_bce_loss = 0.0 # Optional
    running_val_dice_loss = 0.0 # Optional
    all_preds_raw_list = []
    all_targets_list = []

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", leave=False)
    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            with torch.amp.autocast(device_type=DEVICE):
                predictions_raw = model(data) # Raw logits
                
                loss_b_val = bce_loss_fn(predictions_raw, targets)
                loss_d_val = dice_loss_fn(predictions_raw, targets)
                combined_loss_val = (loss_alpha * loss_b_val) + (loss_beta * loss_d_val)
            
            running_val_loss += combined_loss_val.item()
            running_val_bce_loss += loss_b_val.item() # Optional
            running_val_dice_loss += loss_d_val.item() # Optional

            all_preds_raw_list.append(torch.sigmoid(predictions_raw).cpu())
            all_targets_list.append(targets.cpu())
            loop.set_postfix(val_loss=combined_loss_val.item())

    avg_val_loss = running_val_loss / len(loader)
    avg_val_bce_loss = running_val_bce_loss / len(loader) # Optional
    avg_val_dice_loss = running_val_dice_loss / len(loader) # Optional
    
    writer.add_scalar("Loss/validation_combined", avg_val_loss, epoch)
    writer.add_scalar("Loss/validation_bce", avg_val_bce_loss, epoch) # Optional
    writer.add_scalar("Loss/validation_dice", avg_val_dice_loss, epoch) # Optional

    all_preds_raw = torch.cat(all_preds_raw_list)
    all_targets = torch.cat(all_targets_list)

    best_iou_for_epoch = -1
    optimal_threshold_for_epoch = 0.5 
    thresholds_to_test = np.arange(0.1, 0.9, 0.05)

    for th in thresholds_to_test:
        preds_binary = (all_preds_raw > th).int()
        metrics_at_th = calculate_metrics(preds_binary, all_targets)
        iou_at_th = metrics_at_th['iou']
        if iou_at_th > best_iou_for_epoch:
            best_iou_for_epoch = iou_at_th
            optimal_threshold_for_epoch = th
            
    final_preds_binary = (all_preds_raw > optimal_threshold_for_epoch).int()
    final_metrics = calculate_metrics(final_preds_binary, all_targets)

    print(f"Epoch {epoch+1} - Validation Combined Loss: {avg_val_loss:.4f} (BCE: {avg_val_bce_loss:.4f}, Dice: {avg_val_dice_loss:.4f}), Val IoU (at th={optimal_threshold_for_epoch:.2f}): {final_metrics['iou']:.4f}")
    return avg_val_loss, final_metrics, optimal_threshold_for_epoch



def main():

    LOSS_ALPHA_BCE = 0.5  # Weight for BCE Loss
    LOSS_BETA_DICE = 0.5  # Weight for Dice Loss

    global LOAD_MODEL # <<< Add this line
    global DEVICE # <<< Add this line

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{MODEL_NAME_BASE}_{timestamp}" # Use the updated MODEL_NAME_BASE

    # ... (rest of your path setup as before) ...
    TENSORBOARD_LOG_DIR = os.path.join("runs", run_name) # Assuming BASE_LOG_DIR is "runs"
    SAVE_DIR = os.path.join("saved_models", run_name) # Assuming BASE_SAVE_DIR is "saved_models"
    MODEL_FILENAME = f"{run_name}_best.pth.tar"
    MODEL_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    
    print(f"--- Starting Run: {run_name} with Combined BCE+Dice Loss (alpha={LOSS_ALPHA_BCE}, beta={LOSS_BETA_DICE}) ---")
    # ... (print other paths) ...

    model = CustomUNet(in_channels=3, num_classes=1).to(DEVICE)
    # ... (print model params) ...

    # Instantiate individual loss functions
    bce_loss_fn = nn.BCEWithLogitsLoss()
    dice_loss_fn = DiceLoss().to(DEVICE) # Make sure DiceLoss is on the correct device

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning Rate Scheduler (from previous step)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=5, verbose=True
    )

    # ... (DataLoaders, GradScaler, TensorBoard Writer, LOAD_MODEL logic as before) ...
    train_loader, val_loader, _ = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)
    scaler = torch.amp.GradScaler()
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    start_epoch = 0
    best_val_iou = 0.0
    # Add LOAD_MODEL logic here if needed, similar to previous versions

    for epoch in range(start_epoch, NUM_EPOCHS):
        # Pass individual loss functions and their weights to train_one_epoch
        train_loss_combined = train_one_epoch(
            train_loader, model, optimizer, 
            bce_loss_fn, dice_loss_fn, LOSS_ALPHA_BCE, LOSS_BETA_DICE, # Pass losses and weights
            scaler, writer, epoch
        )
        
        # Pass individual loss functions and their weights to validate_model
        val_loss_combined, val_metrics, optimal_threshold = validate_model(
            val_loader, model, 
            bce_loss_fn, dice_loss_fn, LOSS_ALPHA_BCE, LOSS_BETA_DICE, # Pass losses and weights
            writer, epoch
        )
        current_val_iou = val_metrics['iou']

        # LR Scheduler step
        scheduler.step(current_val_iou) # Monitor validation IoU

        writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch) # Log learning rate
        # ... (log other validation metrics as before: IoU, Dice, Acc) ...
        writer.add_scalar("IoU/validation", current_val_iou, epoch)
        writer.add_scalar("Dice/validation", val_metrics['dice'], epoch)
        writer.add_scalar("Accuracy/validation", val_metrics['accuracy'], epoch)
        writer.add_scalar("Optimal_Threshold/validation", optimal_threshold, epoch)


        print(f"Epoch {epoch+1} Summary: Train Combined Loss: {train_loss_combined:.4f}, Val Combined Loss: {val_loss_combined:.4f}, Val IoU: {current_val_iou:.4f} (Optimal Th: {optimal_threshold:.2f})")

        is_best = current_val_iou > best_val_iou
        if is_best:
            best_val_iou = current_val_iou
            # ... (save checkpoint logic as before, ensure 'optimal_threshold' is saved) ...
            print(f"New best validation IoU: {best_val_iou:.4f}. Saving model to {MODEL_PATH}")
            checkpoint_data = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_iou': best_val_iou,
                'best_val_loss': val_loss_combined, # Save combined val loss
                'optimal_threshold': optimal_threshold
            }
            save_checkpoint(checkpoint_data, filename=MODEL_PATH)

        # ... (log images to TensorBoard as before) ...
        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            log_images_to_tensorboard(val_loader, model, writer, epoch, device=DEVICE, num_images=5, threshold=optimal_threshold)


    writer.close()
    # ... (print final summary) ...
    print(f"--- Run {run_name} Finished ---")
    print(f"Best Validation IoU achieved: {best_val_iou:.4f}")
    print(f"Best model and logs saved in directory: {SAVE_DIR} and {TENSORBOARD_LOG_DIR}")

if __name__ == '__main__':
    main()