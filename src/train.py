import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np # <<< Add numpy import
import datetime # <<< Add datetime import

# Güncellenmiş importlar
from model import get_smp_model # <<< DEĞİŞİKLİK
from dataset import get_loaders
from utils import save_checkpoint, load_checkpoint, calculate_metrics, log_images_to_tensorboard

# --- Hiperparametreler ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 150
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
# --- Directory Settings (will be modified by timestamp) ---
BASE_SAVE_DIR = "saved_models"
BASE_LOG_DIR = "runs"
MODEL_NAME_BASE = "kvasir_unet_resnet34"

# --- SMP Model Ayarları ---
ENCODER = 'resnet34' # Denenebilir: 'efficientnet-b0', 'mobilenet_v2' vb.
ENCODER_WEIGHTS = 'imagenet'

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, writer, epoch):
    """Bir epoch için eğitim adımı."""
    loop = tqdm(loader, leave=True)
    model.train()
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        # DEBUG: Print type and shape of data and targets
        #print("DEBUG data type:", type(data), "shape:", getattr(data, 'shape', None))
        #print("DEBUG targets type:", type(targets), "shape:", getattr(targets, 'shape', None))
        data = data.to(device=DEVICE)
        # Targets zaten (N, 1, H, W) float formatında gelmeli (dataset.py'den)
        targets = targets.to(device=DEVICE)

        # Forward
        with torch.amp.autocast(device_type=DEVICE): # <<< DEĞİŞİKLİK: device_type eklendi
            predictions = model(data) # Çıktı: (N, 1, H, W) logits
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_description(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

def validate_model(loader, model, loss_fn, writer, epoch):
    """Validasyon verisi üzerinde modeli değerlendirir."""
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE) # (N, 1, H, W) float

            with torch.amp.autocast(device_type=DEVICE): # <<< DEĞİŞİKLİK: device_type eklendi
                predictions = model(data) # (N, 1, H, W) logits
                loss = loss_fn(predictions, targets)
                val_loss += loss.item()

            # Metrik hesaplaması için tahminleri işleyip topla
            preds_processed = (torch.sigmoid(predictions) > 0.5).int() # (N, 1, H, W) binary (0/1)

            all_preds.append(preds_processed.cpu())
            all_targets.append(targets.cpu()) # Hedefler zaten uygun formatta

            # İlk batch için görüntüleri TensorBoard'a logla
            if batch_idx == 0:
                log_images_to_tensorboard(writer, data, targets, predictions, epoch, phase='Validation') # <<< utils'deki fonksiyonu çağırır

    avg_val_loss = val_loss / len(loader)

    # Metrikleri hesapla
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0).int() # Metrik fonksiyonu int bekleyebilir
    val_metrics = calculate_metrics(all_preds, all_targets)

    # TensorBoard logları
    writer.add_scalar("Loss/validation", avg_val_loss, epoch)
    writer.add_scalar("Accuracy/validation", val_metrics["accuracy"], epoch)
    writer.add_scalar("Precision/validation", val_metrics["precision"], epoch)
    writer.add_scalar("Recall/validation", val_metrics["recall"], epoch)
    writer.add_scalar("F1-Score/validation", val_metrics["f1"], epoch)
    writer.add_scalar("IoU/validation", val_metrics["iou"], epoch)

    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
    print(f"Epoch {epoch+1} Validation Accuracy: {val_metrics['accuracy']:.4f}, IoU: {val_metrics['iou']:.4f}")

    model.train()
    # Return the entire metrics dictionary
    return avg_val_loss, val_metrics # <<< Return metrics dict

def find_best_threshold(model, loader, device):
    """Find the best probability threshold based on validation set IoU."""
    model.eval()
    all_preds_raw, all_targets = [], []
    print("\nFinding best threshold on validation set...")
    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Threshold Search"):
            data = data.to(device=device)
            targets = targets.to(device=device)
            with torch.amp.autocast(device_type=device):
                predictions = model(data) # Logits
            all_preds_raw.append(torch.sigmoid(predictions).cpu()) # Store probabilities
            all_targets.append(targets.cpu())

    all_preds_raw = torch.cat(all_preds_raw, dim=0)
    all_targets = torch.cat(all_targets, dim=0).int()

    best_iou = -1
    best_threshold = 0.5 # Default
    thresholds = np.arange(0.1, 0.9, 0.05) # Test thresholds from 0.1 to 0.85

    for threshold in thresholds:
        preds_binary = (all_preds_raw > threshold).int()
        metrics = calculate_metrics(preds_binary, all_targets)
        iou = metrics['iou']
        print(f"  Threshold: {threshold:.2f}, IoU: {iou:.4f}")
        if iou > best_iou:
            best_iou = iou
            best_threshold = threshold

    print(f"Best threshold found: {best_threshold:.2f} with Validation IoU: {best_iou:.4f}")
    model.train() # Set model back to train mode
    return best_threshold

def main():
    # --- Create unique run directory based on timestamp ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{MODEL_NAME_BASE}_{timestamp}"

    # --- Define run-specific paths ---
    TENSORBOARD_LOG_DIR = os.path.join(BASE_LOG_DIR, run_name)
    SAVE_DIR = os.path.join(BASE_SAVE_DIR, run_name)
    MODEL_FILENAME = f"{run_name}_best.pth.tar"
    MODEL_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)

    print(f"--- Starting Run: {run_name} ---")
    print(f"TensorBoard Logs: {TENSORBOARD_LOG_DIR}")
    print(f"Model Checkpoints: {SAVE_DIR}")
    print(f"Best Model Path: {MODEL_PATH}")
    # --- End of path setup ---

    # Modeli SMP kullanarak oluştur
    model = get_smp_model(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3, # <<< KvasirMSBench için 3 kanal
        classes=1,     # <<< Binary segmentasyon için 1 sınıf
        activation=None # <<< BCEWithLogitsLoss kullanacağımız için aktivasyon yok
    ).to(DEVICE)

    # Loss fonksiyonu (Binary Cross Entropy with Logits)
    loss_fn = nn.BCEWithLogitsLoss()

    # Optimizasyon algoritması
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # DataLoader'lar
    train_loader, val_loader, _ = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    # Scaler
    scaler = torch.amp.GradScaler()

    # TensorBoard
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True) # <<< Use run-specific log dir
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    # Model yükleme
    start_epoch = 0
    best_val_loss = float('inf') # Still track loss for potential reference
    best_val_iou = 0.0 # <<< Initialize best IoU
    if LOAD_MODEL:
        # NOTE: LOAD_MODEL logic might need adjustment if you want to resume
        # a *specific* previous run. Currently assumes starting fresh.
        print(f"Warning: LOAD_MODEL is True, but typically set to False for new timestamped runs.")
        # Example: If you wanted to load, you'd need to specify the exact previous MODEL_PATH
        # previous_model_path = "path/to/previous/run/model.pth.tar"
        # if os.path.exists(previous_model_path):
        #     checkpoint = torch.load(previous_model_path, map_location=DEVICE)
        #     start_epoch, best_val_loss = load_checkpoint(checkpoint, model, optimizer)
        # else:
        #     print(f"Checkpoint for LOAD_MODEL not found at specified path.")
        pass # Keep default start_epoch=0, best_val_loss=inf for now
    else:
        print(f"Starting training from scratch for run {run_name}.")

    # Eğitim döngüsü
    os.makedirs(SAVE_DIR, exist_ok=True) # <<< Use run-specific save dir
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, writer, epoch)
        # Get loss and metrics from validation
        current_val_loss, current_val_metrics = validate_model(val_loader, model, loss_fn, writer, epoch) # <<< Get metrics dict
        current_val_iou = current_val_metrics['iou'] # <<< Extract IoU

        # En iyi modeli IoU'ya göre kaydet
        is_best = current_val_iou > best_val_iou # <<< Check IoU improvement
        if is_best:
            print(f"Validation IoU improved ({best_val_iou:.4f} --> {current_val_iou:.4f}). Saving model...")
            best_val_iou = current_val_iou # <<< Update best IoU
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_iou': best_val_iou, # <<< Save best IoU
                'encoder': ENCODER
            }
            save_checkpoint(checkpoint, filename=MODEL_PATH)

    writer.close()
    print("\nTraining finished.")
    # Print best IoU achieved
    print(f"Best validation IoU achieved: {best_val_iou:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")

    # --- Find best threshold on validation set using the best model ---
    print("\nLoading best model (based on IoU) for threshold tuning...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Best model checkpoint not found at {MODEL_PATH}. Skipping threshold tuning.")
        return
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    load_checkpoint(checkpoint, model) # Load best weights from this run
    best_threshold = find_best_threshold(model, val_loader, DEVICE)
    print(f"---> Use this threshold ({best_threshold:.2f}) in evaluate.py for test set evaluation.")
    print(f"---> Remember to update MODEL_PATH in evaluate.py to: {MODEL_PATH}") # <<< Reminder

if __name__ == "__main__":
    main()