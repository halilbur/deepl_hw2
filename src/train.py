import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

# Güncellenmiş importlar
from model import get_smp_model # <<< DEĞİŞİKLİK
from dataset import get_loaders
from utils import save_checkpoint, load_checkpoint, calculate_metrics, log_images_to_tensorboard

# --- Hiperparametreler ---
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 150
NUM_WORKERS = 4
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_DIR = "saved_models"
MODEL_FILENAME = "kvasir_unet_resnet34_best.pth.tar" # <<< Model adı güncellendi
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)
TENSORBOARD_LOG_DIR = "runs/kvasir_unet_resnet34" # <<< Log dizini güncellendi

# --- SMP Model Ayarları ---
ENCODER = 'resnet34' # Denenebilir: 'efficientnet-b0', 'mobilenet_v2' vb.
ENCODER_WEIGHTS = 'imagenet'

def train_one_epoch(loader, model, optimizer, loss_fn, scaler, writer, epoch):
    """Bir epoch için eğitim adımı."""
    loop = tqdm(loader, leave=True)
    model.train()
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        # Targets zaten (N, 1, H, W) float formatında gelmeli (dataset.py'den)
        targets = targets.to(device=DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
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

            with torch.cuda.amp.autocast():
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
    return avg_val_loss


def main():
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
    scaler = torch.cuda.amp.GradScaler()

    # TensorBoard
    os.makedirs(os.path.dirname(TENSORBOARD_LOG_DIR), exist_ok=True)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    # Model yükleme
    start_epoch = 0
    best_val_loss = float('inf')
    if LOAD_MODEL and os.path.exists(MODEL_PATH):
        # Önceden kaydedilmiş modeli ve en iyi loss değerini yükle
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE) # map_location eklemek iyi olabilir
        start_epoch, best_val_loss = load_checkpoint(checkpoint, model, optimizer)
        print(f"Model loaded from {MODEL_PATH}. Resuming from epoch {start_epoch}. Best validation loss: {best_val_loss:.4f}")
    else:
        print(f"Starting training from scratch or checkpoint not found at {MODEL_PATH}.")


    # Eğitim döngüsü
    os.makedirs(SAVE_DIR, exist_ok=True) # Kayıt dizinini oluştur
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, writer, epoch)
        current_val_loss = validate_model(val_loader, model, loss_fn, writer, epoch)

        # En iyi modeli kaydet
        if current_val_loss < best_val_loss:
            print(f"Validation loss improved ({best_val_loss:.4f} --> {current_val_loss:.4f}). Saving model...")
            best_val_loss = current_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'encoder': ENCODER # Model bilgilerini kaydetmek iyi olabilir
            }
            save_checkpoint(checkpoint, filename=MODEL_PATH)

    writer.close()
    print("Training finished.")
    print(f"Best validation loss achieved: {best_val_loss:.4f}")
    print(f"Best model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()