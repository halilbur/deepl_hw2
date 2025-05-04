import torch
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

# Güncellenmiş importlar
from model import get_smp_model # <<< DEĞİŞİKLİK
from dataset import get_loaders
from utils import load_checkpoint, calculate_metrics

# --- Ayarlar ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Testte batch size artırılabilir
NUM_WORKERS = 4
PIN_MEMORY = True
SAVE_DIR = "saved_models"
MODEL_FILENAME = "kvasir_unet_resnet34_best.pth.tar" # <<< Eğitilen modelin adı
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_FILENAME)
RESULTS_PATH = "test_results"

def evaluate(model, loader):
    """Test seti üzerinde modeli değerlendirir ve metrikleri hesaplar."""
    model.eval()
    all_preds, all_targets = [], []
    loop = tqdm(loader, leave=True, desc="Evaluating")

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE) # (N, 1, H, W) float

            with torch.cuda.amp.autocast(): # Eğitimde kullanıldıysa testte de kullanılabilir
                predictions = model(data) # (N, 1, H, W) logits

            # Tahminleri işle (sigmoid + threshold)
            preds_processed = (torch.sigmoid(predictions) > 0.5).int() # (N, 1, H, W) binary

            all_preds.append(preds_processed.cpu())
            all_targets.append(targets.cpu()) # Hedefler zaten uygun formatta

            # İsteğe bağlı: Tahmin edilen maskeleri görsel olarak kaydet
            # ...

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0).int() # Metrikler için int

    # Metrikleri hesapla
    metrics = calculate_metrics(all_preds, all_targets)

    return metrics

def main():
    # DataLoader (sadece test)
    _, _, test_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    # Modeli oluştur ve en iyi ağırlıkları yükle
    # MODEL_PATH'teki checkpoint'ten encoder bilgisini okumak daha sağlam olur
    # Ama şimdilik train.py'daki ile aynı varsayalım:
    encoder_name = 'resnet34' # Veya checkpoint'ten oku

    model = get_smp_model(
        encoder_name=encoder_name,
        encoder_weights=None, # Testte ağırlık yüklemeye gerek yok, checkpoint'ten gelecek
        in_channels=3,
        classes=1,
        activation=None
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print(f"Loading model checkpoint from {MODEL_PATH}")
        # Sadece modelin state_dict'ini yüklemek genellikle yeterlidir
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Optimizer yüklemeye gerek yok
        load_checkpoint(checkpoint, model) # Sadece state_dict yükler
        print(f"Model loaded (Epoch {checkpoint.get('epoch', 'N/A')}, Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f})")
    else:
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        return

    # Değerlendirme
    test_metrics = evaluate(model, test_loader)

    print("\n--- Test Results ---")
    for key, value in test_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # Sonuçları kaydet
    os.makedirs(RESULTS_PATH, exist_ok=True)
    metrics_df = pd.DataFrame([test_metrics])
    results_filename = f"test_metrics_{os.path.splitext(MODEL_FILENAME)[0]}.csv" # Modele özgü isim
    metrics_df.to_csv(os.path.join(RESULTS_PATH, results_filename), index=False)
    print(f"\nTest metrics saved to {os.path.join(RESULTS_PATH, results_filename)}")

    # İsteğe bağlı: Test görsellerini kaydet

if __name__ == "__main__":
    main()