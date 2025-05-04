import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import torchvision # Görselleştirme için

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
    Segmentasyon metriklerini hesaplar: Accuracy, Precision, Recall, F1, IoU.
    preds: Model tahminleri (Sigmoid sonrası > 0.5 yapılmış, 0 veya 1 içeren tensor).
    targets: Gerçek maskeler (0 veya 1 içeren tensor).
    """
    # CPU'ya alıp numpy'a çevir ve düzleştir
    preds_flat = preds.cpu().numpy().flatten()
    targets_flat = targets.cpu().numpy().flatten()

    accuracy = accuracy_score(targets_flat, preds_flat)
    precision = precision_score(targets_flat, preds_flat, average='binary', zero_division=0)
    recall = recall_score(targets_flat, preds_flat, average='binary', zero_division=0)
    f1 = f1_score(targets_flat, preds_flat, average='binary', zero_division=0)
    iou = jaccard_score(targets_flat, preds_flat, average='binary', zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "iou": iou}

def log_images_to_tensorboard(writer, images, targets, preds, epoch, phase='Validation'):
    """
    Giriş (RGB), hedef ve tahmin görüntülerini TensorBoard'a loglar.
    Args:
        images: Giriş görüntüleri tensorü (N, 3, H, W)
        targets: Hedef maskeler tensorü (N, 1, H, W)
        preds: Modelin ham çıktıları (logits) tensorü (N, 1, H, W)
        epoch: Mevcut epoch numarası
        phase: 'Validation' veya 'Test'
    """
    # Görüntüleri CPU'ya alıp detach et
    images = images.cpu().detach()
    targets = targets.cpu().detach() # Zaten (N, 1, H, W) ve float olmalı
    # Tahminleri Sigmoid'den geçirip threshold uygula
    preds_processed = (torch.sigmoid(preds) > 0.5).int().cpu().detach().float() # (N, 1, H, W)

    # Sadece ilk birkaç örneği (örn. ilk 4) grid olarak logla
    num_samples = min(4, images.shape[0])
    if num_samples == 0: return

    # Giriş görüntülerini (3 kanal) ve maskeleri (1 kanal) birleştirmek için
    # Maskeleri 3 kanala kopyalayarak gri tonlamalı hale getirelim
    targets_rgb = targets[:num_samples].repeat(1, 3, 1, 1) # (N, 1, H, W) -> (N, 3, H, W)
    preds_rgb = preds_processed[:num_samples].repeat(1, 3, 1, 1) # (N, 1, H, W) -> (N, 3, H, W)

    # Grid oluştur: Input | Prediction | Target
    img_grid = torchvision.utils.make_grid(
        torch.cat((images[:num_samples], preds_rgb, targets_rgb), dim=0),
        nrow=num_samples # Her satırda input, pred, target yan yana olacak şekilde ayarla
    )

    writer.add_image(f'{phase}/Input_Pred_Target_Grid', img_grid, global_step=epoch)