import torch
import torch.nn as nn
import segmentation_models_pytorch as smp # <<< SMP kütüphanesini kullanıyoruz

def get_smp_model(encoder_name='resnet34', encoder_weights='imagenet', in_channels=3, classes=1, activation=None):
    """
    segmentation-models-pytorch kullanarak bir segmentasyon modeli oluşturur.
    Args:
        encoder_name (str): Kullanılacak encoder'ın adı (örn: 'resnet34', 'efficientnet-b0').
        encoder_weights (str): Encoder için önceden eğitilmiş ağırlıklar ('imagenet' veya None).
        in_channels (int): Giriş kanalı sayısı (KvasirMSBench için 3).
        classes (int): Çıkış sınıfı sayısı (Binary segmentasyon için 1).
        activation (str): Son katman aktivasyonu ('sigmoid' veya None).
                           BCEWithLogitsLoss kullanıyorsak None olmalı.
    Returns:
        torch.nn.Module: Oluşturulan SMP modeli.
    """
    print(f"SMP Modeli Oluşturuluyor: Encoder={encoder_name}, Ağırlıklar={encoder_weights}, Giriş={in_channels}, Çıkış={classes}, Aktivasyon={activation}")
    # U-Net veya başka bir mimari seçilebilir (FPN, DeepLabV3+, vb.)
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model
