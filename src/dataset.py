import torch
from torch.utils.data import DataLoader, Dataset
from medsegbench import KvasirMSBench as TargetDataset # <<< DEĞİŞİKLİK
import albumentations as A # Veri artırma için (isteğe bağlı)
from albumentations.pytorch import ToTensorV2 # Veri artırma için (isteğe bağlı)

class SegmentationDataset(Dataset):
    def __init__(self, split='train', size=256, transforms=None): # Boyut 256 olarak ayarlandı
        """
        MedSegBench KvasirMSBench veri setini yükler.
        Args:
            split (str): 'train', 'val', veya 'test'.
            size (int): İstenen görüntü boyutu (256).
            transforms: Görüntülere ve maskelere uygulanacak transformasyonlar (albumentations önerilir).
        """
        # MedSegBench kütüphanesini kullanarak KvasirMSBench veri setini yükle
        self.dataset = TargetDataset(split=split, size=size, download=True) # <<< DEĞİŞİKLİK: TargetDataset ve size=256
        self.transforms = transforms
        print(f"KvasirMSBench '{split}' bölümü ({size}x{size}) için {len(self.dataset)} adet örnek yüklendi.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # KvasirMSBench 3 kanallı görüntü döndürür (makaleye göre)
        image = sample['image'] # Beklenen format: (3, H, W) veya (H, W, 3) - kontrol edilmeli
        mask = sample['label']  # Beklenen format: (H, W) veya (1, H, W) - kontrol edilmeli

        # MedSegBench genellikle (C, H, W) formatında tensor döndürür, ama kontrol etmekte fayda var.
        # Eğer (H, W, C) ise: image = image.permute(2, 0, 1)
        # Maskenin (H, W) formatında olduğundan emin olalım (loss için genellikle bu istenir)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0) # (1, H, W) -> (H, W)

        # Veri artırma (Albumentations örneği)
        if self.transforms:
            # Albumentations numpy array bekler, eğer tensor ise dönüştür
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).numpy() # (C, H, W) -> (H, W, C) -> numpy
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy() # (H, W) -> numpy

            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image'] # Genellikle (C, H, W) tensor döner (ToTensorV2 ile)
            mask = augmented['mask'] # Genellikle (H, W) tensor döner
        else:
            # Transformasyon yoksa PyTorch tensorlerine dönüştür
            image = torch.from_numpy(image).float() if not isinstance(image, torch.Tensor) else image.float()
            mask = torch.from_numpy(mask).long() if not isinstance(mask, torch.Tensor) else mask.long() # VEYA float() (loss'a bağlı)

        # Loss fonksiyonu (BCEWithLogitsLoss) float maske bekler ([N, 1, H, W])
        # Bu yüzden maskeyi float yapıp kanal boyutu ekleyelim
        mask = mask.unsqueeze(0).float() # (H, W) -> (1, H, W), float tipinde

        return image, mask

def get_loaders(batch_size, num_workers=4, pin_memory=True):
    """
    Eğitim, validasyon ve test için DataLoader'ları oluşturur.
    """
    # Örnek Albumentations transformasyonları (isteğe bağlı)
    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # ImageNet norm. (eğer pre-trained model ImageNet ise)
        ToTensorV2() # Numpy -> Tensor (C, H, W)
    ])
    val_transforms = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(split='train', size=256, transforms=train_transforms)
    val_dataset = SegmentationDataset(split='val', size=256, transforms=val_transforms)
    test_dataset = SegmentationDataset(split='test', size=256, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader