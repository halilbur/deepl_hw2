import torch
from torch.utils.data import DataLoader, Dataset
from medsegbench import KvasirMSBench as TargetDataset # <<< DEĞİŞİKLİK
import albumentations as A # Veri artırma için (isteğe bağlı)
from albumentations.pytorch import ToTensorV2 # Veri artırma için (isteğe bağlı)
import numpy as np
from PIL import Image # Import Image

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
        image, mask = self.dataset[idx]

        # --- Convert image/mask to NumPy arrays --- 
        # Convert PIL Image to Numpy array (HWC)
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Convert Tensor to Numpy array (CHW -> HWC)
        elif isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        # Ensure it's a numpy array if it's neither PIL nor Tensor (just in case)
        elif not isinstance(image, np.ndarray):
             raise TypeError(f"Unexpected image type received from dataset: {type(image)}")

        # Convert mask (PIL or Tensor) to Numpy array
        if isinstance(mask, Image.Image):
             mask = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        elif not isinstance(mask, np.ndarray):
             raise TypeError(f"Unexpected mask type received from dataset: {type(mask)}")
        # --- End conversion logic ---

        # Mask squeeze if needed (should happen *after* conversion to numpy)
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0) # (1, H, W) -> (H, W)

        # Albumentations expects HWC for image, HW for mask
        if self.transforms:
            # DEBUG: Print type and shape right before Albumentations call
            #print(f"DEBUG [Before Albumentations]: image type={type(image)}, shape={getattr(image, 'shape', 'N/A')}, dtype={getattr(image, 'dtype', 'N/A')}")
            #print(f"DEBUG [Before Albumentations]: mask type={type(mask)}, shape={getattr(mask, 'shape', 'N/A')}, dtype={getattr(mask, 'dtype', 'N/A')}")
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']  # Should be tensor (C, H, W) after ToTensorV2
            mask = augmented['mask']    # Should be tensor (H, W) or (1, H, W) after ToTensorV2, likely Long type
        else:
            # If no transforms, ensure correct tensor format
            if not isinstance(image, torch.Tensor):
                # Ensure image is CHW float tensor
                if image.ndim == 2: # Grayscale image (HW) -> (1HW)
                    image = np.expand_dims(image, axis=0)
                elif image.ndim == 3 and image.shape[2] in [1, 3]: # HWC -> CHW
                    image = image.transpose(2, 0, 1)
                image = torch.from_numpy(image).float()
            else:
                image = image.float()

            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).long() # HW numpy -> HW tensor
            else:
                mask = mask.long()

        # --- Ensure mask is binary (0.0 or 1.0) --- 
        if mask.max() > 1:
            #print("DEBUG: Mask max > 1, dividing by 255.") # Add a debug print here too
            mask = mask.float() / 255.0
        else:
            mask = mask.float() # Ensure it's float if already 0/1
        # --- End binary check ---

        # Ensure mask is (1, H, W) and float for BCEWithLogitsLoss
        if mask.ndim == 2:
            mask = mask.unsqueeze(0) # HW -> 1HW
        # mask = mask.float() # Already float from the check above

        # print('image shape:', image.shape, 'mask shape:', mask.shape, 'image dtype:', image.dtype, 'mask dtype:', mask.dtype)
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