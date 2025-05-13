from torch.utils.data import DataLoader, Dataset
from medsegbench import KvasirMSBench as TargetDataset # <<< DEĞİŞİKLİK
import albumentations as A # Veri artırma için (isteğe bağlı)
from albumentations.pytorch import ToTensorV2 # Veri artırma için (isteğe bağlı)
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, split='train', size=256):
        self.dataset = TargetDataset(split=split, size=size, download=True)
        self.size = size
        self.split = split

        if split == 'train':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.2),
                A.Normalize(),  # Mean/std can be specified if needed
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(size, size),
                A.Normalize(),
                ToTensorV2()
            ])

        print(f"KvasirMSBench '{split}' split loaded with {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask = self.dataset[idx]

        # Convert PIL to numpy (albumentations works on numpy arrays)
        image = np.array(image)
        mask = np.array(mask)

        # Apply the same transform to both image and mask
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']           # Tensor (3, H, W)
        mask = augmented['mask'].unsqueeze(0)  # Tensor (1, H, W)

        # Binarize mask if needed
        mask = (mask > 0.5).float()

        return image, mask

def get_loaders(batch_size, num_workers=4, pin_memory=True):
    """
    Returns DataLoaders for train, val, and test splits without augmentation.
    """
    train_dataset = SegmentationDataset(split='train', size=256)
    val_dataset = SegmentationDataset(split='val', size=256)
    test_dataset = SegmentationDataset(split='test', size=256)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
