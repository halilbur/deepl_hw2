from torch.utils.data import DataLoader, Dataset
from medsegbench import KvasirMSBench as TargetDataset # <<< DEĞİŞİKLİK
import albumentations as A # Veri artırma için (isteğe bağlı)
from albumentations.pytorch import ToTensorV2 # Veri artırma için (isteğe bağlı)
import numpy as np
from torchvision import transforms
from PIL import Image # <<< Add this import

class SegmentationDataset(Dataset):
    def __init__(self, split='train', size=256):
        self.dataset = TargetDataset(split=split, size=size, download=True)
        self.size = size
        self.split = split

        # Define the transformation pipeline using torchvision transforms for resizing and converting to tensor
        # self.transform = transforms.Compose([
        #     transforms.Resize((self.size, self.size)),
        #     transforms.ToTensor()
        # ])
                 

        if split == 'train':
            self.transform = A.Compose([
                A.Resize(size, size),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.2),
                # A.ElasticTransform(
                #     p=0.5,          # Probability of applying the transform
                #     alpha=120,      # Intensity of the displacement field
                #     sigma=120 * 0.05, # Gaussian filter sigma. Smaller sigma means more localized changes.
                #     alpha_affine=120 * 0.03, # Intensity of the affine component of the transform
                #     border_mode=0   # Pixel extrapolation method, 0 for cv2.BORDER_CONSTANT (black)
                # ),
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

    # def __getitem__(self, idx):
    #     image, mask = self.dataset[idx] # image is PIL, mask might be numpy causing the error

    #     # Transform for the image (expects PIL, self.transform is torchvision.Compose)
    #     image = self.transform(image)

    #     # Ensure mask is a PIL Image before applying torchvision transforms to it
    #     if isinstance(mask, np.ndarray):
    #         # If mask is HxWx1 (common for single-channel images), squeeze to HxW.
    #         # PIL's fromarray handles 2D arrays well for mode 'L' (grayscale).
    #         if mask.ndim == 3 and mask.shape[2] == 1:
    #             mask = mask.squeeze(axis=2)
            
    #         # Convert NumPy array to PIL Image.
    #         # Assumes mask data is compatible (e.g., uint8).
    #         # Image.fromarray will try to infer the mode (e.g., 'L' for 2D uint8 array).
    #         mask = Image.fromarray(mask)
    #     elif not isinstance(mask, Image.Image): # If not numpy and not already PIL
    #         raise TypeError(f"Mask from dataset is of an unexpected type: {type(mask)}")

    #     # Now, mask should be a PIL Image.
    #     # Define and apply transformations for the mask
    #     mask_transform_pipeline = transforms.Compose([
    #         transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST),
    #         transforms.ToTensor()
    #     ])
    #     mask = mask_transform_pipeline(mask) # This should now work as mask is PIL

    #     # Binarize mask if needed (ensure it's 0 or 1)
    #     # ToTensor for masks usually scales to [0,1], so this thresholding is correct.
    #     mask = (mask > 0.5).float()

    #     return image, mask
    

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
