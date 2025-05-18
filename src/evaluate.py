import torch
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from model import CustomUNet
from dataset import get_loaders
from utils import load_checkpoint, calculate_metrics, save_evaluation_images, save_confusion_matrix_image
import torchvision.transforms as transforms

# --- Ayarlar ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Testte batch size artırılabilir
NUM_WORKERS = 0
PIN_MEMORY = True
RUN_NAME = "kvasir_unet_resnet34_20250504_044229" 
MODEL_PATH = os.path.join("saved_models", RUN_NAME, f"{RUN_NAME}_best.pth.tar")
RESULTS_PATH = os.path.join("test_results", RUN_NAME)
OPTIMAL_THRESHOLD = 0.5 
DEFAULT_OPTIMAL_THRESHOLD = 0.5 
SAVE_IMAGES = True 
NUM_IMAGES_TO_SAVE = 4 

# --- Main Evaluation Function ---
def evaluate(model, loader, threshold, results_dir):
    model.eval()
    all_preds_list, all_targets_list = [], []
    
    # For saving images
    saved_images_count = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(loader, desc="Evaluating")):
            data_cpu = data.clone() # Keep a CPU copy for image saving
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE) 

            with torch.amp.autocast(device_type=DEVICE):
                predictions_raw = model(data) # Raw logits
            
            predictions_sigmoid = torch.sigmoid(predictions_raw)
            predictions_binary = (predictions_sigmoid > threshold).int()

            all_preds_list.append(predictions_binary.cpu().numpy())
            all_targets_list.append(targets.cpu().numpy())

            # Save some sample images
            if SAVE_IMAGES and saved_images_count < NUM_IMAGES_TO_SAVE:
                for i in range(data.size(0)):
                    if saved_images_count < NUM_IMAGES_TO_SAVE:
                        current_img_idx_in_dataset = batch_idx * loader.batch_size + i
                        save_evaluation_images(
                            data_cpu[i:i+1], # Send as (1, C, H, W)
                            predictions_binary[i:i+1].cpu(), 
                            targets[i:i+1].cpu(), 
                            current_img_idx_in_dataset, 
                            results_dir
                        )
                        saved_images_count += 1
                    else:
                        break 
                        

    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created results directory: {results_dir}")
    if SAVE_IMAGES:
        print(f"{saved_images_count} sample images saved to {results_dir}")

    all_preds_np = np.concatenate(all_preds_list, axis=0).squeeze() # Squeeze if single class
    all_targets_np = np.concatenate(all_targets_list, axis=0).squeeze()
    
    # Ensure targets are also binary (0 or 1) if they aren't already
    all_targets_np = (all_targets_np > 0.5).astype(int)
    all_preds_np = (all_preds_np > 0.5).astype(int) # Ensure preds are also strictly 0 or 1

    metrics = calculate_metrics(torch.from_numpy(all_preds_np), torch.from_numpy(all_targets_np)) # Expects tensors
    
    # Calculate F1 score
    if 'precision' in metrics and 'recall' in metrics:
        precision = metrics['precision']
        recall = metrics['recall']
        if (precision + recall) > 0: # Avoid division by zero
            metrics['f1_score'] = 2 * (precision * recall) / (precision + recall)
        else:
            metrics['f1_score'] = 0.0 # F1 is 0 if precision and recall are both 0
    else:
        metrics['f1_score'] = float('nan') # Indicate F1 couldn't be calculated
        print("Warning: Precision and/or Recall not found in metrics. F1 Score set to NaN.")

    # Generate and save confusion matrix
    try:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        cm = sk_confusion_matrix(all_targets_np.flatten(), all_preds_np.flatten(), labels=[0, 1])
        save_confusion_matrix_image(cm, path=results_dir, filename="confusion_matrix_test.png")
    except ImportError:
        print("sklearn.metrics.confusion_matrix not available. Skipping confusion matrix image.")
    except Exception as e:
        print(f"Error generating/saving confusion matrix: {e}")

    return metrics

def main():
    if RUN_NAME == "kvasir_custom_unet_YYYYMMDD_HHMMSS":
        print("ERROR: Please update 'RUN_NAME' in src/evaluate.py to a specific trained model run.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}. Cannot evaluate.")
        print(f"Please ensure RUN_NAME ('{RUN_NAME}') is correct and the model file exists.")
        return

    # DataLoader (only test set for evaluation)
    _, _, test_loader = get_loaders(BATCH_SIZE, NUM_WORKERS, PIN_MEMORY)

    # Model
    print("Creating CustomUNet model for evaluation...")
    model = CustomUNet(in_channels=3, num_classes=1).to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CustomUNet model structure created with {num_params:,} parameters.")

    # Load checkpoint
    print(f"Loading model checkpoint from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False) # <<< Set weights_only=False
    
    # Try to load only state_dict if load_checkpoint util is not robust
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model state_dict loaded successfully.")
    elif isinstance(checkpoint, dict): # Check if it's a dict but not the expected structure
        # Fallback: try loading the whole checkpoint if it's just the state_dict
        try:
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict (assumed checkpoint was raw state_dict).")
        except RuntimeError as e:
            print(f"Error loading state_dict directly: {e}. Checkpoint keys: {checkpoint.keys()}")
            return # Cannot proceed if model weights are not loaded
    else: # If checkpoint is not a dict at all (e.g. just the model itself, older PyTorch save)
         print("Warning: Checkpoint does not seem to contain a 'state_dict'. Attempting to load entire object.")
         try:
            model = checkpoint # This might work if the entire model object was saved
            model.to(DEVICE) # Ensure it's on the correct device
            print("Loaded entire model object (older save format).")
         except Exception as e:
            print(f"Could not load model from checkpoint: {e}")
            return


    epoch_loaded = checkpoint.get('epoch', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
    best_val_iou = checkpoint.get('best_val_iou', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
    optimal_threshold = checkpoint.get('optimal_threshold', DEFAULT_OPTIMAL_THRESHOLD) if isinstance(checkpoint, dict) else DEFAULT_OPTIMAL_THRESHOLD

    print(f"Model from epoch {epoch_loaded} (Reported Best Val IoU: {best_val_iou}) loaded.")
    print(f"Using optimal threshold from checkpoint: {optimal_threshold:.2f}")

    # Perform evaluation
    print(f"\nEvaluating model: {RUN_NAME}")
    print(f"Saving evaluation results and images to: {RESULTS_PATH}")

    test_metrics = evaluate(model, test_loader, threshold=optimal_threshold, results_dir=RESULTS_PATH)

    print("\n--- Test Set Evaluation Results ---")
    print(f"  Using Threshold: {optimal_threshold:.2f}")
    print(f"  IoU: {test_metrics['iou']:.4f}")
    print(f"  Dice Coefficient: {test_metrics['dice']:.4f}")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    if 'precision' in test_metrics: print(f"  Precision: {test_metrics['precision']:.4f}")
    if 'recall' in test_metrics: print(f"  Recall (Sensitivity): {test_metrics['recall']:.4f}")
    if 'specificity' in test_metrics: print(f"  Specificity: {test_metrics['specificity']:.4f}")
    if 'f1_score' in test_metrics:
        if isinstance(test_metrics['f1_score'], float) and not np.isnan(test_metrics['f1_score']):
            print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
        else:
            print(f"  F1 Score: {test_metrics['f1_score']}") # Handles NaN or other non-float representations

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame([test_metrics])
    metrics_df['threshold_used'] = optimal_threshold
    metrics_filename = os.path.join(RESULTS_PATH, "test_set_metrics.csv")
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"Test metrics saved to {metrics_filename}")

if __name__ == '__main__':
    main()