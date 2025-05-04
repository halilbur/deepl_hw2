import torch
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torchvision.utils # <<< Import torchvision.utils
import matplotlib.pyplot as plt # <<< Add matplotlib import
import seaborn as sns # <<< Add seaborn import
from PIL import Image, ImageDraw, ImageFont # <<< Add PIL imports
import random # <<< Add random import

from model import get_smp_model
from dataset import get_loaders
from utils import load_checkpoint, calculate_metrics

# --- Ayarlar ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 # Testte batch size artırılabilir
NUM_WORKERS = 0
PIN_MEMORY = True
# <<< CHANGE THESE PATHS based on the output of train.py >>>
RUN_NAME = "kvasir_unet_resnet34_20250504_044617" # <<< Example: kvasir_unet_resnet34_20250504_153000
MODEL_PATH = os.path.join("saved_models", RUN_NAME, f"{RUN_NAME}_best.pth.tar")
RESULTS_PATH = os.path.join("test_results", RUN_NAME)

# <<< Add this line: Set the optimal threshold found during training >>>
OPTIMAL_THRESHOLD = 0.5 # <<< CHANGE THIS based on train.py output

# <<< Add settings for saving images >>>
SAVE_IMAGES = True # Set to False to disable image saving
NUM_IMAGES_TO_SAVE = 4 # How many sample images to save

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalizes an image tensor."""
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    return tensor * std + mean

# <<< Add function to save confusion matrix plot >>>
def save_confusion_matrix_image(cm, path, filename="confusion_matrix_normalized.jpg"):
    """Plots and saves the confusion matrix normalized by true label (rows) as a JPG image."""
    # Normalize by true label (rows)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Handle potential NaN if a row sum is 0 (no true samples for a class)
    cm_normalized = np.nan_to_num(cm_normalized)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues", # <<< Format as percentage
                xticklabels=['Background', 'Foreground'], 
                yticklabels=['Background', 'Foreground'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (Normalized by True Label - Recall)") # <<< Updated title
    save_path = os.path.join(path, filename)
    try:
        plt.savefig(save_path)
        print(f"Normalized confusion matrix plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving normalized confusion matrix plot: {e}")
    plt.close() # Close the plot to free memory

def evaluate(model, loader, threshold, results_path):
    """Test seti üzerinde modeli değerlendirir, metrikleri hesaplar ve rastgele örnek görselleri kaydeder."""
    model.eval()
    all_preds_list, all_targets_list = [], [] # Renamed to avoid conflict
    loop = tqdm(loader, leave=True, desc="Evaluating")

    # --- For Reservoir Sampling ---
    sampled_data = []
    sampled_targets = []
    sampled_preds = []
    items_processed_count = 0
    # ---

    # --- Font setup ---
    try:
        title_font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        print("Arial font not found, using default PIL font.")
        title_font = ImageFont.load_default()
    title_height = 20
    # ---

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE) # (N, 1, H, W) float

            with torch.amp.autocast(device_type=DEVICE):
                predictions = model(data) # (N, 1, H, W) logits

            preds_processed = (torch.sigmoid(predictions) > threshold).int() # (N, 1, H, W) binary

            # --- Store all preds/targets for metrics ---
            all_preds_list.append(preds_processed.cpu())
            all_targets_list.append(targets.cpu()) # Hedefler zaten uygun formatta
            # ---

            # --- Reservoir Sampling Logic for Images ---
            if SAVE_IMAGES:
                batch_size_current = data.shape[0]
                for i in range(batch_size_current):
                    current_data = data[i].cpu() # Store on CPU to save GPU memory
                    current_target = targets[i].cpu()
                    current_pred = preds_processed[i].cpu()

                    if len(sampled_data) < NUM_IMAGES_TO_SAVE:
                        sampled_data.append(current_data)
                        sampled_targets.append(current_target)
                        sampled_preds.append(current_pred)
                    else:
                        # Probability of replacing an existing sample decreases as more items are seen
                        replace_idx = random.randint(0, items_processed_count)
                        if replace_idx < NUM_IMAGES_TO_SAVE:
                            sampled_data[replace_idx] = current_data
                            sampled_targets[replace_idx] = current_target
                            sampled_preds[replace_idx] = current_pred
                    items_processed_count += 1
            # --- End Reservoir Sampling ---

    # --- Calculate Metrics ---
    all_preds = torch.cat(all_preds_list, dim=0)
    all_targets = torch.cat(all_targets_list, dim=0).int() # Metrikler için int
    metrics = calculate_metrics(all_preds, all_targets)
    # ---

    # --- Save Sampled Images (moved outside the loop) ---
    if SAVE_IMAGES and sampled_data:
        print(f"\nSaving {len(sampled_data)} random sample images...")
        os.makedirs(results_path, exist_ok=True) # Ensure dir exists

        for img_idx, (s_data, s_target, s_pred) in enumerate(zip(sampled_data, sampled_targets, sampled_preds)):
            # --- Prepare individual images ---
            original_img_tensor = denormalize(s_data.unsqueeze(0)) # Add batch dim for denormalize
            original_img_tensor = original_img_tensor.contiguous()
            original_pil = Image.fromarray((original_img_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8))

            target_mask_tensor = s_target.float() # Shape (1, H, W)
            # DEBUG print removed for final version
            # print(f"DEBUG Target Tensor Min: {target_mask_tensor.min()}, Max: {target_mask_tensor.max()}, Shape: {target_mask_tensor.shape}")
            target_numpy = target_mask_tensor.squeeze().numpy() # Squeeze to (H, W)
            target_numpy_uint8 = (target_numpy * 255).astype(np.uint8)
            target_pil = Image.fromarray(target_numpy_uint8, 'L')

            pred_mask_tensor = s_pred.float() # Shape (1, H, W)
            pred_numpy = pred_mask_tensor.squeeze().numpy() # Squeeze to (H, W)
            pred_numpy_uint8 = (pred_numpy * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_numpy_uint8, 'L')

            # --- Create combined image with titles --- 
            width, height = original_pil.size
            combined_width = width * 3
            combined_height = height + title_height
            combined_img = Image.new('RGB', (combined_width, combined_height), color='white')
            draw = ImageDraw.Draw(combined_img)

            input_title = "Input"
            pred_title = "Prediction"
            target_title = "Target"
            
            input_text_width = draw.textlength(input_title, font=title_font)
            pred_text_width = draw.textlength(pred_title, font=title_font)
            target_text_width = draw.textlength(target_title, font=title_font)

            draw.text(((width - input_text_width) // 2 + width * 0, 2), input_title, fill="black", font=title_font)
            draw.text(((width - pred_text_width) // 2 + width * 1, 2), pred_title, fill="black", font=title_font)
            draw.text(((width - target_text_width) // 2 + width * 2, 2), target_title, fill="black", font=title_font)

            combined_img.paste(original_pil, (width * 0, title_height))
            combined_img.paste(pred_pil.convert('RGB'), (width * 1, title_height))
            combined_img.paste(target_pil.convert('RGB'), (width * 2, title_height))

            combined_filename = f"random_sample_{img_idx}_combined_titled.jpg" # Changed filename
            combined_img.save(os.path.join(results_path, combined_filename))
        print(f"Sample images saved to {results_path}")
    # --- End Image Saving ---

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
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        load_checkpoint(checkpoint, model)

        # --- Updated print statement ---
        epoch_loaded = checkpoint.get('epoch', 'N/A')
        # Check for IoU first, then loss
        best_metric_val = checkpoint.get('best_val_iou', checkpoint.get('best_val_loss', 'N/A'))
        metric_name = "IoU" if 'best_val_iou' in checkpoint else "Loss"

        # Format only if the value is a number
        if isinstance(best_metric_val, (int, float)):
            print(f"Model loaded (Epoch {epoch_loaded}, Best Val {metric_name}: {best_metric_val:.4f})")
        else:
            print(f"Model loaded (Epoch {epoch_loaded}, Best Val {metric_name}: {best_metric_val})")
        # --- End of update ---

    else:
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        return

    # Değerlendirme
    print(f"\nEvaluating model from: {MODEL_PATH}") # <<< Print model path
    print(f"Using threshold: {OPTIMAL_THRESHOLD:.2f} for evaluation")
    # Pass results_path to evaluate function
    test_metrics = evaluate(model, test_loader, threshold=OPTIMAL_THRESHOLD, results_path=RESULTS_PATH)

    print("\n--- Test Results ---")
    # Extract confusion matrix before iterating
    cm = test_metrics.pop("confusion_matrix", None) # Remove cm from dict, store it

    for key, value in test_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    # Print Confusion Matrix and save plot
    if cm is not None:
        print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
        print("          Predicted 0  Predicted 1")
        print(f"Actual 0: {cm[0, 0]:>10d}  {cm[0, 1]:>10d}")
        print(f"Actual 1: {cm[1, 0]:>10d}  {cm[1, 1]:>10d}")
        tn, fp, fn, tp = cm.ravel()
        print(f"(TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp})")
        # <<< Save CM plot >>>
        save_confusion_matrix_image(cm, RESULTS_PATH)

    # Sonuçları kaydet
    os.makedirs(RESULTS_PATH, exist_ok=True) # <<< Use run-specific results path
    # Add confusion matrix elements back for saving if needed, or save separately
    test_metrics_to_save = test_metrics.copy()
    if cm is not None:
        test_metrics_to_save['TN'] = tn
        test_metrics_to_save['FP'] = fp
        test_metrics_to_save['FN'] = fn
        test_metrics_to_save['TP'] = tp

    metrics_df = pd.DataFrame([test_metrics_to_save])
    results_filename = f"test_metrics.csv"
    results_filepath = os.path.join(RESULTS_PATH, results_filename)
    metrics_df.to_csv(results_filepath, index=False)
    print(f"\nTest metrics saved to {results_filepath}")

    if SAVE_IMAGES:
        print(f"Sample prediction images saved to {RESULTS_PATH}")

    # İsteğe bağlı: Test görsellerini kaydet

if __name__ == "__main__":
    main()