import argparse
import train
import evaluate
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Segmentation Project using MedSegBench (KvasirMSBench)")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Select mode: train or test (default: train)')
    # Diğer argümanlar eklenebilir

    args = parser.parse_args()

    print(f"Selected Mode: {args.mode}")
    print("="*30)
    print("Dataset: KvasirMSBench")
    print("Model: SMP U-Net (veya benzeri)")
    print("Input Size: 256x256")
    print("="*30)

    if args.mode == 'train':
        print("Starting training process...")
        # Gerekli kütüphanelerin kontrolü train.py içinde yapılabilir
        train.main()

    elif args.mode == 'test':
        print("Starting evaluation process...")
        # Gerekli kütüphanelerin ve model dosyasının kontrolü evaluate.py içinde yapılabilir
        evaluate.main()