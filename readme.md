# Intro to Deep Learning Homework 2

This project is for the second homework of an Introduction to Deep Learning course. It focuses on image segmentation using a Custom U-Net model on the Kvasir-SEG dataset. The project includes scripts for training the model, evaluating its performance, and generating a summary of experiment results.

## Project Structure

```
deepl_hw2/
├── experiment_summary.md       # Markdown file summarizing experiment results
├── experiment_summary.xlsx     # Excel file summarizing experiment results
├── generate_summary_table.py   # Python script to generate the summary table
├── homework_2.pdf              # PDF document for the homework (if any)
├── readme.md                   # This readme file
├── requirements.txt            # Python dependencies
├── test.ipynb                  # Jupyter notebook for testing or exploration
├── src/                        # Source code directory
│   ├── dataset.py              # Script for dataset loading and preprocessing
│   ├── evaluate.py             # Script for evaluating the trained model
│   ├── main.py                 # Main script to run training or testing
│   ├── model.py                # Script defining the U-Net model architecture
│   ├── train.py                # Script for training the model
│   ├── utils.py                # Utility functions
│   ├── runs/                   # Directory for storing training runs/logs
│   ├── saved_models/           # Directory for storing trained model checkpoints
│   └── test_results/           # Directory for storing test set metrics and predictions
└── tb_logs/                    # Directory for TensorBoard logs
```

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # git clone <repository-url>
    # cd deepl_hw2
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    # source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes necessary packages such as PyTorch, TorchVision, Pandas, Albumentations, etc. You will also need `openpyxl` to save experiment summaries to Excel. If not already listed, you can install it via:
    ```bash
    pip install openpyxl
    ```

## Usage

The main script for running the project is `src/main.py`. You can use it to either train the model or test a pre-trained model.

### Training

To train the model, run the following command from the root directory of the project:

```bash
python src/main.py --mode train
```

Training configurations (like batch size, learning rate, epochs, etc.) can be adjusted within the `src/train.py` script or by adding more command-line arguments to `src/main.py` if implemented.

### Testing/Evaluation

To evaluate a trained model on the test set, run:

```bash
python src/main.py --mode test
```

This will load a pre-trained model (ensure the model path is correctly specified in `src/evaluate.py` or passed as an argument if implemented) and compute evaluation metrics.

### Generating Experiment Summary

To generate or update the `experiment_summary.md` and `experiment_summary.xlsx` files based on the results in `src/test_results/`, run:

```bash
python generate_summary_table.py
```
This script will iterate through the test results, compile them into a table, and save it in both Markdown and Excel formats.

## Experiment Results

A summary of the experiment results can be found in [experiment_summary.md](experiment_summary.md). This table includes metrics like Accuracy, Precision, Recall, F1-Score, and IOU for different experimental setups.

## Model

The project uses a custom U-Net architecture for semantic segmentation. Details of the model architecture can be found in `src/model.py`.

## Dataset

The Kvasir-SEG dataset is used for this project. The `src/dataset.py` script handles loading and any necessary preprocessing or augmentations.
(Details about how to obtain/prepare the Kvasir-SEG dataset should be added here if not automatically handled by the scripts).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.