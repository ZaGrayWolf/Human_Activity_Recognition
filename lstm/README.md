# LSTM — Human Activity Recognition (UCI HAR)

**Owner:** Rishabh Jha

## Description

This folder contains the LSTM implementation for Human Activity Recognition (UCI HAR).

The model is a **2-layer LSTM** trained on preprocessed HAR windows. It uses:
- Input channels: 9
- Hidden size: 64
- LSTM layers: 2
- LSTM dropout: 0.3
- Classification classes: 6
- Output head: Dropout(0.3) + Linear(64 → 6)
- Optimizer: Adam
- Learning rate: 1e-3 by default
- Batch size: 64 by default
- Maximum epochs: 30 by default
- Early-stopping patience: 8
- Random seed: 42 by default
- Loss: CrossEntropyLoss

The six activity classes are:

1. WALKING
2. WALKING_UPSTAIRS
3. WALKING_DOWNSTAIRS
4. SITTING
5. STANDING
6. LAYING

## Files

```text
lstm/
├── README.md
├── train_lstm.py
└── results/
    ├── lstm_best.pt
    ├── lstm_confusion_matrix.png
    ├── lstm_metrics.json
    ├── lstm_training_curves.png
    └── lstm_training_log.txt
```

## Input Data

The training script expects a preprocessed NumPy archive named:

```text
preprocessed_har.npz
```

It must contain:

```text
seq_train
y_train
seq_val
y_val
seq_test
y_test
```

## Run Locally

From the repository root:

```bash
python lstm/train_lstm.py --data data/preprocessed_har.npz
```

Optional arguments:

```text
--out       Output folder
--epochs    Maximum number of epochs (default: 30)
--patience  Early-stopping patience (default: 8)
--lr        Learning rate (default: 1e-3)
--bs        Batch size (default: 64)
--seed      Random seed (default: 42)
```

Example:

```bash
python lstm/train_lstm.py   --data data/preprocessed_har.npz   --epochs 30   --patience 8   --lr 1e-3   --bs 64   --seed 42
```

## Run on Kaggle

The script automatically searches for `preprocessed_har.npz` under `/kaggle/input/`.

```python
!python train_lstm.py
```

On Kaggle, outputs are written to:

```text
/kaggle/working/results/lstm
```

## Outputs

The training script saves:

- `lstm_training_log.txt` — training and test log
- `lstm_metrics.json` — model metadata, training history, test metrics, and confusion matrix
- `lstm_confusion_matrix.png` — confusion matrix visualization
- `lstm_training_curves.png` — training/validation loss and validation accuracy
- `lstm_best.pt` — best-validation model weights

The test evaluation is performed using the checkpoint with the best validation accuracy.

## Reproducibility

The default random seed is `42`. PyTorch and NumPy seeds are set in the training script.
