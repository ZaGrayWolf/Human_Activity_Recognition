# Human Activity Recognition (UCI HAR) – MLP / 1D-CNN / LSTM / Transformer
ICT-4442 Deep Learning Mini Project

## Setup
1. `pip install torch scikit-learn numpy matplotlib`
2. Download the UCI HAR dataset (https://archive.ics.uci.edu/dataset/240) and extract it to `data_raw/`
   so that `data_raw/train/Inertial Signals/body_acc_x_train.txt` exists.

## Run
```
python src/train.py --model mlp          # or cnn | lstm | transformer
python src/train.py --model cnn --seed 1
python src/make_figures.py               # confusion matrices, curves, multi-seed aggregate
```
Seeds used in the interim report: 42, 1, 2. Results are written to `results/`.

## Layout / ownership
| File | Owner |
|---|---|
| src/models/mlp.py | Ayush Bansal |
| src/models/cnn.py | Sree Sai Thottempudi |
| src/models/lstm.py | Rishabh Jha |
| src/models/transformer.py | Kunwar Abhuday Singh |
| src/data.py, src/train.py, src/make_figures.py | shared |
