"""Shared data loading + preprocessing for all four HAR models (UCI HAR).

Same windows, labels, official subject-wise test split and a subject-grouped
validation split are used for every model. Only the MLP gets a different
*representation* (36 statistical features) by design.
"""
import os
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]
CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
           "SITTING", "STANDING", "LAYING"]


def _load_split(root, split):
    sig_dir = os.path.join(root, split, "Inertial Signals")
    chans = [np.loadtxt(os.path.join(sig_dir, f"{s}_{split}.txt")) for s in SIGNALS]
    X = np.stack(chans, axis=-1).astype(np.float32)            # (N, 128, 9)
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt"), dtype=int) - 1  # 1..6 -> 0..5
    subj = np.loadtxt(os.path.join(root, split, f"subject_{split}.txt"), dtype=int)
    return X, y, subj


def stat_features(X):
    """(N,128,9) -> (N,36): mean, std, min, max per channel."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)


def load_har(root="data_raw", val_size=0.2, seed=42):
    Xtr_all, ytr_all, str_all = _load_split(root, "train")
    Xte, yte, ste = _load_split(root, "test")

    # group-based validation split: no subject appears in both train and val
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(gss.split(Xtr_all, ytr_all, groups=str_all))
    assert not set(str_all[tr_idx]) & set(str_all[va_idx]), "subject leakage!"
    Xtr, ytr, Xva, yva = Xtr_all[tr_idx], ytr_all[tr_idx], Xtr_all[va_idx], ytr_all[va_idx]

    # --- sequence representation: per-channel z-score, statistics from TRAIN only
    mu = Xtr.mean(axis=(0, 1), keepdims=True)
    sd = Xtr.std(axis=(0, 1), keepdims=True) + 1e-8
    seq = {k: (v - mu) / sd for k, v in dict(train=Xtr, val=Xva, test=Xte).items()}

    # --- MLP representation: 36 statistical features, scaler fit on TRAIN only
    sc = StandardScaler().fit(stat_features(Xtr))
    feat = {k: sc.transform(stat_features(v)).astype(np.float32)
            for k, v in dict(train=Xtr, val=Xva, test=Xte).items()}

    labels = dict(train=ytr, val=yva, test=yte)
    info = dict(train_subjects=sorted(set(str_all[tr_idx].tolist())),
                val_subjects=sorted(set(str_all[va_idx].tolist())),
                test_subjects=sorted(set(ste.tolist())),
                n=dict(train=len(ytr), val=len(yva), test=len(yte)),
                class_counts={k: np.bincount(v, minlength=6).tolist() for k, v in labels.items()})
    return seq, feat, labels, info
