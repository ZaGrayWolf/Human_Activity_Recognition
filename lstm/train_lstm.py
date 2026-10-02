"""LSTM for Human Activity Recognition (UCI HAR) - owner: Rishabh Jha.

Trains a 2-layer LSTM on the preprocessed windows and saves ALL outputs
(log, metrics, confusion matrix, training curves, best weights) to results/.

Run locally :  python lstm/train_lstm.py --data data/preprocessed_har.npz
Run on Kaggle: !python train_lstm.py        (auto-finds preprocessed_har.npz; outputs go to /kaggle/working/results/lstm)
"""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
ON_KAGGLE = os.path.exists("/kaggle/input")

p = argparse.ArgumentParser()
p.add_argument("--data", default=None, help="path to preprocessed_har.npz")
p.add_argument("--out", default=None, help="output folder")
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

if a.data is None:
    pats = ["/kaggle/input/**/preprocessed_har.npz", "data/preprocessed_har.npz", "../data/preprocessed_har.npz"]
    a.data = next((h for pat in pats for h in glob.glob(pat, recursive=True)), None)
    assert a.data, "preprocessed_har.npz not found - pass --data"
if a.out is None:
    a.out = "/kaggle/working/results/lstm" if ON_KAGGLE else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(a.out, exist_ok=True)

log_f = open(os.path.join(a.out, "lstm_training_log.txt"), "w")
def log(s=""):
    print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

torch.set_num_threads(1)
torch.manual_seed(a.seed); np.random.seed(a.seed)
log(f"data: {a.data}\nseed={a.seed} epochs<={a.epochs} patience={a.patience} lr={a.lr} bs={a.bs} torch={torch.__version__}")

d = np.load(a.data)
Xtr, ytr, Xva, yva, Xte, yte = (d["seq_train"], d["y_train"], d["seq_val"], d["y_val"], d["seq_test"], d["y_test"])
log(f"shapes: train {Xtr.shape} val {Xva.shape} test {Xte.shape}")


class LSTMNet(nn.Module):
    def __init__(self, in_ch=9, hidden=64, layers=2, n_cls=6):
        super().__init__()
        self.lstm = nn.LSTM(in_ch, hidden, layers, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, n_cls))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])          # last time-step


net = LSTMNet()
n_params = sum(q.numel() for q in net.parameters() if q.requires_grad)
log(f"params: {n_params}")

mk = lambda X, y, sh: DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=a.bs, shuffle=sh)
tr, va, te = mk(Xtr, ytr, True), mk(Xva, yva, False), mk(Xte, yte, False)
opt, crit = torch.optim.Adam(net.parameters(), lr=a.lr), nn.CrossEntropyLoss()


def run_eval(dl):
    net.eval(); P, L, tot = [], [], 0.0
    with torch.no_grad():
        for x, y in dl:
            o = net(x); tot += crit(o, y).item() * len(y)
            P.append(o.argmax(1).numpy()); L.append(y.numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    return tot / len(L), P, L


hist, best, bad, best_state, t0 = [], -1, 0, None, time.time()
for ep in range(1, a.epochs + 1):
    net.train(); tl = 0.0
    for x, y in tr:
        opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
        tl += loss.item() * len(y)
    vl, vp, vy = run_eval(va); vacc = accuracy_score(vy, vp)
    hist.append(dict(epoch=ep, train_loss=tl / len(ytr), val_loss=vl, val_acc=vacc))
    log(f"[lstm] ep{ep:02d} train_loss={tl/len(ytr):.4f} val_loss={vl:.4f} val_acc={vacc:.4f}")
    if vacc > best:
        best, bad = vacc, 0
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience: log("early stop"); break
train_time = time.time() - t0
log(f"train time: {train_time:.1f}s")

net.load_state_dict(best_state)               # test only the best-validation checkpoint
_, tp, ty = run_eval(te)
pr, rc, f1, _ = precision_recall_fscore_support(ty, tp, average="macro", zero_division=0)
cm = confusion_matrix(ty, tp)
test = dict(accuracy=accuracy_score(ty, tp), macro_precision=pr, macro_recall=rc, macro_f1=f1)
log("TEST: " + json.dumps(test, indent=1)); log(str(cm))
log(classification_report(ty, tp, target_names=CLASSES, zero_division=0))

json.dump(dict(model="lstm", seed=a.seed, params=n_params, epochs_run=len(hist), best_val_acc=best,
               train_time_s=round(train_time, 1), test=test, confusion_matrix=cm.tolist(), history=hist),
          open(os.path.join(a.out, "lstm_metrics.json"), "w"), indent=1)
torch.save(best_state, os.path.join(a.out, "lstm_best.pt"))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6)); e = [h["epoch"] for h in hist]
ax[0].plot(e, [h["train_loss"] for h in hist], label="train"); ax[0].plot(e, [h["val_loss"] for h in hist], label="val")
ax[0].set_title("Loss"); ax[0].legend(); ax[1].plot(e, [h["val_acc"] for h in hist]); ax[1].set_title("Validation accuracy")
for q in ax: q.set_xlabel("Epoch"); q.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(a.out, "lstm_training_curves.png"), dpi=150); plt.close()

fig, q = plt.subplots(figsize=(5.2, 4.4)); cmn = cm / cm.sum(1, keepdims=True); q.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(6):
    for j in range(6): q.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .5 else "black")
sh = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]; q.set_xticks(range(6)); q.set_yticks(range(6)); q.set_xticklabels(sh); q.set_yticklabels(sh)
q.set_xlabel("Predicted"); q.set_ylabel("True"); q.set_title(f"LSTM confusion matrix (acc {test['accuracy']*100:.1f}%)")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "lstm_confusion_matrix.png"), dpi=150); plt.close()
log(f"\nSaved outputs to: {os.path.abspath(a.out)}"); log_f.close()
