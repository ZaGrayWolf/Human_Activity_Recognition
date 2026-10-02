"""Train + evaluate one model under the common protocol.
usage: python src/train.py --model {mlp,cnn,lstm,transformer} [--epochs N]
"""
import argparse, json, time, os
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from data import load_har, CLASSES
from models import MLP, CNN1D, LSTMNet, TransformerNet

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, choices=["mlp", "cnn", "lstm", "transformer"])
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=64)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
seq, feat, lab, info = load_har(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_raw"), seed=a.seed)
data = feat if a.model == "mlp" else seq
net = dict(mlp=MLP, cnn=CNN1D, lstm=LSTMNet, transformer=TransformerNet)[a.model]()
n_params = sum(q.numel() for q in net.parameters() if q.requires_grad)

def loader(split, shuffle):
    ds = TensorDataset(torch.tensor(data[split]), torch.tensor(lab[split]))
    return DataLoader(ds, batch_size=a.bs, shuffle=shuffle)

tr, va, te = loader("train", True), loader("val", False), loader("test", False)
opt = torch.optim.Adam(net.parameters(), lr=a.lr)
crit = nn.CrossEntropyLoss()

def run_eval(dl):
    net.eval(); P, L, tot = [], [], 0.0
    with torch.no_grad():
        for x, y in dl:
            o = net(x); tot += crit(o, y).item() * len(y)
            P.append(o.argmax(1).numpy()); L.append(y.numpy())
    P, L = np.concatenate(P), np.concatenate(L)
    return tot / len(L), P, L

hist, best, bad, best_state = [], -1, 0, None
t0 = time.time()
for ep in range(1, a.epochs + 1):
    net.train(); tl = 0.0
    for x, y in tr:
        opt.zero_grad(); loss = crit(net(x), y); loss.backward(); opt.step()
        tl += loss.item() * len(y)
    vl, vp, vy = run_eval(va); vacc = accuracy_score(vy, vp)
    hist.append(dict(epoch=ep, train_loss=tl / len(lab["train"]), val_loss=vl, val_acc=vacc))
    print(f"[{a.model}] ep{ep:02d} train_loss={hist[-1]['train_loss']:.4f} val_loss={vl:.4f} val_acc={vacc:.4f}", flush=True)
    if vacc > best:
        best, bad = vacc, 0
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= a.patience: print("early stop"); break
train_time = time.time() - t0

net.load_state_dict(best_state)           # test only the best-on-validation checkpoint
_, tp, ty = run_eval(te)
pr, rc, f1, _ = precision_recall_fscore_support(ty, tp, average="macro", zero_division=0)
res = dict(model=a.model, params=n_params, epochs_run=len(hist), best_val_acc=best,
           train_time_s=round(train_time, 1),
           test=dict(accuracy=accuracy_score(ty, tp), macro_precision=pr, macro_recall=rc, macro_f1=f1),
           confusion_matrix=confusion_matrix(ty, tp).tolist(),
           per_class=classification_report(ty, tp, target_names=CLASSES, output_dict=True, zero_division=0),
           history=hist, split_info=info, seed=a.seed)
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)
json.dump(res, open(f"{RES}/{a.model}{"" if a.seed == 42 else "_s" + str(a.seed)}.json", "w"), indent=1)
print(json.dumps(res["test"], indent=1)); print("params", n_params, "time", res["train_time_s"], "s")
