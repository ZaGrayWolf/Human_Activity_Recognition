import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
M = [("mlp", "MLP"), ("cnn", "1D-CNN"), ("lstm", "LSTM"), ("transformer", "Transformer")]
SHORT = ["Walk", "Up", "Down", "Sit", "Stand", "Lay"]
res = {m: json.load(open(f"{R}/{m}.json")) for m, _ in M}

fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
for a, (m, n) in zip(ax, M):
    cm = np.array(res[m]["confusion_matrix"]); cmn = cm / cm.sum(1, keepdims=True)
    a.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            a.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                   color="white" if cmn[i, j] > .5 else "black")
    a.set_xticks(range(6)); a.set_yticks(range(6))
    a.set_xticklabels(SHORT, rotation=45, fontsize=8); a.set_yticklabels(SHORT, fontsize=8)
    a.set_title(f"{n} (acc {res[m]['test']['accuracy']*100:.1f}%)", fontsize=10)
    a.set_xlabel("Predicted", fontsize=8)
ax[0].set_ylabel("True", fontsize=8)
plt.tight_layout(); plt.savefig(f"{R}/confusion_matrices.png", dpi=170); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
for m, n in M:
    h = res[m]["history"]; e = [x["epoch"] for x in h]
    ax[0].plot(e, [x["train_loss"] for x in h], label=n)
    ax[1].plot(e, [x["val_acc"] for x in h], label=n)
ax[0].set_title("Training loss"); ax[1].set_title("Validation accuracy")
for a in ax: a.set_xlabel("Epoch"); a.grid(alpha=.3)
ax[1].legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{R}/training_curves.png", dpi=170); plt.close()

# multi-seed aggregate
agg = {}
for m, _ in M:
    runs = [json.load(open(f)) for f in [f"{R}/{m}.json"] + sorted(glob.glob(f"{R}/{m}_s*.json"))]
    agg[m] = {k: (float(np.mean([r["test"][k] for r in runs])), float(np.std([r["test"][k] for r in runs])))
              for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]}
    agg[m]["n_seeds"] = len(runs)
json.dump(agg, open(f"{R}/aggregate.json", "w"), indent=1)
print(json.dumps(agg, indent=1))
