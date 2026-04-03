from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def export_head_heatmap(score_matrix, out_path: Path) -> None:
    if hasattr(score_matrix, "detach"):  # torch tensor
        arr = score_matrix.detach().float().cpu().numpy()
    else:
        arr = np.asarray(score_matrix, dtype=np.float32)

    vmax = float(np.max(np.abs(arr))) if arr.size else 1.0
    if vmax == 0:
        vmax = 1.0

    plt.figure(figsize=(max(8, arr.shape[1] * 0.45), max(4, arr.shape[0] * 0.35)))
    im = plt.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Delta score (out_boundary - in_boundary)")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Head Importance Delta Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
