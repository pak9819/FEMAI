"""Exportiert ein brick8-Energienetz (.mat) ins Binaerformat von brick8_bench.c.

    python export_c_weights.py <netz.mat> <netz.bin>

Format (little endian): int32 magic 'ELFD', int32 L, je Lage int32 rows,
int32 cols, W (row-major, float64), b (float64); dann c_mean (24),
c_std (24), D_scale (28) als float64.
"""
import sys

import numpy as np
import scipy.io

src, dst = sys.argv[1], sys.argv[2]
S = scipy.io.loadmat(src)
sf = str(np.squeeze(S["state_form"]))
assert sf.strip() == "gram_modal_F0_hourglass_triu", sf
L = int(np.squeeze(S["num_linear_layers"]))
with open(dst, "wb") as f:
    np.array([0x44464C45, L], dtype="<i4").tofile(f)
    for l in range(1, L + 1):
        W = np.asarray(S[f"W{l}"], dtype="<f8")
        b = np.asarray(S[f"b{l}"], dtype="<f8").reshape(-1)
        np.array(W.shape, dtype="<i4").tofile(f)
        np.ascontiguousarray(W).tofile(f)
        b.tofile(f)
    for k in ("input_norm_c_mean", "input_norm_c_std", "input_norm_D_scale"):
        np.asarray(S[k], dtype="<f8").reshape(-1).tofile(f)
print(f"{dst}: {L} Lagen, Breite {S['W1'].shape[0]}")
