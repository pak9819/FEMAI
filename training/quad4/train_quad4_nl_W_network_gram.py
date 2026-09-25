"""quad4-Voll-Energie-Netz mit METRIK-Eingang (Gram-Kette) -- Phase 0 brick8.

Validiert die dimensionsunabhaengige Gram-Kette (training/common/dlfe_gram.py)
am quad4, bevor sie auf das brick8 uebertragen wird. Das bestehende Ko-
Rotationsnetz (train_quad4_nl_W_network.py) bleibt unangetastet.

    Netz-Eingang: [c_hat (8), D (10)]      D = triu(x u' + u x' + u u')
    W_hat = f(c~, D~) - f(c~, 0) - grad f(c~, 0)' D~

Daten: identisch zum Ko-Rotationsnetz (derselbe Datensatz-Cache). Die dort
gespeicherten Zustaende z sind gueltige Verschiebungen u der kanonischen
Geometrie -- die Gram-Form ist rotationsinvariant, die Ko-Rotation entfaellt.

Aufruf:
    python train_quad4_nl_W_network_gram.py
    QUAD4_HIDDEN=48 python train_quad4_nl_W_network_gram.py
    QUAD4_QUICK=1  python train_quad4_nl_W_network_gram.py

Deployt nach sourcecode/elements/quad4/quad4_nl_W_network_gram.mat (nur bei
gruenen Gates a/b/g/c). MATLAB waehlt es mit QUAD4_STATE_FORM=gram.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch

save_dir = Path(__file__).parent
sys.path.insert(0, str(save_dir / ".." / "common"))

import dlfe_gram as dg          # noqa: E402
import quad4_nl_ref as ref      # noqa: E402

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

element_dir = save_dir / ".." / ".." / "sourcecode" / "elements" / "quad4"
sweep_dir = save_dir / "arch_sweep_nl_W_gram"
sweep_dir.mkdir(exist_ok=True)
_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
HIDDEN = int(os.environ.get("QUAD4_HIDDEN", 32))
DEPTH = int(os.environ.get("QUAD4_DEPTH", 3))
QUICK = bool(os.environ.get("QUAD4_QUICK"))
# Zustandsform: modal (Standard, F0 + Hourglass) oder Knoten-Gram (QUAD4_GRAM_MODAL=0)
MODAL = os.environ.get("QUAD4_GRAM_MODAL", "1") != "0"
OUT_NAME = os.environ.get("QUAD4_GRAM_OUT", "quad4_nl_W_network_gram.mat")
_form = "modal" if MODAL else "node"
log_path = sweep_dir / f"quad4_nl_W_gram_{_form}_h{HIDDEN}d{DEPTH}_{_stamp}.log"

logger = logging.getLogger("quad4_nl_W_gram")
logger.setLevel(logging.INFO)
logger.handlers.clear()
for h in (logging.StreamHandler(), logging.FileHandler(log_path, "w", "utf-8")):
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)


def log(msg=""):
    logger.info(msg)


SPEC = dg.GramSpec(n=4, dim=2, modal=MODAL)
SPEC_NODE = dg.GramSpec(n=4, dim=2)          # fuer die Energie-als-Funktion-von-D-Gates
CFG = dg.TrainCfg()
if QUICK:
    CFG.epochs = 60


# ---------------------------------------------------------------------------
# Gate g: Gram-Kette ohne Netz
# ---------------------------------------------------------------------------

def gate_gram(n_samples=40, verbose=True):
    """g1  analytische StVenant-Energie als Funktion von D == Referenz-W
       g2  autograd F, K ueber D(u) == Referenz-F, K
       g3  D invariant unter Starrkoerperdrehung + Translation des Zustands
       g4  reine Starrkoerperbewegung -> D = 0
       g5  (modal) linearer Block von D == F0'F0 - I = 2 E(Mitte)"""
    rng = np.random.RandomState(7)
    lam = ref.E_TRAIN * ref.NU_TRAIN / ((1 + ref.NU_TRAIN) * (1 - 2 * ref.NU_TRAIN))
    mu = ref.E_TRAIN / (2 * (1 + ref.NU_TRAIN))
    w1 = w2 = w3 = w4 = w5 = 0.0
    k = 0
    while k < n_samples:
        coords = ref.generate_distorted_quad(rng)
        if coords is None:
            continue
        cc = ref.canonicalize_coords(coords)
        u = ref.sample_state(cc, rng)
        if np.linalg.norm(u) < 1e-6:
            continue
        k += 1
        Wr, Fr, Kr = ref.energy_stiffness_force_ref(cc, u)
        grads, dvs = [], []
        for (r, s) in ref.GAUSS_POINTS:
            hx, hy, detJ = ref.shape_quad4_ref(cc, r, s)
            grads.append(torch.tensor(np.stack([hx, hy], 1), dtype=torch.float64))
            dvs.append(detJ)
        c_t = torch.tensor(cc.reshape(-1), dtype=torch.float64)
        u_t = torch.tensor(u.reshape(-1), dtype=torch.float64)

        def Wfun(uu):
            Dfull = dg.D_to_full(SPEC_NODE, dg.gram_D(SPEC_NODE, c_t, uu))
            return dg.stvenant_energy_from_D(Dfull, grads, dvs, lam, mu)

        W = float(Wfun(u_t))
        F = torch.func.grad(Wfun)(u_t).numpy()
        K = torch.func.jacfwd(torch.func.grad(Wfun))(u_t).numpy()
        w1 = max(w1, abs(W - Wr) / max(abs(Wr), 1e-14))
        w2 = max(w2, np.linalg.norm(F - Fr) / max(np.linalg.norm(Fr), 1e-14),
                 np.linalg.norm(K - Kr) / np.linalg.norm(Kr))
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        u_rot = (cc + u) @ R.T + rng.uniform(-3, 3, 2) - cc
        D0 = dg.gram_D(SPEC, c_t, u_t).numpy()
        D1 = dg.gram_D(SPEC, c_t, torch.tensor(u_rot.reshape(-1))).numpy()
        w3 = max(w3, np.linalg.norm(D1 - D0) / np.linalg.norm(D0))
        u_rig = cc @ R.T + rng.uniform(-3, 3, 2) - cc
        w4 = max(w4, float(dg.gram_D(SPEC, c_t, torch.tensor(u_rig.reshape(-1))).abs().max()))
        if SPEC.modal:
            hx, hy, _ = ref.shape_quad4_ref(cc, 0.0, 0.0)
            F0 = np.eye(2) + np.array([[u[:, 0] @ hx, u[:, 0] @ hy],
                                       [u[:, 1] @ hx, u[:, 1] @ hy]])
            C0 = F0.T @ F0 - np.eye(2)
            w5 = max(w5, np.abs(D0[[0, 1, 2]] - C0[[0, 0, 1], [0, 1, 1]]).max()
                     / np.abs(C0).max())
    ok = w1 < 1e-12 and w2 < 1e-10 and w3 < 1e-10 and w4 < 1e-13 and w5 < 1e-12
    if verbose:
        log(f"  Gate g1 W(D) vs Referenz     : {w1:.2e}  (< 1e-12)")
        log(f"  Gate g2 F,K ueber D vs Ref.  : {w2:.2e}  (< 1e-10)")
        log(f"  Gate g3 D rotationsinvariant : {w3:.2e}  (< 1e-10, Translation +-3 rundet)")
        log(f"  Gate g4 D(Starrkoerper) = 0  : {w4:.2e}  (< 1e-13)")
        if SPEC.modal:
            log(f"  Gate g5 D_lin == 2 E(Mitte)  : {w5:.2e}  (< 1e-12)")
        log(f"  Gate g -> {'GRUEN' if ok else 'ROT'}")
    return ok


# ---------------------------------------------------------------------------
# Daten (Cache des Ko-Rotationsnetzes)
# ---------------------------------------------------------------------------

def load_or_build_dataset():
    traj = save_dir / "newton_traj_states.mat"
    traj_stamp = (f"{traj.stat().st_size}_{int(traj.stat().st_mtime)}"
                  if traj.exists() else "none")
    # identisch zu train_quad4_nl_W_network.dataset_cache_key()
    n_geom_tr, n_geom_va, spe = (1200, 300, 6) if QUICK else (14000, 2000, 12)
    parts = ["v1", QUICK, n_geom_tr, n_geom_va, spe, 0.70, 0.20, 0.10, traj_stamp,
             ref.E_MAX, ref.ENV_RATIO_MAX, ref.ENV_ANGLE_MIN, ref.ENV_ANGLE_MAX,
             ref.ENV_TAPER_MAX, ref.AMP_LOG_MIN, ref.P_ZERO,
             ref.NU_TRAIN, ref.CONDITION, ref.MATERIAL, 1234]
    key = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]
    path = save_dir / f"dataset_cache_{key}.npz"
    if not path.exists():
        log(f"  Cache {path.name} fehlt -- Aufbau ueber train_quad4_nl_W_network.")
        import train_quad4_nl_W_network as base   # noqa: E402
        tr, va, nsv = base.build_full_dataset()
        base.save_dataset_cache(key, tr, va, nsv)
    d = np.load(path)
    tr = {k[3:]: d[k] for k in d.files if k.startswith("tr_")}
    va = {k[3:]: d[k] for k in d.files if k.startswith("va_")}
    for a in (tr, va):
        a["u"] = a.pop("z")
    log(f"  Datensatz: {path.name} ({len(tr['chat'])} train / {len(va['chat'])} val)")
    return tr, va, int(d["n_syn_val"])


# ---------------------------------------------------------------------------

def main():
    log("=== quad4 NL Voll-Energie-Netz, Gram-Kette (Phase 0 brick8) ===")
    log(f"Log: {log_path}\nDevice: {device} | {SPEC.state_form} | h{HIDDEN} d{DEPTH} | "
        f"{'QUICK' if QUICK else 'voll'}\n")

    log("Gates a/b (Referenz, Ko-Rotationskette):")
    ok_ab, _ = ref.run_gates_a_b(verbose=True, strict=True)
    log("\nGate g (Gram-Kette, ohne Netz):")
    if not (ok_ab and gate_gram()):
        raise SystemExit("Gates a/b/g rot -- Abbruch.")

    tr_arr, va_arr, n_syn_val = load_or_build_dataset()
    norm = dg.normalization(SPEC, tr_arr, device)
    log(f"  D_scale min/max {float(norm[2].min()):.3e} / {float(norm[2].max()):.3e}")
    tr = dg.Batch(SPEC, tr_arr, device)
    va = dg.Batch(SPEC, va_arr, device)

    model, _ = dg.train_one(SPEC, HIDDEN, DEPTH, tr, va, norm, CFG, log, device,
                            tag="[voll] ")
    eF, eK = dg.evaluate(model, va)
    res = {}
    is_traj = np.zeros(len(va), bool)
    is_traj[n_syn_val:] = True
    log("\n=== Ergebnis (relative TOTAL-Fehler [%]) ===")
    dg.report_bins(log, eF, eK, va.amp, va.dist, "Validierung gesamt", res)
    if is_traj.any():
        dg.report_bins(log, eF[~is_traj], eK[~is_traj], va.amp[~is_traj],
                       va.dist[~is_traj], "nur synthetisch", res)
        dg.report_bins(log, eF[is_traj], eK[is_traj], va.amp[is_traj],
                       va.dist[is_traj], "nur Trajektorien", res)
    ok_c, fd, sym = dg.gate_c(model, va, log)
    go = dg.go_ok(eF, eK, CFG)
    sF, sK = dg.stats(eF), dg.stats(eK)
    log(f"\nGo-Kriterium: eF {sF['mean']:.3f}/{sF['p99']:.3f} | "
        f"eK {sK['mean']:.3f}/{sK['p99']:.3f} -> {'ERFUELLT' if go else 'VERFEHLT'}")
    res.update(gate_c=dict(fd=fd, sym=sym, ok=bool(ok_c)), go=bool(go))
    if not ok_c:
        raise SystemExit("Gate c ROT -> kein Export.")

    meta = {
        "canonicalization": "edge_n1n2_to_pos_x_after_centroid_Lc",
        "scaling": "W=E*d*Lc^2*What; Finte=E*d*Lc*dWhat/du_hat; Ke=E*d*d2What/du_hat2",
        "material": ref.MATERIAL,
        "nu_train": ref.NU_TRAIN,
        "condition": ref.CONDITION,
        "state_E_max": ref.E_MAX,
        "env_ratio_max": ref.ENV_RATIO_MAX,
        "train_script": Path(__file__).name,
        "git_hash": dg.git_hash(save_dir),
        "seed": 42,
    }
    md = dg.build_mat(model, HIDDEN, DEPTH, va, meta, res)
    out = (element_dir / OUT_NAME).resolve()
    scipy.io.savemat(str(out), md)
    torch.save(model.state_dict(), sweep_dir / f"quad4_nl_W_gram_{_form}_h{HIDDEN}d{DEPTH}_{_stamp}.pt")
    with open(sweep_dir / f"quad4_nl_W_gram_{_form}_h{HIDDEN}d{DEPTH}_{_stamp}_metrics.json", "w") as fh:
        json.dump(res, fh, indent=2)
    log(f"\nDeployt: {out}")


if __name__ == "__main__":
    main()
