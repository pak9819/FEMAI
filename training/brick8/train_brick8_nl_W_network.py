"""Voll-Energie-Netz fuer das nichtlineare brick8-Element (StVenant, 3D).

Uebertragung des quad4-Energienetzes auf das 8-Knoten-Hexaeder. Statt der
2D-Ko-Rotation (in 3D ohne geschlossene Form) nutzt das Netz den MODALEN
Metrik-Eingang (training/common/dlfe_gram.py, in Phase 0 am quad4
validiert):

    Z = L' Y,   L = [dh0 | gamma]  (kanonische Geometrie, 8 x 7)
    D = triu(Z Z' - Z0 Z0') = triu([F0'F0 - I, F0'q ; q'F0, q'q])   (28)
    W_hat = f(c~, D~) - f(c~, 0) - grad f(c~, 0)' D~
    Eingang [c_hat (24), D (28)] = 52, Ausgang skalar, GELU (erf)

Exakt (nicht gelernt): Objektivitaet, W = F = 0 bei Starrkoerperbewegung,
Ke = dFinte/dUe, Symmetrie, Kraeftegleichgewicht, Skalierung
W = E*Lc^3*W_hat, Finte = E*Lc^2*dW_hat/du_hat, Ke = E*Lc*d2W_hat/du_hat2.
nu = 0.3 und StVenant sind eintrainiert (MATLAB prueft hart).

Daten: 70 % synthetisch / 20 % Newton-Trajektorien
(generate_newton_trajectories_brick8.m) / 10 % Slot (synthetisch).

Aufruf:
    python train_brick8_nl_W_network.py
    BRICK8_HIDDEN=128 BRICK8_DEPTH=3 python train_brick8_nl_W_network.py
    BRICK8_QUICK=1 python train_brick8_nl_W_network.py        # Pipeline-Test

Deployt nach sourcecode/elements/brick8/brick8_nl_W_network.mat, wenn die
Gates a/b (a' falls vorhanden) und c gruen sind (BRICK8_OUT aendert den
Dateinamen, z.B. fuer Architekturvergleiche).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch

save_dir = Path(__file__).parent
sys.path.insert(0, str(save_dir / ".." / "common"))

import dlfe_gram as dg          # noqa: E402
import brick8_nl_ref as ref     # noqa: E402

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

element_dir = save_dir / ".." / ".." / "sourcecode" / "elements" / "brick8"
sweep_dir = save_dir / "arch_sweep_nl_W"
sweep_dir.mkdir(exist_ok=True)
_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

HIDDEN = int(os.environ.get("BRICK8_HIDDEN", 96))
DEPTH = int(os.environ.get("BRICK8_DEPTH", 3))
QUICK = bool(os.environ.get("BRICK8_QUICK"))
OUT_NAME = os.environ.get("BRICK8_OUT", "brick8_nl_W_network.mat")
EPOCHS = int(os.environ.get("BRICK8_EPOCHS", 400))

N_GEOM_TRAIN, N_GEOM_VAL, STATES_PER_ELEM = 14000, 2000, 12
FRAC_SYNTH, FRAC_TRAJ, FRAC_AL = 0.70, 0.20, 0.10
if QUICK:
    N_GEOM_TRAIN, N_GEOM_VAL, STATES_PER_ELEM = 800, 200, 6
    EPOCHS = 40

log_path = sweep_dir / f"brick8_nl_W_h{HIDDEN}d{DEPTH}_{_stamp}.log"
logger = logging.getLogger("brick8_nl_W")
logger.setLevel(logging.INFO)
logger.handlers.clear()
for h in (logging.StreamHandler(), logging.FileHandler(log_path, "w", "utf-8")):
    h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(h)


def log(msg=""):
    logger.info(msg)


SPEC = dg.GramSpec(8, 3, modal=True)
CFG = dg.TrainCfg(epochs=EPOCHS)
TRAJ_FILE = save_dir / "newton_traj_states_brick8.mat"


# ---------------------------------------------------------------------------
# Datensatz
# ---------------------------------------------------------------------------

def compute_targets(chat, u, chunk=4096):
    """fp64-Targets batched: W, F (24), Ktriu (300)."""
    W, F, K = [], [], []
    ki, kj = SPEC.kti, SPEC.ktj
    for s in range(0, len(chat), chunk):
        c = torch.tensor(chat[s:s + chunk], dtype=torch.float64)
        v = torch.tensor(u[s:s + chunk], dtype=torch.float64)
        w, f, k = ref.targets(c, v)
        W.append(w.numpy())
        F.append(f.numpy())
        K.append(k[:, ki, kj].numpy())
    return np.concatenate(W), np.concatenate(F), np.concatenate(K)


def pack(chat, u, amp, dist):
    chat = np.asarray(chat)
    u = np.asarray(u)
    W, F, K = compute_targets(chat, u)
    return dict(chat=chat.astype(np.float32), u=u.astype(np.float32),
                W=W.astype(np.float32), F=F.astype(np.float32),
                Ktriu=K.astype(np.float32), amp=np.asarray(amp, np.float32),
                dist=np.asarray(dist, np.float32))


def build_synthetic(n_geom, spe, rng, label):
    t0 = time.time()
    C, U, A, Dd = [], [], [], []
    n = 0
    while n < n_geom:
        X = ref.generate_distorted_hex(rng)
        if X is None:
            continue
        Xc, _, _ = ref.canonical_frame(X)
        dr = ref.detj_ratio(Xc)
        for _ in range(spe):
            u = ref.sample_state(Xc, rng)
            C.append(Xc.reshape(-1))
            U.append(u.reshape(-1))
            A.append(ref.hull_E(Xc, u))
            Dd.append(dr)
        n += 1
    out = pack(C, U, A, Dd)
    log(f"  synthetisch ({label}): {len(C)} Samples in {time.time()-t0:.1f} s")
    return out


def build_trajectories():
    if not TRAJ_FILE.exists():
        log("  WARNUNG: newton_traj_states_brick8.mat fehlt -- Trajektorien-Anteil")
        log("           wird synthetisch aufgefuellt.")
        return None, None
    md = scipy.io.loadmat(str(TRAJ_FILE))
    coords, Ue, is_val = md["coords"], md["Ue"], md["is_val"].reshape(-1)
    N = Ue.shape[1]
    tr, va = ([], [], [], []), ([], [], [], [])
    skipped = 0
    for i in range(N):
        X = coords[:, :, i]
        Xc, Lc, Rc = ref.canonical_frame(X)
        if not ref.in_geometry_hull(Xc):
            skipped += 1
            continue
        u = (Ue[:, i].reshape(8, 3) @ Rc.T) / Lc
        u -= u.mean(axis=0)
        mE = ref.hull_E(Xc, u)
        if mE > ref.E_MAX:
            skipped += 1
            continue
        tgt = va if is_val[i] > 0.5 else tr
        tgt[0].append(Xc.reshape(-1)); tgt[1].append(u.reshape(-1))
        tgt[2].append(mE); tgt[3].append(ref.detj_ratio(Xc))
    log(f"  Trajektorien: {N} roh -> {len(tr[0])} train / {len(va[0])} val "
        f"(verworfen {skipped})")
    return tr, va


def concat(*arrs):
    return {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}


def subsample_lists(lst, n, rng):
    if len(lst[0]) <= n:
        return lst
    idx = rng.choice(len(lst[0]), n, replace=False)
    return tuple([x[i] for i in idx] for x in lst)


def build_full_dataset():
    rng = np.random.RandomState(1234)
    traj_tr, traj_va = build_trajectories()
    n_syn = N_GEOM_TRAIN * STATES_PER_ELEM
    total = int(round(n_syn / FRAC_SYNTH))
    n_traj = int(round(total * FRAC_TRAJ))
    n_al = int(round(total * FRAC_AL))

    parts = [build_synthetic(N_GEOM_TRAIN, STATES_PER_ELEM, rng, "train")]
    n_have = 0
    if traj_tr is not None and len(traj_tr[0]) > 0:
        tt = subsample_lists(traj_tr, n_traj, rng)
        parts.append(pack(*tt))
        n_have = len(tt[0])
        log(f"  Trajektorien im Training: {n_have}")
    if n_traj - n_have > 0:
        parts.append(build_synthetic(max(1, (n_traj - n_have) // STATES_PER_ELEM),
                                     STATES_PER_ELEM, rng, "Traj-Ersatz"))
    parts.append(build_synthetic(max(1, n_al // STATES_PER_ELEM), STATES_PER_ELEM,
                                 rng, "Slot"))
    tr = concat(*parts)

    va_parts = [build_synthetic(N_GEOM_VAL, STATES_PER_ELEM, rng, "val")]
    n_syn_val = len(va_parts[0]["W"])
    if traj_va is not None and len(traj_va[0]) > 0:
        va_parts.append(pack(*subsample_lists(traj_va, 5000, rng)))
    return tr, concat(*va_parts), n_syn_val


def dataset_cache():
    stamp = (f"{TRAJ_FILE.stat().st_size}_{int(TRAJ_FILE.stat().st_mtime)}"
             if TRAJ_FILE.exists() else "none")
    parts = ["b8v2", QUICK, ref.P_HOURGLASS, N_GEOM_TRAIN, N_GEOM_VAL, STATES_PER_ELEM, FRAC_SYNTH,
             FRAC_TRAJ, FRAC_AL, stamp, ref.E_MAX, ref.ENV_RATIO_MAX,
             ref.ENV_ANGLE_MIN, ref.ENV_ANGLE_MAX, ref.ENV_TAPER_MAX,
             ref.AMP_LOG_MIN, ref.P_ZERO, ref.NU_TRAIN, ref.MATERIAL, 1234]
    key = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]
    path = save_dir / f"dataset_cache_brick8_{key}.npz"
    if path.exists():
        d = np.load(path)
        tr = {k[3:]: d[k] for k in d.files if k.startswith("tr_")}
        va = {k[3:]: d[k] for k in d.files if k.startswith("va_")}
        log(f"  Datensatz aus Cache: {path.name}")
        return tr, va, int(d["n_syn_val"])
    log("\nDatensatz aufbauen:")
    tr, va, nsv = build_full_dataset()
    payload = {f"tr_{k}": v for k, v in tr.items()}
    payload.update({f"va_{k}": v for k, v in va.items()})
    payload["n_syn_val"] = np.array(nsv)
    np.savez(path, **payload)
    log(f"  gecacht: {path.name} ({path.stat().st_size/1e6:.0f} MB)")
    return tr, va, nsv


# ---------------------------------------------------------------------------

def main():
    log("=== brick8 NL Voll-Energie-Netz (modale Metrik-Kette, StVenant 3D) ===")
    log(f"Log: {log_path}\nDevice: {device} | h{HIDDEN} d{DEPTH} | Eingang {SPEC.n_in} | "
        f"{'QUICK' if QUICK else 'voll'}\n")

    ok_gates = ref.run_gates(verbose=True, require_oracle=False)
    if not ok_gates:
        raise SystemExit("Gates a/b/a' rot -- Abbruch.")

    tr_arr, va_arr, n_syn_val = dataset_cache()
    log(f"  Training {len(tr_arr['W'])} | Validierung {len(va_arr['W'])} "
        f"({n_syn_val} synth + {len(va_arr['W']) - n_syn_val} traj)")
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
    log(f"\nGo-Kriterium (mean < {CFG.go_mean} %, p99 < {CFG.go_p99} %): "
        f"eF {sF['mean']:.3f}/{sF['p99']:.3f} | eK {sK['mean']:.3f}/{sK['p99']:.3f} "
        f"-> {'ERFUELLT' if go else 'VERFEHLT'}")
    res.update(gate_c=dict(fd=fd, sym=sym, ok=bool(ok_c)), go=bool(go),
               arch=dict(hidden=HIDDEN, depth=DEPTH,
                         params=sum(p.numel() for p in model.parameters()),
                         macs=dg.arch_macs(SPEC, HIDDEN, DEPTH)))
    if not ok_c:
        raise SystemExit("Gate c ROT -> kein Export.")

    meta = {
        "canonicalization": "edge_n1n2_x_node4_xy_after_centroid_Lc",
        "scaling": "W=E*Lc^3*What; Finte=E*Lc^2*dWhat/du_hat; Ke=E*Lc*d2What/du_hat2",
        "material": ref.MATERIAL,
        "nu_train": ref.NU_TRAIN,
        "condition": ref.CONDITION,
        "state_E_max": ref.E_MAX,
        "env_ratio_max": ref.ENV_RATIO_MAX,
        "env_angle_min": ref.ENV_ANGLE_MIN,
        "env_angle_max": ref.ENV_ANGLE_MAX,
        "train_script": Path(__file__).name,
        "git_hash": dg.git_hash(save_dir),
        "seed": 42,
    }
    md = dg.build_mat(model, HIDDEN, DEPTH, va, meta, res)
    element_dir.mkdir(parents=True, exist_ok=True)
    out = (element_dir / OUT_NAME).resolve()
    scipy.io.savemat(str(out), md)
    scipy.io.savemat(str(sweep_dir / f"brick8_nl_W_h{HIDDEN}d{DEPTH}_{_stamp}.mat"), md)
    torch.save(model.state_dict(), sweep_dir / f"brick8_nl_W_h{HIDDEN}d{DEPTH}_{_stamp}.pt")
    with open(sweep_dir / f"brick8_nl_W_h{HIDDEN}d{DEPTH}_{_stamp}_metrics.json", "w") as fh:
        json.dump(res, fh, indent=2)
    log(f"\nDeployt: {out}")


if __name__ == "__main__":
    main()
