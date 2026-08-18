"""Residual-Energie-Netz fuer das nichtlineare quad4-Element (Sobolev-Training).

Modell (K0-Split, kein Cache):

    W(c,z) = 0.5*z' K0(c) z + W_NL(c,z)          K0 = exakte lineare Steifigkeit
    F      = K0 z + grad_z W_NL
    K      = K0   + hess_z W_NL

Das Netz lernt NUR die nichtlineare Energieabweichung W_NL. Fuer StVenant ist
W exakt ein Polynom 4. Grades in z; der K0-Split entfernt den quadratischen
Term, das Netz sieht also nur den kubisch/quartischen Rest. Folgen:

  * Konsistenz Ke = dFinte/dz gilt PER KONSTRUKTION (beides aus einem
    Skalarpotential) -> Newton konvergiert wieder quadratisch.
  * Das Kleinamplituden-Regime (Newton-Endphase, historischer Schmerzpunkt)
    wird von K0 exakt dominiert.
  * Leichtere Zielfunktion -> kleineres Netz -> schnelleres Element.

Strukturelle Nullform (Subtraktionsform, nur W und F):

    W_NL = f(c~,z~) - f(c~,0) - grad_z~ f(c~,0)' z~

liefert exakt W_NL(c,0) = 0 und F_NL(c,0) = 0 (reine Starrkoerperbewegung ->
Finte = 0 in Maschinengenauigkeit). K_NL(c,0) ~ 0 wird weich ueber die
Targets erzwungen (Residual-K-Target bei z=0 ist exakt 0) und in Gate e4
(MATLAB) hart geprueft.

Trainiert wird auf RESIDUEN, normiert und bewertet wird auf TOTALGROESSEN --
denn das ist, was der Newton-Loeser sieht.

Aufruf:
    python train_quad4_nl_W_network.py
    QUAD4_QUICK=1 python train_quad4_nl_W_network.py     # Pipeline-Test
    QUAD4_SWEEP=1 python train_quad4_nl_W_network.py     # Architektur-Sweep

Deployt nach sourcecode/elements/quad4/quad4_nl_W_network.mat -- aber NUR,
wenn alle Gates gruen sind (kein stiller Fallback, Lektion 2026-07-15).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.func import grad, jacfwd, vmap

import quad4_nl_ref as ref

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Pfade und Logging
# ---------------------------------------------------------------------------

save_dir = Path(__file__).parent
element_dir = save_dir / ".." / ".." / "sourcecode" / "elements" / "quad4"
sweep_dir = save_dir / "arch_sweep_nl_W"
sweep_dir.mkdir(exist_ok=True)

_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = sweep_dir / f"quad4_nl_W_{_stamp}.log"

logger = logging.getLogger("quad4_nl_W")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_hc = logging.StreamHandler()
_hc.setFormatter(logging.Formatter("%(message)s"))
_hf = logging.FileHandler(log_path, mode="w", encoding="utf-8")
_hf.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
logger.addHandler(_hc)
logger.addHandler(_hf)


def log(msg=""):
    logger.info(msg)


# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------

QUICK = bool(os.environ.get("QUAD4_QUICK"))
SWEEP = bool(os.environ.get("QUAD4_SWEEP"))

N_GEOM_TRAIN = 14000          # synthetische Geometrien (70 % Anteil)
N_GEOM_VAL = 2000
STATES_PER_ELEM = 12

FRAC_SYNTH = 0.70
FRAC_TRAJ = 0.20
FRAC_AL = 0.10                # Active-Learning-Slot (Runde 1: synthetisch gefuellt)

NUM_EPOCHS = 400              # 240k Samples -> ~60 Batches/Epoche (viele Schritte)
EARLY_STOP_PATIENCE = 60
MINI_BATCH = 4096
K_SUBSAMPLE = 1024            # K-Term je Batch auf rotierendem Subsample
LEARN_RATE = 1e-3
GRAD_CLIP = 1.0

LAMBDA_W = 0.1
LAMBDA_F = 1.0
LAMBDA_K = 1.0

# Einzeltraining (ohne QUAD4_SWEEP): per Env ueberschreibbar.
# Default h48/d3 -- im Sweep-Quick-Pass bereits eK 1.24 % / eF 0.82 % bei nur
# 5 424 MACs. Da der K0-Split die Zielfunktion stark vereinfacht, zahlen sich
# groessere Netze kaum aus, kosten aber direkt Assemblierungszeit.
HIDDEN_DEFAULT = int(os.environ.get("QUAD4_HIDDEN", 48))
DEPTH_DEFAULT = int(os.environ.get("QUAD4_DEPTH", 3))
SWEEP_HIDDEN = [48, 64, 96, 128]
SWEEP_DEPTH = [3, 4]

GO_MEAN = 2.0                 # % mittlerer Total-Fehler (Go-Kriterium)
GO_P99 = 5.0                  # % P99

if QUICK:
    N_GEOM_TRAIN = 1200
    N_GEOM_VAL = 300
    STATES_PER_ELEM = 6
    NUM_EPOCHS = 120
    EARLY_STOP_PATIENCE = 60

TRIU_IDX = np.array(ref.TRIU_IJ)


# ---------------------------------------------------------------------------
# Datensatz
# ---------------------------------------------------------------------------

def _sample_pack(coords_c, z_disp):
    """Ein Sample: kanonische Geometrie + kanonischer Zustand -> alle Targets."""
    W, F, K = ref.energy_stiffness_force_ref(coords_c, z_disp)
    K0 = ref.k0_ref(coords_c)
    mE, _ = ref.hull_measures(coords_c, z_disp)
    return dict(
        chat=coords_c.reshape(-1),
        z=z_disp.reshape(-1),
        W=W,
        F=F,
        Ktriu=ref.triu_vec(K),
        K0triu=ref.triu_vec(K0),
        amp=mE,
        dist=ref.detj_ratio(coords_c),
    )


def build_synthetic(n_geom, states_per_elem, rng, label=""):
    out = []
    t0 = time.time()
    n = 0
    while n < n_geom:
        coords = ref.generate_distorted_quad(rng)
        if coords is None:
            continue
        coords_c = ref.canonicalize_coords(coords)
        K0 = ref.k0_ref(coords_c)
        K0triu = ref.triu_vec(K0)
        dist = ref.detj_ratio(coords_c)
        for _ in range(states_per_elem):
            u = ref.corot_state(coords_c, ref.sample_state(coords_c, rng))
            W, F, K = ref.energy_stiffness_force_ref(coords_c, u)
            mE, _ = ref.hull_measures(coords_c, u)
            out.append(dict(chat=coords_c.reshape(-1), z=u.reshape(-1), W=W, F=F,
                            Ktriu=ref.triu_vec(K), K0triu=K0triu, amp=mE, dist=dist))
        n += 1
    log(f"  synthetisch ({label}): {len(out)} Samples in {time.time()-t0:.1f} s")
    return out


def build_trajectories(rng):
    """Newton-Trajektorien aus MATLAB laden, kanonisieren, ko-rotieren."""
    traj_file = save_dir / "newton_traj_states.mat"
    if not traj_file.exists():
        log("  WARNUNG: newton_traj_states.mat fehlt -- Trajektorien-Anteil wird")
        log("           mit synthetischen Samples aufgefuellt. Fuer den vollen")
        log("           Datenmix zuerst generate_newton_trajectories.m ausfuehren.")
        return [], []
    md = scipy.io.loadmat(str(traj_file))
    coords_all = md["coords"]                     # (4,2,N)
    Ue_all = md["Ue"]                             # (8,N)
    is_val = md["is_val"].reshape(-1)
    N = Ue_all.shape[1]
    log(f"  Trajektorien-Rohdaten: {N} Zustaende "
        f"({int(is_val.sum())} aus Val-Strukturen)")

    train_s, val_s = [], []
    skipped = 0
    for i in range(N):
        coord_e = coords_all[:, :, i]
        Ue = Ue_all[:, i]
        coords_c, Lc, Rc = ref.canonicalization_frame(coord_e)
        if not ref.in_geometry_hull(coords_c):
            skipped += 1
            continue
        U_nodes = Ue.reshape(4, 2)
        u_c = (U_nodes @ Rc.T) / Lc
        u_c = u_c - u_c.mean(axis=0)
        u_c = ref.corot_state(coords_c, u_c)
        mE, _ = ref.hull_measures(coords_c, u_c)
        if mE > ref.E_MAX:
            skipped += 1
            continue
        s = _sample_pack(coords_c, u_c)
        (val_s if is_val[i] > 0.5 else train_s).append(s)
    log(f"  Trajektorien nach Huellen-Filter: {len(train_s)} train / {len(val_s)} val "
        f"(verworfen: {skipped})")
    return train_s, val_s


def _subsample(lst, n, rng):
    if len(lst) <= n:
        return lst
    idx = rng.choice(len(lst), size=n, replace=False)
    return [lst[i] for i in idx]


def to_arrays(samples):
    return dict(
        chat=np.array([s["chat"] for s in samples], dtype=np.float32),
        z=np.array([s["z"] for s in samples], dtype=np.float32),
        W=np.array([s["W"] for s in samples], dtype=np.float32),
        F=np.array([s["F"] for s in samples], dtype=np.float32),
        Ktriu=np.array([s["Ktriu"] for s in samples], dtype=np.float32),
        K0triu=np.array([s["K0triu"] for s in samples], dtype=np.float32),
        amp=np.array([s["amp"] for s in samples], dtype=np.float32),
        dist=np.array([s["dist"] for s in samples], dtype=np.float32),
    )


def build_full_dataset():
    """Kompletter Datensatz im Mix 70 % synthetisch / 20 % Trajektorien /
    10 % Active-Learning-Slot.

    Rueckgabe: (tr_arr, va_arr, n_syn_val) -- n_syn_val = Anzahl synthetischer
    Validierungssamples; alles danach stammt aus Newton-Trajektorien (fuer die
    getrennte Auswertung).
    """
    rng = np.random.RandomState(1234)
    traj_train, traj_val = build_trajectories(rng)

    n_syn_states = N_GEOM_TRAIN * STATES_PER_ELEM
    total_target = int(round(n_syn_states / FRAC_SYNTH))
    n_traj_target = int(round(total_target * FRAC_TRAJ))
    n_al_target = int(round(total_target * FRAC_AL))

    syn_train = build_synthetic(N_GEOM_TRAIN, STATES_PER_ELEM, rng, "train")
    traj_train = _subsample(traj_train, n_traj_target, rng)

    # AL-Slot: in Runde 1 mit zusaetzlichen synthetischen Samples gefuellt
    # (Active Learning ersetzt ihn in der Kontingenz-Runde, Plan Phase 6).
    n_al_geom = max(1, int(round(n_al_target / STATES_PER_ELEM)))
    al_train = build_synthetic(n_al_geom, STATES_PER_ELEM, rng, "AL-Slot")

    # fehlende Trajektorien ggf. synthetisch auffuellen
    missing = n_traj_target - len(traj_train)
    if missing > 0:
        extra_geom = max(1, int(round(missing / STATES_PER_ELEM)))
        traj_train = traj_train + build_synthetic(extra_geom, STATES_PER_ELEM,
                                                  rng, "Traj-Ersatz")

    train_samples = syn_train + traj_train + al_train
    syn_val = build_synthetic(N_GEOM_VAL, STATES_PER_ELEM, rng, "val")
    val_samples = syn_val + _subsample(traj_val, 5000, rng)

    log(f"  Mix: {len(syn_train)} synth + {len(traj_train)} traj/ersatz "
        f"+ {len(al_train)} AL-Slot")
    return to_arrays(train_samples), to_arrays(val_samples), len(syn_val)


# ---------------------------------------------------------------------------
# Datensatz-Cache
# ---------------------------------------------------------------------------
#   Der Aufbau kostet ~10 min reine CPU-Zeit (analytische Targets in NumPy),
#   liefert aber bei gleichen Parametern und gleichem Seed exakt denselben
#   Datensatz. Der Cache-Schluessel deckt alles ab, was den Inhalt bestimmt --
#   aendert sich ein Parameter, wird automatisch neu gebaut.

def dataset_cache_key():
    traj = save_dir / "newton_traj_states.mat"
    traj_stamp = (f"{traj.stat().st_size}_{int(traj.stat().st_mtime)}"
                  if traj.exists() else "none")
    parts = [
        "v1", QUICK, N_GEOM_TRAIN, N_GEOM_VAL, STATES_PER_ELEM,
        FRAC_SYNTH, FRAC_TRAJ, FRAC_AL, traj_stamp,
        ref.E_MAX, ref.ENV_RATIO_MAX, ref.ENV_ANGLE_MIN, ref.ENV_ANGLE_MAX,
        ref.ENV_TAPER_MAX, ref.AMP_LOG_MIN, ref.P_ZERO,
        ref.NU_TRAIN, ref.CONDITION, ref.MATERIAL, 1234,
    ]
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


def load_dataset_cache(key):
    path = save_dir / f"dataset_cache_{key}.npz"
    if not path.exists():
        return None
    try:
        d = np.load(path)
        tr = {k[3:]: d[k] for k in d.files if k.startswith("tr_")}
        va = {k[3:]: d[k] for k in d.files if k.startswith("va_")}
        n_syn_val = int(d["n_syn_val"])
        log(f"  Datensatz aus Cache: {path.name} "
            f"({len(tr['chat'])} train / {len(va['chat'])} val)")
        return tr, va, n_syn_val
    except Exception as exc:                      # beschaedigter Cache
        log(f"  Cache unlesbar ({exc}) -- wird neu gebaut.")
        return None


def save_dataset_cache(key, tr_arr, va_arr, n_syn_val):
    path = save_dir / f"dataset_cache_{key}.npz"
    payload = {f"tr_{k}": v for k, v in tr_arr.items()}
    payload.update({f"va_{k}": v for k, v in va_arr.items()})
    payload["n_syn_val"] = np.array(n_syn_val)
    # unkomprimiert: Ziel ist Zeit, nicht Platz -- Komprimieren kostet mehr
    # CPU, als das Laden spart.
    np.savez(path, **payload)
    log(f"  Datensatz gecacht: {path.name} "
        f"({path.stat().st_size / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Netz: roher Skalar-MLP + Subtraktionsform + Normalisierung
# ---------------------------------------------------------------------------

class WNet(nn.Module):
    """Residual-Energie W_NL(c, z). Ableitungen werden extern gebildet."""

    def __init__(self, hidden, depth, c_mean, c_std, z_scale):
        super().__init__()
        layers = [nn.Linear(16, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*layers)
        with torch.no_grad():                      # Klein-Init der Ausgabeschicht
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.zero_()
        self.register_buffer("c_mean", c_mean)
        self.register_buffer("c_std", c_std)
        self.register_buffer("z_scale", z_scale)

    def raw(self, ct, zt):
        return self.mlp(torch.cat([ct, zt], dim=-1)).squeeze(-1)

    def norm_c(self, c):
        return (c - self.c_mean) / self.c_std

    def norm_z(self, z):
        return z / self.z_scale


def net_terms(model, c, z, need_hess=True):
    """f, grad und Hessian des ROHEN MLP bzgl. zt -- plus Subtraktionsform.

    Rueckgabe (alles auf ROHE z bezogen, d.h. Normalisierung ausgerechnet):
        W_NL (B,), F_NL (B,8), K_NL (B,8,8) oder None
    Identische Zerlegung rechnet das MATLAB-Element.

    need_hess=False ueberspringt den Hessian -- er ist der mit Abstand teuerste
    Teil (8 Vorwaertsrichtungen durch den Reverse-Pass) und wird fuer den
    W/F-Anteil des Losses gar nicht gebraucht.
    """
    ct = model.norm_c(c)
    zt = model.norm_z(z)
    zs = model.z_scale
    zero = torch.zeros_like(zt)

    def f_of_zt(zt_row, ct_row):
        return model.raw(ct_row, zt_row)

    g_fn = grad(f_of_zt)
    f_val = vmap(f_of_zt)(zt, ct)
    f_0 = vmap(f_of_zt)(zero, ct)
    g_val = vmap(g_fn)(zt, ct)
    g_0 = vmap(g_fn)(zero, ct)

    W_NL = f_val - f_0 - (g_0 * zt).sum(dim=-1)
    F_NL = (g_val - g_0) / zs
    if not need_hess:
        return W_NL, F_NL, None

    H_val = vmap(jacfwd(g_fn))(zt, ct)
    K_NL = H_val / (zs.unsqueeze(-1) * zs.unsqueeze(-2))
    return W_NL, F_NL, K_NL


# ---------------------------------------------------------------------------
# Loss und Metriken (Residuen lernen, Totalgroessen normieren)
# ---------------------------------------------------------------------------

_ti = torch.tensor(TRIU_IDX[:, 0], dtype=torch.long)
_tj = torch.tensor(TRIU_IDX[:, 1], dtype=torch.long)


def triu_to_full_t(vec):
    """(B,36) -> (B,8,8) symmetrisch."""
    B = vec.shape[0]
    K = torch.zeros(B, 8, 8, device=vec.device, dtype=vec.dtype)
    ii = _ti.to(vec.device)
    jj = _tj.to(vec.device)
    K[:, ii, jj] = vec
    K[:, jj, ii] = vec
    return K


class Batch:
    """Vorbereitete Tensoren eines Datensatzes."""

    def __init__(self, arr, dev):
        self.c = torch.tensor(arr["chat"], device=dev)
        self.z = torch.tensor(arr["z"], device=dev)
        self.W_tot = torch.tensor(arr["W"], device=dev)
        self.F_tot = torch.tensor(arr["F"], device=dev)
        self.K_tot = triu_to_full_t(torch.tensor(arr["Ktriu"], device=dev))
        self.K0 = triu_to_full_t(torch.tensor(arr["K0triu"], device=dev))
        self.amp = torch.tensor(arr["amp"], device=dev)
        self.dist = torch.tensor(arr["dist"], device=dev)

        # Residual-Targets
        self.F_lin = torch.einsum("bij,bj->bi", self.K0, self.z)
        self.W_lin = 0.5 * (self.z * self.F_lin).sum(dim=-1)
        self.W_NL_t = self.W_tot - self.W_lin
        self.F_NL_t = self.F_tot - self.F_lin
        self.K_NL_t = self.K_tot - self.K0

        # Normen der TOTALGROESSEN (das, was Newton sieht)
        self.nW = self.W_tot.abs()
        self.nF = self.F_tot.norm(dim=-1)
        self.nK = self.K_tot.norm(dim=(-2, -1))

    def __len__(self):
        return self.c.shape[0]


def make_floors(b: Batch):
    floorW = 0.05 * torch.sqrt((b.W_tot ** 2).mean())
    floorF = 0.05 * torch.sqrt((b.nF ** 2).mean())
    return floorW, floorF


def sobolev_loss(model, b: Batch, idx, kidx, floorW, floorF):
    c = b.c[idx]
    z = b.z[idx]
    W_NL, F_NL, _ = net_terms(model, c, z, need_hess=False)
    lW = (((W_NL - b.W_NL_t[idx]) ** 2) / (b.W_tot[idx] ** 2 + floorW ** 2)).mean()
    lF = (((F_NL - b.F_NL_t[idx]) ** 2).sum(-1)
          / (b.nF[idx] ** 2 + floorF ** 2)).mean()
    # K-Term auf kleinerem Subsample (Speicher/Zeit)
    _, _, K_NL = net_terms(model, b.c[kidx], b.z[kidx])
    lK = (((K_NL - b.K_NL_t[kidx]) ** 2).sum((-2, -1))
          / (b.nK[kidx] ** 2)).mean()
    return LAMBDA_W * lW + LAMBDA_F * lF + LAMBDA_K * lK, lW, lF, lK


def evaluate(model, b: Batch, chunk=8192):
    """Relative TOTAL-Fehler eF, eK je Sample [%]."""
    model.eval()
    eF_all, eK_all = [], []
    floorF = 0.02 * torch.sqrt((b.nF ** 2).mean())
    for s in range(0, len(b), chunk):
        sl = slice(s, min(s + chunk, len(b)))
        _, F_NL, K_NL = net_terms(model, b.c[sl], b.z[sl])
        F_pred = b.F_lin[sl] + F_NL
        K_pred = b.K0[sl] + K_NL
        eF = (F_pred - b.F_tot[sl]).norm(dim=-1) / torch.clamp(b.nF[sl], min=floorF)
        eK = (K_pred - b.K_tot[sl]).norm(dim=(-2, -1)) / b.nK[sl]
        eF_all.append(eF.detach())
        eK_all.append(eK.detach())
    return (torch.cat(eF_all).cpu().numpy() * 100.0,
            torch.cat(eK_all).cpu().numpy() * 100.0)


def arch_macs(hidden, depth):
    """MACs eines Forward-Pass des Skalar-MLP (Kostenmass fuer das Element)."""
    return 16 * hidden + (depth - 1) * hidden * hidden + hidden


def stats(e):
    return dict(mean=float(np.mean(e)), p50=float(np.percentile(e, 50)),
                p90=float(np.percentile(e, 90)), p95=float(np.percentile(e, 95)),
                p99=float(np.percentile(e, 99)), max=float(np.max(e)))


def report_bins(eF, eK, amp, dist, tag, results):
    """Perzentil-Tabellen gesamt, je Amplituden-Terzil und Verzerrungs-Bin."""
    log(f"\n  --- {tag} ---")
    log("    Gruppe                 |   n   | eF mean   p50   p90   p95   p99 "
        "|  eK mean   p50   p90   p95   p99")

    def row(name, mask):
        if mask.sum() < 5:
            return
        sF, sK = stats(eF[mask]), stats(eK[mask])
        log(f"    {name:22s} | {int(mask.sum()):5d} | "
            f"{sF['mean']:8.3f} {sF['p50']:5.2f} {sF['p90']:5.2f} {sF['p95']:5.2f} {sF['p99']:6.2f} | "
            f"{sK['mean']:8.3f} {sK['p50']:5.2f} {sK['p90']:5.2f} {sK['p95']:5.2f} {sK['p99']:6.2f}")
        results[f"{tag}|{name}"] = dict(n=int(mask.sum()), eF=sF, eK=sK)

    row("gesamt", np.ones_like(amp, dtype=bool))
    q = np.quantile(amp, [1 / 3, 2 / 3])
    row("Amp-Terzil 1 (klein)", amp <= q[0])
    row("Amp-Terzil 2", (amp > q[0]) & (amp <= q[1]))
    row("Amp-Terzil 3 (gross)", amp > q[1])
    row("Verzerrung < 1.5", dist < 1.5)
    row("Verzerrung 1.5-2.5", (dist >= 1.5) & (dist < 2.5))
    row("Verzerrung >= 2.5", dist >= 2.5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(hidden, depth, tr: Batch, va: Batch, norm, epochs=NUM_EPOCHS, tag=""):
    c_mean, c_std, z_scale = norm
    model = WNet(hidden, depth, c_mean, c_std, z_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-5)

    floorW, floorF = make_floors(tr)
    n = len(tr)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"\n--- {tag}GELU | Hidden {hidden} | Tiefe {depth} | "
        f"Parameter {nparam:,} | MACs/Forward {arch_macs(hidden, depth):,} ---")

    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, MINI_BATCH):
            idx = perm[s:s + MINI_BATCH]
            kidx = idx[torch.randperm(len(idx), device=device)[:K_SUBSAMPLE]]
            loss, _, _, _ = sobolev_loss(model, tr, idx, kidx, floorW, floorF)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

        with torch.no_grad():
            vidx = torch.arange(min(len(va), 8192), device=device)
            vkidx = vidx[:K_SUBSAMPLE]
        vloss, lW, lF, lK = sobolev_loss(model, va, vidx, vkidx, floorW, floorF)
        vloss = float(vloss.detach())
        lW, lF, lK = float(lW.detach()), float(lF.detach()), float(lK.detach())
        sched.step(vloss)

        if vloss < best - 1e-6:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1

        if ep % 25 == 0 or ep == 1:
            log(f"    Epoch {ep:4d}/{epochs} | Val {vloss:.4e} "
                f"(W {lW:.2e} / F {lF:.2e} / K {lK:.2e}) | "
                f"LR {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f} s")
        if bad >= EARLY_STOP_PATIENCE:
            log(f"    Early stop bei Epoche {ep} (Val-Plateau).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    Fertig in {time.time()-t0:.1f} s | bester Val-Loss {best:.4e}")
    return model, best


# ---------------------------------------------------------------------------
# Gate c: autograd-K vs FD(F) in fp64
# ---------------------------------------------------------------------------

def gate_c(model, b: Batch, n=5, verbose=True):
    model_d = model.double()
    worst_fd, worst_sym = 0.0, 0.0
    for i in range(n):
        c = b.c[i:i + 1].double()
        z = b.z[i:i + 1].double()
        _, F0, K0n = net_terms(model_d, c, z)
        F0 = F0[0]
        K0n = K0n[0]
        h = 1e-6
        Jfd = torch.zeros(8, 8, dtype=torch.float64, device=z.device)
        for j in range(8):
            zp = z.clone(); zp[0, j] += h
            zm = z.clone(); zm[0, j] -= h
            _, Fp, _ = net_terms(model_d, c, zp)
            _, Fm, _ = net_terms(model_d, c, zm)
            Jfd[:, j] = (Fp[0] - Fm[0]) / (2 * h)
        den = max(float(K0n.norm()), 1e-12)
        worst_fd = max(worst_fd, float((K0n - Jfd).norm()) / den)
        worst_sym = max(worst_sym, float((K0n - K0n.T).norm()) / den)
    model.float()
    ok = worst_fd <= 1e-6 and worst_sym <= 1e-10
    if verbose:
        log(f"\nGate c -- K_NL vs FD(F_NL): {worst_fd:.3e} (<= 1e-6), "
            f"Symmetrie {worst_sym:.3e}  -> {'GRUEN' if ok else 'ROT'}")
    return ok, worst_fd, worst_sym


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(save_dir),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "no-git"


def build_mat(model, hidden, depth, va: Batch, res_json):
    model.eval()
    md = {
        "model_form": "resid_energy_K0split_subtract_f0_gradf0",
        "activation": "GELU_erf",
        "input_order": "canonical_[c_hat(8)_then_z(8)]",
        "canonicalization": "edge_n1n2_to_pos_x_after_centroid_Lc",
        "state_corotation": "mean_polar_angle_at_center_removed__u_c=R(-th)(x+u)-x",
        "state_input": "canonical_displacements_translation_projected_corotated",
        "scaling": "W=E*d*Lc^2*What; Finte=E*d*Lc*(chain); Ke=E*d*(chain)",
        "material": ref.MATERIAL,
        "nu_train": ref.NU_TRAIN,
        "condition": ref.CONDITION,
        "k0_source": "analytic_linear_gauss_loop_per_call_no_cache",
        "env_ratio_max": ref.ENV_RATIO_MAX,
        "env_angle_min": ref.ENV_ANGLE_MIN,
        "env_angle_max": ref.ENV_ANGLE_MAX,
        "state_E_max": ref.E_MAX,
        "state_rot_max_deg": 180.0,
        "hidden": hidden,
        "depth": depth,
        "lambda_W": LAMBDA_W, "lambda_F": LAMBDA_F, "lambda_K": LAMBDA_K,
        "train_script": "train_quad4_nl_W_network.py",
        "git_hash": git_hash(),
        "seed": 42,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_norm_c_mean": model.c_mean.detach().cpu().numpy().astype(np.float64),
        "input_norm_c_std": model.c_std.detach().cpu().numpy().astype(np.float64),
        "input_norm_z_scale": model.z_scale.detach().cpu().numpy().astype(np.float64),
        "metrics_json": json.dumps(res_json),
    }

    lins = [m for m in model.mlp if isinstance(m, nn.Linear)]
    md["num_linear_layers"] = len(lins)
    for i, lin in enumerate(lins, start=1):
        md[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float64)
        md[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float64)

    # Oracle-Testvektoren fuer MATLAB (Gate d): MODELL-TOTALS auf 64 Val-Samples.
    # Nur NICHT-degenerierte Zustaende: ~12 % der Samples sind exakt z = 0
    # (P_ZERO im Sampler) -- dort sind W, F exakt null, ein relativer Vergleich
    # in MATLAB waere eine Division durch null und wuerde nichts pruefen.
    zn = va.z.norm(dim=-1)
    cand = torch.nonzero(zn > torch.quantile(zn, 0.5), as_tuple=False).flatten()
    sel = cand[torch.linspace(0, len(cand) - 1, min(64, len(cand))).long()]

    # Auswertung in fp64: MATLAB rechnet in double. Ein fp32-Export waere kein
    # sauberer Test der Rekurrenzen, weil K0*z durch den Nullraum von K0
    # Ausloeschung enthaelt -- der fp32-Rundungsfehler wird dort um Groessen-
    # ordnungen verstaerkt und ueberdeckt den eigentlichen Vergleich.
    c = va.c[sel].double()
    z = va.z[sel].double()
    model.double()
    with torch.no_grad():
        pass
    W_NL, F_NL, K_NL = net_terms(model, c, z)
    W_NL, F_NL, K_NL = W_NL.detach(), F_NL.detach(), K_NL.detach()
    model.float()

    c_np = c.cpu().numpy()
    K0_64 = torch.tensor(
        np.stack([ref.k0_ref(cc.reshape(4, 2)) for cc in c_np]),
        dtype=torch.float64, device=c.device)
    F_lin = torch.einsum("bij,bj->bi", K0_64, z)

    W_tot = 0.5 * (z * F_lin).sum(-1) + W_NL
    F_tot = F_lin + F_NL
    K_tot = K0_64 + K_NL
    ii = torch.tensor(TRIU_IDX[:, 0], device=K_tot.device)
    jj = torch.tensor(TRIU_IDX[:, 1], device=K_tot.device)
    md["test_C"] = c.cpu().numpy()
    md["test_Z"] = z.cpu().numpy()
    md["test_W"] = W_tot.cpu().numpy().reshape(-1, 1)
    md["test_F"] = F_tot.cpu().numpy()
    md["test_K"] = K_tot[:, ii, jj].cpu().numpy()
    md["test_precision"] = "float64"
    return md


# ---------------------------------------------------------------------------
# Hauptlauf
# ---------------------------------------------------------------------------

def main():
    log("=== Residual-Energie-Netz quad4 nichtlinear (K0-Split, Sobolev) ===\n")
    log(f"Logdatei: {log_path}")
    log(f"Device:   {device}")
    if device.type == "cuda":
        log(f"GPU:      {torch.cuda.get_device_name(0)}")
    log(f"Material: {ref.MATERIAL}, nu = {ref.NU_TRAIN}, {ref.CONDITION} | E = d = 1")
    log(f"Modus:    {'QUICK' if QUICK else 'voll'}{' + SWEEP' if SWEEP else ''}\n")

    # --- Gates a und b (ohne Netz, PFLICHT vor jedem Training) --------------
    log("Gates a/b (Referenz + Kette, ohne Netz):")
    ok_ab, _ = ref.run_gates_a_b(verbose=True, strict=True)
    if not ok_ab:
        raise SystemExit("Gates a/b rot -- Training abgebrochen.")

    # --- Datensatz (einmal pro Lauf, ueber Laeufe hinweg gecacht) -----------
    key = dataset_cache_key()
    cached = load_dataset_cache(key)
    if cached is not None:
        tr_arr, va_arr, n_syn_val = cached
    else:
        log("\nDatensatz aufbauen:")
        tr_arr, va_arr, n_syn_val = build_full_dataset()
        save_dataset_cache(key, tr_arr, va_arr, n_syn_val)

    log(f"\n  Training gesamt   : {len(tr_arr['chat'])} Samples")
    log(f"  Validierung gesamt: {len(va_arr['chat'])} Samples "
        f"({n_syn_val} synth + {len(va_arr['chat']) - n_syn_val} traj)")

    is_traj_val = np.zeros(len(va_arr["chat"]), dtype=bool)
    is_traj_val[n_syn_val:] = True

    # --- Normalisierung (aus dem Trainingssatz) -----------------------------
    c_mean = torch.tensor(tr_arr["chat"].mean(axis=0), device=device)
    c_std = torch.tensor(np.maximum(tr_arr["chat"].std(axis=0), 1e-6), device=device)
    z_scale = torch.tensor(np.maximum(tr_arr["z"].std(axis=0), 1e-8), device=device)
    log(f"\n  Normalisierung: z_scale (min/max) = "
        f"{float(z_scale.min()):.3e} / {float(z_scale.max()):.3e}")

    tr = Batch(tr_arr, device)
    va = Batch(va_arr, device)
    norm = (c_mean, c_std, z_scale)

    # --- Architektur: Sweep oder Default ------------------------------------
    if SWEEP and not QUICK:
        log("\n=== Quick-Pass Sweep ===")
        quick_n = min(len(tr), 60000)
        qidx = np.random.RandomState(0).choice(len(tr), quick_n, replace=False)
        tr_q = Batch({k: v[qidx] for k, v in tr_arr.items()}, device)
        cands = []
        for h in SWEEP_HIDDEN:
            for d in SWEEP_DEPTH:
                m, _ = train_one(h, d, tr_q, va, norm, epochs=200, tag="[quick] ")
                eF, eK = evaluate(m, va)
                sF, sK = stats(eF), stats(eK)
                score = sK["mean"] + 0.5 * sK["p99"]
                cands.append(dict(score=score, h=h, d=d, macs=arch_macs(h, d),
                                  eF=sF["mean"], eK=sK["mean"]))
                log(f"    -> eK mean {sK['mean']:.3f} % | p99 {sK['p99']:.3f} % | "
                    f"eF mean {sF['mean']:.3f} % | MACs {arch_macs(h, d):,}")

        # Auswahl fuer das Volltraining: EIN genauester + EIN billigster
        # Kandidat. Das Forschungsziel ist Geschwindigkeit -- eine reine
        # Genauigkeits-Rangliste wuerde systematisch das groesste Netz
        # waehlen und den Speedup verschenken.
        best_c = min(cands, key=lambda c: c["score"])
        tol = 1.5 * best_c["score"]
        cheap = min([c for c in cands if c["score"] <= tol],
                    key=lambda c: c["macs"])
        archs = [(cheap["h"], cheap["d"])]
        if (best_c["h"], best_c["d"]) != (cheap["h"], cheap["d"]):
            archs.append((best_c["h"], best_c["d"]))
        log(f"\n  Volltraining: billigste brauchbare {archs[0]} "
            f"({cheap['macs']:,} MACs, Score {cheap['score']:.2f})"
            + (f" + genaueste {archs[1]} (Score {best_c['score']:.2f})"
               if len(archs) > 1 else " (zugleich die genaueste)"))
    else:
        archs = [(HIDDEN_DEFAULT, DEPTH_DEFAULT)]

    # --- Volltraining -------------------------------------------------------
    trained = []
    for (h, d) in archs:
        model, _ = train_one(h, d, tr, va, norm, tag="[voll] ")
        eF, eK = evaluate(model, va)
        sF, sK = stats(eF), stats(eK)
        ok_go = (sF["mean"] < GO_MEAN and sK["mean"] < GO_MEAN
                 and sF["p99"] < GO_P99 and sK["p99"] < GO_P99)
        trained.append(dict(model=model, h=h, d=d, macs=arch_macs(h, d),
                            eF=eF, eK=eK, go=ok_go,
                            score=sK["mean"] + 0.5 * sK["p99"]))
        log(f"    -> eF mean {sF['mean']:.3f} % / p99 {sF['p99']:.3f} % | "
            f"eK mean {sK['mean']:.3f} % / p99 {sK['p99']:.3f} % | "
            f"Go: {'ja' if ok_go else 'nein'}")

    # Deployment: BILLIGSTE Variante, die das Go-Kriterium erfuellt (Speedup
    # ist das Ziel); erfuellt keine das Kriterium, die genaueste.
    passing = [t for t in trained if t["go"]]
    if passing:
        pick = min(passing, key=lambda t: t["macs"])
        log(f"\n  Deployment: billigste Variante mit erfuelltem Go-Kriterium "
            f"(h{pick['h']} d{pick['d']}, {pick['macs']:,} MACs)")
    else:
        pick = min(trained, key=lambda t: t["score"])
        log(f"\n  Deployment: keine Variante erfuellt das Go-Kriterium -> "
            f"genaueste (h{pick['h']} d{pick['d']})")

    model, hidden, depth, eF, eK = pick["model"], pick["h"], pick["d"], pick["eF"], pick["eK"]

    # --- Metriken -----------------------------------------------------------
    res_json = {}
    log("\n=== Ergebnis (relative TOTAL-Fehler [%], das was Newton sieht) ===")
    report_bins(eF, eK, va_arr["amp"], va_arr["dist"], "Validierung gesamt", res_json)
    if is_traj_val.any():
        report_bins(eF[~is_traj_val], eK[~is_traj_val], va_arr["amp"][~is_traj_val],
                    va_arr["dist"][~is_traj_val], "nur synthetisch", res_json)
        report_bins(eF[is_traj_val], eK[is_traj_val], va_arr["amp"][is_traj_val],
                    va_arr["dist"][is_traj_val], "nur Trajektorien", res_json)

    # --- Gate c -------------------------------------------------------------
    ok_c, fd_err, sym_err = gate_c(model, va)
    res_json["gate_c"] = dict(fd=fd_err, sym=sym_err, ok=bool(ok_c))

    # --- Go-Kriterium -------------------------------------------------------
    sF, sK = stats(eF), stats(eK)
    go = (sF["mean"] < GO_MEAN and sK["mean"] < GO_MEAN
          and sF["p99"] < GO_P99 and sK["p99"] < GO_P99)
    log(f"\nGo-Kriterium (mean < {GO_MEAN} % UND p99 < {GO_P99} %): "
        f"eF {sF['mean']:.3f}/{sF['p99']:.3f} | eK {sK['mean']:.3f}/{sK['p99']:.3f}"
        f"  -> {'ERFUELLT' if go else 'VERFEHLT'}")
    res_json["go"] = bool(go)

    # --- Export (nur bei gruenem Gate c) ------------------------------------
    if not ok_c:
        log("\nGate c ROT -> KEIN Deployment (kein stiller Fallback).")
        raise SystemExit(1)

    md = build_mat(model, hidden, depth, va, res_json)
    out_mat = (element_dir / "quad4_nl_W_network.mat").resolve()
    scipy.io.savemat(str(out_mat), md)
    torch.save(model.state_dict(), sweep_dir / f"quad4_nl_W_{_stamp}.pt")
    with open(sweep_dir / f"quad4_nl_W_{_stamp}_metrics.json", "w") as fh:
        json.dump(res_json, fh, indent=2)

    log(f"\nDeployt: {out_mat}")
    log(f"         Architektur GELU h{hidden} d{depth}, "
        f"{sum(p.numel() for p in model.parameters()):,} Parameter")
    if not go:
        log("HINWEIS: Go-Kriterium verfehlt -- Kontingenzliste (Plan Phase 6) "
            "abarbeiten, beginnend mit der Active-Learning-Runde.")


if __name__ == "__main__":
    main()
