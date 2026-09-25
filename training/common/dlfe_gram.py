"""Gram-Kette + Voll-Energie-Netz, dimensionsunabhaengig (quad4 UND brick8).

Idee (ersetzt die 2D-Ko-Rotation, die in 3D keine geschlossene Form hat):
Fuer jedes hyperelastische Material haengt die Elementenergie nur von
C = F'F ab, und C ist an jedem Punkt eine LINEARE Funktion der Gram-Matrix
der deformierten Knotenpositionen. Deshalb ist

    W_hat(c, u) = f(c~, D~) - f(c~, 0) - grad_D~ f(c~, 0)' D~

mit dem METRIK-Eingang (zentrierte Knoten, Lc = 1, beliebige Orientierung)

    D_ij = y_i . y_j - x_i . x_j = x_i.u_j + u_i.x_j + u_i.u_j     (i <= j)

exakt objektiv (Starrkoerperbewegung -> D = 0 -> W = 0, F = 0), ohne
Polarzerlegung. Die Geometrie c_hat wird wie bisher kanonisiert (Zentroid,
Lc, Orientierung) -- die ZUSTANDSgroesse braucht keine Drehung.

Konventionen (identisch MATLAB):
    c, u : (ndof,) = [x1, y1, (z1), x2, ...]      (knotenweise, row-major)
    D    : (m,)    triu der n x n Gram-Differenz, column-major
                   (for j: for i <= j)  == MATLAB find(triu(true(n)))
    K    : triu der ndof x ndof Matrix, column-major (Oracle-Export)

Zwei Zustandsformen (GramSpec.modal):
    Knoten-Gram  D = triu(Y Y' - X X')               state_form 'gram_metric_centered_triu'
    Modal-Gram   D = triu(L'(Y Y' - X X')L), L = [dh0 | gamma]
                 = triu([F0'F0 - I, F0'q ; q'F0, q'q])  'gram_modal_F0_hourglass_triu'
Die modale Form ist in Phase 0 (quad4) rund 3x genauer und der Standard.

Das MATLAB-Gegenstueck ist sourcecode/elements/dlfe/ (dlfe_gram_energy,
dlfe_mlp, dlfe_load_network, dlfe_canonical_frame, dlfe_mode_matrix). Der
Export traegt state_form -- MATLAB prueft das hart.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.func import grad, jacfwd, vmap

STATE_FORM = "gram_metric_centered_triu"
STATE_FORM_MODAL = "gram_modal_F0_hourglass_triu"
MODEL_FORM = "total_energy_subtract_f0_gradf0"
ACTIVATION = "GELU_erf"


# ---------------------------------------------------------------------------
# Spezifikation (Knotenzahl, Dimension, Indexmengen)
# ---------------------------------------------------------------------------

NODE_NAT = {
    (4, 2): [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    (8, 3): [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
}


def mode_matrix(n, dim):
    """Isoparametrische Moden Phi (n x (n-1)) = [lineare | Hourglass] / 2^dim.

    quad4 : r, s | rs          brick8 : r, s, t | rs, st, rt, rst
    Alle Spalten summieren zu null (orthogonal zu Translationen)."""
    nat = np.array(NODE_NAT[(n, dim)], dtype=np.float64)
    if dim == 2:
        r, s = nat[:, 0], nat[:, 1]
        cols = [r, s, r * s]
    else:
        r, s, t = nat[:, 0], nat[:, 1], nat[:, 2]
        cols = [r, s, t, r * s, s * t, r * t, r * s * t]
    return np.stack(cols, axis=1) / 2.0 ** dim


class GramSpec:
    """modal=False: Knoten-Gram D = triu(Y Y' - X X')            (k = n)
    modal=True : modale Gram  D = triu(L' (Y Y' - X X') L)      (k = n-1)
                 L = [dh0 | gamma] aus der kanonischen Geometrie:
                 dh0   = Phi_lin * J0^-1  (Formfunktionsgradienten im Zentrum)
                 gamma = Phi_hg - dh0 * (X' Phi_hg)   (Flanagan-Belytschko)
                 -> Z = L'Y = [F0' ; q'],  D = triu([2 E0, F0'q ; q'F0, q'q])
    """

    def __init__(self, n: int, dim: int, modal: bool = False):
        self.n = n
        self.dim = dim
        self.modal = modal
        self.ndof = n * dim
        self.k = n - 1 if modal else n
        self.triu = [(i, j) for j in range(self.k) for i in range(j + 1)]
        self.m = len(self.triu)
        self.ti = torch.tensor([a for a, _ in self.triu], dtype=torch.long)
        self.tj = torch.tensor([b for _, b in self.triu], dtype=torch.long)
        self.ktriu = [(i, j) for j in range(self.ndof) for i in range(j + 1)]
        self.kti = torch.tensor([a for a, _ in self.ktriu], dtype=torch.long)
        self.ktj = torch.tensor([b for _, b in self.ktriu], dtype=torch.long)
        self.Phi = torch.tensor(mode_matrix(n, dim)) if modal else None

    @property
    def n_in(self):
        return self.ndof + self.m

    @property
    def state_form(self):
        return STATE_FORM_MODAL if self.modal else STATE_FORM


def modal_L(spec: GramSpec, x):
    """L (n x (n-1)) aus zentrierten kanonischen Koordinaten x (n x dim)."""
    Phi = spec.Phi.to(dtype=x.dtype, device=x.device)
    d = spec.dim
    Pl, Ph = Phi[:, :d], Phi[:, d:]
    J0 = x.T @ Pl                                   # J0(i,k) = dx_i/dr_k
    dh0 = torch.linalg.solve(J0.T, Pl.T).T          # Pl * inv(J0)
    gam = Ph - dh0 @ (x.T @ Ph)
    return torch.cat([dh0, gam], dim=1)


def gram_D(spec: GramSpec, c, u):
    """Metrik-Eingang eines Samples: c, u (ndof,) -> D (m,).

    Ausloeschungsfreie Form (x u' + u x' + u u') -- in fp32 waere
    y y' - x x' fuer kleine u um Groessenordnungen ungenauer.
    """
    x = c.reshape(spec.n, spec.dim)
    v = u.reshape(spec.n, spec.dim)
    x = x - x.mean(0)
    v = v - v.mean(0)
    if spec.modal:
        L = modal_L(spec, x)
        x = L.T @ x
        v = L.T @ v
    G = x @ v.T + v @ x.T + v @ v.T
    return G[spec.ti.to(c.device), spec.tj.to(c.device)]


def gram_D_batch(spec: GramSpec, c, u):
    return vmap(lambda a, b: gram_D(spec, a, b))(c, u)


# ---------------------------------------------------------------------------
# Netz
# ---------------------------------------------------------------------------

class GramWNet(nn.Module):
    """Skalar-MLP f([c~, D~]) mit GELU (exakte erf-Form)."""

    def __init__(self, spec: GramSpec, hidden, depth, c_mean, c_std, D_scale):
        super().__init__()
        self.spec = spec
        layers = [nn.Linear(spec.n_in, hidden), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, 1)]
        self.mlp = nn.Sequential(*layers)
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.1)
            self.mlp[-1].bias.zero_()
        self.register_buffer("c_mean", c_mean)
        self.register_buffer("c_std", c_std)
        self.register_buffer("D_scale", D_scale)

    def raw(self, ct, Dt):
        return self.mlp(torch.cat([ct, Dt], dim=-1)).squeeze(-1)

    def norm_c(self, c):
        return (c - self.c_mean) / self.c_std


def net_terms(model: GramWNet, c, u, need_hess=True):
    """W_hat, F_hat = dW/du, K_hat = d2W/du2 (kanonische Geometrie, Lc = 1).

    Subtraktionsform in D~; f(c~,0) und grad f(c~,0) haengen nur von der
    Geometrie ab und werden einmal je Sample berechnet.
    """
    spec = model.spec
    ct = model.norm_c(c)
    Ds = model.D_scale

    def raw1(Dt, ct1):
        return model.raw(ct1, Dt)

    zero = torch.zeros(c.shape[0], spec.m, device=c.device, dtype=c.dtype)
    f0 = vmap(raw1)(zero, ct)
    g0 = vmap(grad(raw1))(zero, ct)

    def W1(u1, c1, ct1, f01, g01):
        Dt = gram_D(spec, c1, u1) / Ds
        return raw1(Dt, ct1) - f01 - (g01 * Dt).sum()

    W = vmap(W1)(u, c, ct, f0, g0)
    F = vmap(grad(W1))(u, c, ct, f0, g0)
    if not need_hess:
        return W, F, None
    K = vmap(jacfwd(grad(W1)))(u, c, ct, f0, g0)
    return W, F, K


def arch_macs(spec: GramSpec, hidden, depth):
    return spec.n_in * hidden + (depth - 1) * hidden * hidden + hidden


# ---------------------------------------------------------------------------
# Daten-Container, Loss, Metriken
# ---------------------------------------------------------------------------

def triu_to_full_t(spec: GramSpec, vec):
    B = vec.shape[0]
    K = torch.zeros(B, spec.ndof, spec.ndof, device=vec.device, dtype=vec.dtype)
    ii = spec.kti.to(vec.device)
    jj = spec.ktj.to(vec.device)
    K[:, ii, jj] = vec
    K[:, jj, ii] = vec
    return K


class Batch:
    """arr: dict mit chat (N,ndof), u (N,ndof), W, F (N,ndof), Ktriu, amp, dist."""

    def __init__(self, spec: GramSpec, arr, dev):
        self.c = torch.tensor(arr["chat"], device=dev)
        self.u = torch.tensor(arr["u"], device=dev)
        self.W_tot = torch.tensor(arr["W"], device=dev)
        self.F_tot = torch.tensor(arr["F"], device=dev)
        self.K_tot = triu_to_full_t(spec, torch.tensor(arr["Ktriu"], device=dev))
        self.amp = np.asarray(arr["amp"])
        self.dist = np.asarray(arr["dist"])
        self.nF = self.F_tot.norm(dim=-1)
        self.nK = self.K_tot.norm(dim=(-2, -1))

    def __len__(self):
        return self.c.shape[0]


@dataclass
class TrainCfg:
    epochs: int = 400
    patience: int = 60
    batch: int = 4096
    k_sub: int = 1024
    lr: float = 1e-3
    clip: float = 1.0
    lam_W: float = 0.1
    lam_F: float = 1.0
    lam_K: float = 1.0
    go_mean: float = 2.0
    go_p99: float = 5.0
    extra: dict = field(default_factory=dict)


def make_floors(b: Batch):
    floorW = 0.05 * torch.sqrt((b.W_tot ** 2).mean())
    floorF = 0.05 * torch.sqrt((b.nF ** 2).mean())
    return floorW, floorF


def sobolev_loss(model, b: Batch, idx, kidx, floorW, floorF, cfg: TrainCfg):
    W, F, _ = net_terms(model, b.c[idx], b.u[idx], need_hess=False)
    lW = (((W - b.W_tot[idx]) ** 2) / (b.W_tot[idx] ** 2 + floorW ** 2)).mean()
    lF = (((F - b.F_tot[idx]) ** 2).sum(-1) / (b.nF[idx] ** 2 + floorF ** 2)).mean()
    _, _, K = net_terms(model, b.c[kidx], b.u[kidx])
    lK = (((K - b.K_tot[kidx]) ** 2).sum((-2, -1)) / (b.nK[kidx] ** 2)).mean()
    return cfg.lam_W * lW + cfg.lam_F * lF + cfg.lam_K * lK, lW, lF, lK


def evaluate(model, b: Batch, chunk=4096):
    model.eval()
    eF_all, eK_all = [], []
    floorF = 0.02 * torch.sqrt((b.nF ** 2).mean())
    for s in range(0, len(b), chunk):
        sl = slice(s, min(s + chunk, len(b)))
        _, F, K = net_terms(model, b.c[sl], b.u[sl])
        eF_all.append(((F - b.F_tot[sl]).norm(dim=-1)
                       / torch.clamp(b.nF[sl], min=floorF)).detach())
        eK_all.append(((K - b.K_tot[sl]).norm(dim=(-2, -1)) / b.nK[sl]).detach())
    return (torch.cat(eF_all).cpu().numpy() * 100.0,
            torch.cat(eK_all).cpu().numpy() * 100.0)


def stats(e):
    return dict(mean=float(np.mean(e)), p50=float(np.percentile(e, 50)),
                p90=float(np.percentile(e, 90)), p95=float(np.percentile(e, 95)),
                p99=float(np.percentile(e, 99)), max=float(np.max(e)))


def go_ok(eF, eK, cfg: TrainCfg):
    sF, sK = stats(eF), stats(eK)
    return (sF["mean"] < cfg.go_mean and sK["mean"] < cfg.go_mean
            and sF["p99"] < cfg.go_p99 and sK["p99"] < cfg.go_p99)


def report_bins(log, eF, eK, amp, dist, tag, results):
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


def normalization(spec: GramSpec, tr_arr, device):
    c_mean = torch.tensor(tr_arr["chat"].mean(axis=0), device=device)
    c_std = torch.tensor(np.maximum(tr_arr["chat"].std(axis=0), 1e-6), device=device)
    D = gram_D_batch(spec, torch.tensor(tr_arr["chat"], dtype=torch.float64),
                     torch.tensor(tr_arr["u"], dtype=torch.float64)).numpy()
    D_scale = torch.tensor(np.maximum(D.std(axis=0), 1e-8).astype(np.float32),
                           device=device)
    return c_mean, c_std, D_scale


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(spec, hidden, depth, tr: Batch, va: Batch, norm, cfg: TrainCfg,
              log, device, epochs=None, tag=""):
    epochs = cfg.epochs if epochs is None else epochs
    model = GramWNet(spec, hidden, depth, *norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=20, min_lr=1e-5)
    floorW, floorF = make_floors(tr)
    n = len(tr)
    nparam = sum(p.numel() for p in model.parameters())
    log(f"\n--- {tag}GELU | Hidden {hidden} | Tiefe {depth} | Eingang {spec.n_in} | "
        f"Parameter {nparam:,} | MACs/Forward {arch_macs(spec, hidden, depth):,} ---")

    best, best_state, bad = np.inf, None, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, cfg.batch):
            idx = perm[s:s + cfg.batch]
            kidx = idx[torch.randperm(len(idx), device=device)[:cfg.k_sub]]
            loss, _, _, _ = sobolev_loss(model, tr, idx, kidx, floorW, floorF, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()

        vidx = torch.arange(min(len(va), 8192), device=device)
        vloss, lW, lF, lK = sobolev_loss(model, va, vidx, vidx[:cfg.k_sub],
                                         floorW, floorF, cfg)
        vloss = float(vloss.detach())
        sched.step(vloss)
        if vloss < best - 1e-7:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if ep % 25 == 0 or ep == 1:
            log(f"    Epoch {ep:4d}/{epochs} | Val {vloss:.4e} "
                f"(W {float(lW):.2e} / F {float(lF):.2e} / K {float(lK):.2e}) | "
                f"LR {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f} s")
        if bad >= cfg.patience:
            log(f"    Early stop bei Epoche {ep} (Val-Plateau).")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"    Fertig in {time.time()-t0:.1f} s | bester Val-Loss {best:.4e}")
    return model, best


# ---------------------------------------------------------------------------
# Gate c: autograd-K vs FD(F) in fp64 (ueber u)
# ---------------------------------------------------------------------------

def gate_c(model, b: Batch, log, n=5):
    spec = model.spec
    model.double()
    worst_fd, worst_sym = 0.0, 0.0
    # nicht-triviale Zustaende waehlen
    order = torch.argsort(b.u.norm(dim=-1), descending=True)
    for i in order[: n].tolist():
        c = b.c[i:i + 1].double()
        u = b.u[i:i + 1].double()
        _, _, K = net_terms(model, c, u)
        K = K[0]
        h = 1e-6
        J = torch.zeros(spec.ndof, spec.ndof, dtype=torch.float64, device=u.device)
        for j in range(spec.ndof):
            up = u.clone(); up[0, j] += h
            um = u.clone(); um[0, j] -= h
            _, Fp, _ = net_terms(model, c, up, need_hess=False)
            _, Fm, _ = net_terms(model, c, um, need_hess=False)
            J[:, j] = (Fp[0] - Fm[0]) / (2 * h)
        den = max(float(K.norm()), 1e-12)
        worst_fd = max(worst_fd, float((K - J).norm()) / den)
        worst_sym = max(worst_sym, float((K - K.T).norm()) / den)
    model.float()
    ok = worst_fd <= 1e-6 and worst_sym <= 1e-10
    log(f"\nGate c -- K vs FD(F): {worst_fd:.3e} (<= 1e-6), Symmetrie "
        f"{worst_sym:.3e}  -> {'GRUEN' if ok else 'ROT'}")
    return ok, worst_fd, worst_sym


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def git_hash(cwd):
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(cwd), stderr=subprocess.DEVNULL
                                       ).decode().strip()
    except Exception:
        return "no-git"


def build_mat(model: GramWNet, hidden, depth, va: Batch, meta: dict, res_json):
    """Gewichte + Normierung + Metadaten + fp64-Oracle-Testvektoren."""
    spec = model.spec
    model.eval()
    md = {
        "model_form": MODEL_FORM,
        "activation": ACTIVATION,
        "state_form": spec.state_form,
        "input_order": f"[c_hat({spec.ndof})_then_D({spec.m})]",
        "num_nodes": spec.n,
        "dim": spec.dim,
        "hidden": hidden,
        "depth": depth,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_norm_c_mean": model.c_mean.detach().cpu().numpy().astype(np.float64),
        "input_norm_c_std": model.c_std.detach().cpu().numpy().astype(np.float64),
        "input_norm_D_scale": model.D_scale.detach().cpu().numpy().astype(np.float64),
        "metrics_json": json.dumps(res_json),
    }
    md.update(meta)
    lins = [m for m in model.mlp if isinstance(m, nn.Linear)]
    md["num_linear_layers"] = len(lins)
    for i, lin in enumerate(lins, start=1):
        md[f"W{i}"] = lin.weight.detach().cpu().numpy().astype(np.float64)
        md[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float64)

    un = va.u.norm(dim=-1)
    cand = torch.nonzero(un > torch.quantile(un, 0.5), as_tuple=False).flatten()
    sel = cand[torch.linspace(0, len(cand) - 1, min(64, len(cand))).long()]
    c = va.c[sel].double()
    u = va.u[sel].double()
    model.double()
    W, F, K = net_terms(model, c, u)
    model.float()
    md["test_C"] = c.cpu().numpy()
    md["test_U"] = u.cpu().numpy()
    md["test_W"] = W.detach().cpu().numpy().reshape(-1, 1)
    md["test_F"] = F.detach().cpu().numpy()
    md["test_K"] = K.detach()[:, spec.kti, spec.ktj].cpu().numpy()
    md["test_precision"] = "float64"
    return md


# ---------------------------------------------------------------------------
# Referenz: StVenant-Energie als Funktion von D (fuer die Gates ohne Netz)
# ---------------------------------------------------------------------------

def stvenant_energy_from_D(Dfull, grads, detJw, lam, mu):
    """W = sum_gp 0.5*(lam tr(E)^2 + 2 mu E:E) * detJ*w,
    E_gp = 0.5 * Hx' Dfull Hx   (Hx: n x dim Formfunktionsgradienten).
    Belegt, dass die Elementenergie EXAKT eine Funktion von D ist."""
    W = 0.0
    for Hx, dv in zip(grads, detJw):
        E = 0.5 * Hx.T @ Dfull @ Hx
        tr = torch.trace(E)
        W = W + 0.5 * (lam * tr * tr + 2 * mu * (E * E).sum()) * dv
    return W


def D_to_full(spec: GramSpec, D):
    G = torch.zeros(spec.n, spec.n, dtype=D.dtype, device=D.device)
    G[spec.ti, spec.tj] = D
    G[spec.tj, spec.ti] = D
    return G
