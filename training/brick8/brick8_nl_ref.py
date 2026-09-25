"""Referenzmathematik fuer das nichtlineare DLFE-brick8-Element (StVenant).

Einzige Quelle der Wahrheit fuer:
  * die analytische Referenz W, F = dW/du, K = d2W/du2 (Total Lagrange,
    2x2x2 Gauss, St.-Venant-Kirchhoff 3D) -- batched in torch (fp64),
    Ableitungen per torch.func (exakt, keine FD),
  * die lineare Steifigkeit K0 (explizite B'CB-Formel, unabhaengig),
  * die 3D-Kanonisierung (identisch dlfe_canonical_frame.m),
  * Geometrie- und Zustands-Sampling in der Huelle.

Gates (ohne Netz):
  a   F vs FD(W), K vs FD(F), K(u=0) == explizites K0, Skalierungsgesetz
      W(s X, s u) = s^3 W
  b   Metrik-Kette: Energie als Funktion der Knoten-Gram D == W, modale
      Form: Rotationsinvarianz, D(Starrkoerper) = 0, D_lin == 2 E(Mitte)
  a'  Python-Targets vs. MATLAB element_brick8_nl (Oracle-Datei)

    python brick8_nl_ref.py                  # Gates a/b (+ a' falls Oracle da)
    python brick8_nl_ref.py --oracle-states  # Zustaende fuer Gate a' schreiben
    # MATLAB: export_brick8_oracle           # -> brick8_oracle.mat

Konventionen (identisch MATLAB): coords (8,3), disp (8,3); flach (24,)
= [u1x,u1y,u1z,u2x,...]; Knotenreihenfolge wie shape_brick8.m.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.func import grad, jacfwd, vmap

sys.path.insert(0, str(Path(__file__).parent / ".." / "common"))
import dlfe_gram as dg  # noqa: E402

DT = torch.float64

NU_TRAIN = 0.3
CONDITION = "3D"
MATERIAL = "StVenant"
E_TRAIN = 1.0

ENV_RATIO_MAX = 4.5
ENV_ANGLE_MIN = 20.0
ENV_ANGLE_MAX = 160.0
ENV_TAPER_MAX = 2.5

E_MAX = 0.2
AMP_LOG_MIN = 2e-3
P_ZERO = 0.12
P_HOURGLASS = 0.5    # Anteil Zustaende mit ausgepraegten Hourglass-Moden

LAM = E_TRAIN * NU_TRAIN / ((1 + NU_TRAIN) * (1 - 2 * NU_TRAIN))
MU = E_TRAIN / (2 * (1 + NU_TRAIN))

NAT = np.array(dg.NODE_NAT[(8, 3)], dtype=np.float64)          # (8,3)
_a = 1.0 / np.sqrt(3.0)
GP = np.array([[-_a, -_a, -_a], [_a, -_a, -_a], [_a, _a, -_a], [-_a, _a, -_a],
               [-_a, -_a, _a], [_a, -_a, _a], [_a, _a, _a], [-_a, _a, _a]])
FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]

PHI_HG = dg.mode_matrix(8, 3)[:, 3:]                           # (8,4) rs, st, rt, rst
SPEC_NODE = dg.GramSpec(8, 3, modal=False)
SPEC_MODAL = dg.GramSpec(8, 3, modal=True)


# ---------------------------------------------------------------------------
# Formfunktionen
# ---------------------------------------------------------------------------

def dh_dr(pts):
    """(P,3) natuerliche Punkte -> (P,8,3) Ableitungen nach r,s,t."""
    r, s, t = pts[:, 0:1], pts[:, 1:2], pts[:, 2:3]
    ar = 1 + NAT[None, :, 0] * r
    as_ = 1 + NAT[None, :, 1] * s
    at = 1 + NAT[None, :, 2] * t
    return np.stack([NAT[None, :, 0] * as_ * at, ar * NAT[None, :, 1] * at,
                     ar * as_ * NAT[None, :, 2]], axis=-1) / 8.0


DHDR_GP = torch.tensor(dh_dr(GP))                                  # (8gp,8,3)
DHDR_CORNER = dh_dr(NAT)                                           # (8,8,3)


def shape_grads(X):
    """X (8,3) torch -> dh/dX an den GP (8gp,8,3), detJ (8gp)."""
    J = torch.einsum("ai,gak->gik", X, DHDR_GP.to(X))              # dx_i/dr_k
    detJ = torch.linalg.det(J)
    dh = torch.linalg.solve(J.transpose(1, 2), DHDR_GP.to(X).transpose(1, 2)).transpose(1, 2)
    return dh, detJ


def energy(X, u):
    """StVenant-Elementenergie (E = 1), X, u (8,3) oder flach (24,)."""
    X = X.reshape(8, 3)
    u = u.reshape(8, 3)
    dh, detJ = shape_grads(X)
    Fd = torch.eye(3, dtype=X.dtype) + torch.einsum("ak,gaj->gkj", u, dh)
    E = 0.5 * (Fd.transpose(1, 2) @ Fd - torch.eye(3, dtype=X.dtype))
    tr = E.diagonal(dim1=1, dim2=2).sum(-1)
    psi = 0.5 * LAM * tr ** 2 + MU * (E * E).sum((-2, -1))
    return (psi * detJ).sum()


_gradW = grad(energy, argnums=1)
_hessW = jacfwd(_gradW, argnums=1)


def targets(X, u):
    """Batched: X (B,24), u (B,24) -> W (B,), F (B,24), K (B,24,24)."""
    W = vmap(energy)(X, u)
    F = vmap(_gradW)(X, u)
    K = vmap(_hessW)(X, u)
    return W, F.reshape(-1, 24), K.reshape(-1, 24, 24)


def k0_explicit(X):
    """Lineare Steifigkeit (explizit B'CB, unabhaengig von der Energie)."""
    X = torch.as_tensor(X, dtype=DT).reshape(8, 3)
    dh, detJ = shape_grads(X)
    C = torch.zeros(6, 6, dtype=DT)
    C[:3, :3] = LAM
    C[:3, :3] += 2 * MU * torch.eye(3, dtype=DT)
    C[3:, 3:] = MU * torch.eye(3, dtype=DT)
    K = torch.zeros(24, 24, dtype=DT)
    for g in range(8):
        hx, hy, hz = dh[g, :, 0], dh[g, :, 1], dh[g, :, 2]
        B = torch.zeros(6, 24, dtype=DT)
        B[0, 0::3] = hx
        B[1, 1::3] = hy
        B[2, 2::3] = hz
        B[3, 0::3] = hy; B[3, 1::3] = hx
        B[4, 1::3] = hz; B[4, 2::3] = hy
        B[5, 0::3] = hz; B[5, 2::3] = hx
        K += B.T @ C @ B * detJ[g]
    return K


def green_at_points(X, u, pts_dhdr):
    """max ||E_green||_F an gegebenen Punkten (numpy)."""
    Xt = torch.as_tensor(X, dtype=DT).reshape(8, 3)
    ut = torch.as_tensor(u, dtype=DT).reshape(8, 3)
    D = torch.as_tensor(pts_dhdr, dtype=DT)
    J = torch.einsum("ai,gak->gik", Xt, D)
    dh = torch.linalg.solve(J.transpose(1, 2), D.transpose(1, 2)).transpose(1, 2)
    Fd = torch.eye(3, dtype=DT) + torch.einsum("ak,gaj->gkj", ut, dh)
    E = 0.5 * (Fd.transpose(1, 2) @ Fd - torch.eye(3, dtype=DT))
    return float(torch.linalg.matrix_norm(E).max())


def hull_E(X, u):
    return green_at_points(X, u, DHDR_GP.numpy())


# ---------------------------------------------------------------------------
# Kanonisierung (identisch dlfe_canonical_frame.m)
# ---------------------------------------------------------------------------

def canonical_frame(coords):
    c = coords.mean(axis=0)
    xc = coords - c
    Lc = float(np.mean(np.linalg.norm(xc, axis=1)))
    xb = xc / Lc
    e1 = xb[1] - xb[0]
    e1 /= np.linalg.norm(e1)
    a = xb[3] - xb[0]
    e2 = a - (a @ e1) * e1
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    Rc = np.stack([e1, e2, e3])
    return xb @ Rc.T, Lc, Rc


# ---------------------------------------------------------------------------
# Geometrie-Huelle und Sampling
# ---------------------------------------------------------------------------

def detJ_points(coords, dhdr):
    J = np.einsum("ai,gak->gik", coords, dhdr)
    return np.linalg.det(J)


def face_angles(coords):
    ang = []
    for f in FACES:
        for i in range(4):
            p = coords[f[i]]
            u = coords[f[(i - 1) % 4]] - p
            v = coords[f[(i + 1) % 4]] - p
            cang = u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15)
            ang.append(np.degrees(np.arccos(np.clip(cang, -1, 1))))
    return np.array(ang)


def detj_ratio(coords):
    dJ = detJ_points(coords, DHDR_GP.numpy())
    return float(dJ.max() / dJ.min())


def in_geometry_hull(coords):
    dJg = detJ_points(coords, DHDR_GP.numpy())
    dJc = detJ_points(coords, DHDR_CORNER)
    if dJg.min() <= 1e-9 or dJc.min() <= 1e-9:
        return False
    if dJg.max() / dJg.min() > ENV_RATIO_MAX:
        return False
    ang = face_angles(coords)
    return not (ang.min() < ENV_ANGLE_MIN or ang.max() > ENV_ANGLE_MAX)


def generate_distorted_hex(rng=np.random):
    """Wohlgestellter, verzerrter Hexaeder (Seitenverhaeltnis, Taper,
    Scherung, Knoten-Jitter) in der Geometrie-Huelle."""
    for _ in range(300):
        ext = np.exp(rng.uniform(np.log(0.4), np.log(2.5), 3))
        X = NAT * 0.5 * ext
        # Taper: obere Flaeche (t = +1) skaliert, in x und/oder y
        tap = np.exp(rng.uniform(-1, 1, 2) * np.log(ENV_TAPER_MAX) * (rng.rand(2) < 0.5))
        top = NAT[:, 2] > 0
        X[top, 0] *= tap[0]
        X[top, 1] *= tap[1]
        X = X[:, rng.permutation(3)] if rng.rand() < 0.5 else X
        G = np.eye(3) + rng.uniform(-0.35, 0.35, (3, 3)) * (1 - np.eye(3))
        X = X @ G.T + rng.uniform(-0.08, 0.08, (8, 3)) * ext.mean()
        if np.linalg.det(np.einsum("ai,ak->ik", X, DHDR_CORNER.mean(0))) < 0:
            continue
        if in_geometry_hull(X):
            return X
    return None


def sample_state(Xc, rng=np.random):
    """Verschiebungszustand in der Huelle ||E_green|| <= E_MAX (kanonisch,
    zentriert). Affin + nicht-affin, log-uniforme Amplitude. Mit
    Wahrscheinlichkeit P_HOURGLASS ist der nicht-affine Anteil aus den
    Hourglass-Moden aufgebaut (Biegung/Verwindung, v2), sonst Knotenrauschen."""
    if rng.rand() < P_ZERO:
        return np.zeros((8, 3))
    for _ in range(20):
        G = rng.normal(0, 0.15, (3, 3))
        if rng.rand() < P_HOURGLASS:
            # Biege-/Verwindungszustaende: Hourglass-Moden (rs, st, rt, rst)
            # mit Amplitude in der Groessenordnung des affinen Anteils
            Hm = rng.normal(0, 0.12, (4, 3)) * (rng.rand(4, 1) < 0.6)
            u = Xc @ G.T + 8.0 * PHI_HG @ Hm + rng.normal(0, 0.02, (8, 3))
        else:
            u = Xc @ G.T + rng.normal(0, 0.05, (8, 3))
        u -= u.mean(axis=0)
        u *= np.exp(rng.uniform(np.log(AMP_LOG_MIN), 0.0))
        mE = hull_E(Xc, u)
        if mE <= E_MAX:
            return u
        u2 = u * 0.9 * E_MAX / mE
        if hull_E(Xc, u2) <= E_MAX:
            return u2
    return np.zeros((8, 3))


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def rotmat(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def _random_pairs(n, seed):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        X = generate_distorted_hex(rng)
        if X is None:
            continue
        Xc, _, _ = canonical_frame(X)
        u = sample_state(Xc, rng)
        if np.linalg.norm(u) < 1e-4:
            continue
        out.append((Xc, u))
    return out, rng


def gate_a(verbose=True, n=10):
    pairs, rng = _random_pairs(n, 11)
    wF = wK = wK0 = wS = 0.0
    h = 1e-6
    for Xc, u in pairs:
        X = torch.tensor(Xc.reshape(-1))
        ut = torch.tensor(u.reshape(-1))
        W, F, K = targets(X[None], ut[None])
        F, K = F[0], K[0]
        Ffd = torch.zeros(24, dtype=DT)
        Kfd = torch.zeros(24, 24, dtype=DT)
        for j in range(24):
            e = torch.zeros(24, dtype=DT); e[j] = h
            Ffd[j] = (energy(X, ut + e) - energy(X, ut - e)) / (2 * h)
            Kfd[:, j] = (_gradW(X, ut + e) - _gradW(X, ut - e)) / (2 * h)
        wF = max(wF, float((F - Ffd).norm() / F.norm()))
        wK = max(wK, float((K - Kfd).norm() / K.norm()))
        K0 = _hessW(X, torch.zeros(24, dtype=DT))
        wK0 = max(wK0, float((K0 - k0_explicit(X)).norm() / K0.norm()))
        s = 2.7
        wS = max(wS, abs(float(energy(s * X, s * ut)) / (s ** 3 * float(W[0])) - 1))
    ok = wF < 1e-8 and wK < 1e-8 and wK0 < 1e-12 and wS < 1e-12
    if verbose:
        print("Gate a -- Referenz-Targets brick8")
        print(f"    F  vs FD(W)          : {wF:.3e}  (< 1e-8)")
        print(f"    K  vs FD(F)          : {wK:.3e}  (< 1e-8)")
        print(f"    K(0) vs explizit K0  : {wK0:.3e}  (< 1e-12)")
        print(f"    Skalierung s^3       : {wS:.3e}  (< 1e-12)")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    return ok


def gate_b(verbose=True, n=20):
    pairs, rng = _random_pairs(n, 12)
    w1 = w2 = w3 = w4 = w5 = 0.0
    for Xc, u in pairs:
        c = torch.tensor(Xc.reshape(-1))
        ut = torch.tensor(u.reshape(-1))
        dh, detJ = shape_grads(c.reshape(8, 3))
        grads = [dh[g] for g in range(8)]
        dvs = [detJ[g] for g in range(8)]

        def Wd(uu):
            Df = dg.D_to_full(SPEC_NODE, dg.gram_D(SPEC_NODE, c, uu))
            return dg.stvenant_energy_from_D(Df, grads, dvs, LAM, MU)
        W0 = float(energy(c, ut))
        w1 = max(w1, abs(float(Wd(ut)) - W0) / abs(W0))
        F0 = _gradW(c, ut)
        w2 = max(w2, float((grad(Wd)(ut) - F0).norm() / F0.norm()))
        R = rotmat(rng)
        t = rng.uniform(-3, 3, 3)
        u_rot = torch.tensor(((Xc + u) @ R.T + t - Xc).reshape(-1))
        u_rig = torch.tensor((Xc @ R.T + t - Xc).reshape(-1))
        Dm = dg.gram_D(SPEC_MODAL, c, ut)
        w3 = max(w3, float((dg.gram_D(SPEC_MODAL, c, u_rot) - Dm).norm() / Dm.norm()))
        w4 = max(w4, float(dg.gram_D(SPEC_MODAL, c, u_rig).abs().max()))
        # linearer Block (triu 3x3 -> Indizes 0..5) == F0'F0 - I im Zentrum
        dh0 = torch.tensor(dh_dr(np.zeros((1, 3)))[0])
        J0 = c.reshape(8, 3).T @ dh0
        g0 = torch.linalg.solve(J0.T, dh0.T).T
        Fc = torch.eye(3, dtype=DT) + ut.reshape(8, 3).T @ g0
        Cc = Fc.T @ Fc - torch.eye(3, dtype=DT)
        ii = [a for a, _ in SPEC_MODAL.triu[:6]]
        jj = [b for _, b in SPEC_MODAL.triu[:6]]
        w5 = max(w5, float((Dm[:6] - Cc[ii, jj]).abs().max() / Cc.abs().max()))
    ok = w1 < 1e-12 and w2 < 1e-10 and w3 < 1e-10 and w4 < 1e-13 and w5 < 1e-12
    if verbose:
        print("Gate b -- Metrik-Kette brick8 (ohne Netz)")
        print(f"    W(D_Knoten) vs W       : {w1:.3e}  (< 1e-12)")
        print(f"    F ueber D vs F         : {w2:.3e}  (< 1e-10)")
        print(f"    D_modal rot.-invariant : {w3:.3e}  (< 1e-10)")
        print(f"    D_modal(Starrk.) = 0   : {w4:.3e}  (< 1e-13)")
        print(f"    D_lin == 2 E(Mitte)    : {w5:.3e}  (< 1e-12)")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    return ok


ORACLE_STATES = Path(__file__).parent / "brick8_oracle_states.mat"
ORACLE = Path(__file__).parent / "brick8_oracle.mat"


def write_oracle_states(n=40):
    """Physische (frei gedrehte/skalierte/verschobene) Elemente + Zustaende."""
    pairs, rng = _random_pairs(n, 13)
    coords = np.zeros((8, 3, n))
    Ue = np.zeros((24, n))
    for i, (Xc, u) in enumerate(pairs):
        R = rotmat(rng)
        s = rng.uniform(0.3, 4.0)
        t = rng.uniform(-5, 5, 3)
        X = s * Xc @ R.T + t
        coords[:, :, i] = X
        Ue[:, i] = (s * u @ R.T).reshape(-1)
    scipy.io.savemat(str(ORACLE_STATES), {"coords": coords, "Ue": Ue})
    print(f"geschrieben: {ORACLE_STATES} ({n} Zustaende)")


def gate_a_prime(verbose=True, E=1000.0):
    if not ORACLE.exists():
        if verbose:
            print("Gate a' -- brick8_oracle.mat fehlt (MATLAB export_brick8_oracle).")
        return None
    md = scipy.io.loadmat(str(ORACLE))
    coords, Ue = md["coords"], md["Ue"]
    Fm, Km = md["Finte"], md["Ke"]
    wF = wK = 0.0
    for i in range(Ue.shape[1]):
        X = torch.tensor(coords[:, :, i].reshape(-1))
        u = torch.tensor(Ue[:, i])
        _, F, K = targets(X[None], u[None])
        F = E * F[0].numpy()
        K = E * K[0].numpy()
        wF = max(wF, np.linalg.norm(F - Fm[:, i]) / np.linalg.norm(Fm[:, i]))
        wK = max(wK, np.linalg.norm(K - Km[:, :, i]) / np.linalg.norm(Km[:, :, i]))
    ok = wF < 1e-10 and wK < 1e-10
    if verbose:
        print(f"Gate a' -- Python vs MATLAB element_brick8_nl ({Ue.shape[1]} Zustaende)")
        print(f"    Finte : {wF:.3e}  (< 1e-10)")
        print(f"    Ke    : {wK:.3e}  (< 1e-10)")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    return ok


def run_gates(verbose=True, require_oracle=False):
    ok = gate_a(verbose) and gate_b(verbose)
    ap = gate_a_prime(verbose)
    if ap is None:
        return ok and not require_oracle
    return ok and ap


if __name__ == "__main__":
    if "--oracle-states" in sys.argv:
        write_oracle_states()
    else:
        ok = run_gates(verbose=True)
        raise SystemExit(0 if ok else 1)
