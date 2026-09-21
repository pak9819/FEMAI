"""Referenzmathematik fuer das nichtlineare DLFE-quad4-Element mit Neo-Hooke.

Gegenstueck zu quad4_nl_ref.py (St.-Venant-Kirchhoff) fuer das Material
'NeoHookean1' aus sourcecode/material/material_elasticity.m ('NeoHooke' ist
dort ein Alias). Geometrie, Kanonisierung, Ko-Rotation und die exakte
Ableitungskette sind materialunabhaengig und werden aus quad4_nl_ref
uebernommen -- nur die Energie, die Zustands-Huelle und der Sampler sind neu.

Energie (ebener Verzerrungszustand, 2D-Deformationsgradient F, C = F'F):

    W = mu/2 * (tr C - 2) - mu * ln J + lambda/2 * (J - 1)^2
    S = mu * I - (mu - lambda*(J^2 - J)) * C^-1
    CC = lambda*(2J^2 - J) C^-1 (x) C^-1 + 2*(mu - lambda*(J^2 - J)) * I_{C^-1}

mit I_{C^-1,ijkl} = 1/2 (C^-1_ik C^-1_jl + C^-1_il C^-1_jk). Das ist exakt das,
was material_elasticity.m rechnet (Gate a' vergleicht gegen das MATLAB-Element).

Was sich gegenueber St.-Venant aendert:

  * W ist KEIN Polynom und hat ueber -mu*ln J eine Barriere (W -> inf fuer
    J -> 0). Die Huelle braucht deshalb eine untere Grenze fuer J.
  * ||E_green|| ist fuer grosse Deformationen ein schlechtes Huellenmass
    (quadratisch unter Zug, beschraenkt unter Druck). Stattdessen:
        ||ln U||_F <= H_MAX   (Hencky-Dehnung, symmetrisch in Zug/Druck)
        J_MIN <= J <= J_MAX
    jeweils an allen vier Gausspunkten.
  * Der Sampler zieht den affinen Anteil direkt als U = exp(H) (Hencky-
    Dehnung mit Modus-Mix Zug/Druck/Scherung) statt als Verschiebungsgradient.

Direkter Aufruf fuehrt die Gates a und b aus (ohne Netz); mit einer von
export_nh_oracle.m erzeugten Datei nh_oracle.mat zusaetzlich Gate a':

    python quad4_nh_ref.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

import quad4_nl_ref as base
from quad4_nl_ref import (GAUSS_POINTS, I2, TRIU_IJ, canonicalization_frame,  # noqa: F401
                          canonicalize_coords, chain_assemble, chain_context,
                          corot_state, detj_ratio, generate_distorted_quad,
                          in_geometry_hull, material_C, shape_quad4_ref,
                          triu_vec, _bmat, _fd_grad, _fd_jac, _rel,
                          _random_physical_element)

# ---------------------------------------------------------------------------
# Feste Trainingskonfiguration
# ---------------------------------------------------------------------------

NU_TRAIN = 0.3
CONDITION = "planeStrain"
MATERIAL = "NeoHookean1"
E_TRAIN = 1.0
D_TRAIN = 1.0

ENV_RATIO_MAX = base.ENV_RATIO_MAX
ENV_ANGLE_MIN = base.ENV_ANGLE_MIN
ENV_ANGLE_MAX = base.ENV_ANGLE_MAX
ENV_TAPER_MAX = base.ENV_TAPER_MAX

# Zustands-Huelle (Stufe 1, docs/DLFE_quad4_neohooke_plan.md): deckt im
# eingespannten Rechteck etwa +100 % Zug und -45 % Druck ab.
STATE_MEASURE = "hencky_J"
H_MAX = 1.0
J_MIN = 0.4
J_MAX = 1.8

AMP_LOG_MIN = 2e-3     # untere Amplitudengrenze (log-uniform, relativ zu H_MAX)
P_ZERO = 0.08          # Anteil exakt u = 0
NONAFFINE = 0.25       # nicht-affiner Anteil relativ zur affinen Amplitude
ROT_SHAPE_DEG = 45.0   # nur Sampling-Formung (Ko-Rotation entfernt den Mittelwert)

# Modus-Mix des affinen Anteils
MODE_P = dict(uniaxial=0.35, biaxial=0.15, shear=0.20, general=0.30)


def lame(E=E_TRAIN, nu=NU_TRAIN):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


# ---------------------------------------------------------------------------
# Material: Neo-Hooke (NeoHookean1), ebener Verzerrungszustand
# ---------------------------------------------------------------------------

def neohooke_point(F, lam, mu):
    """W, 2. PK-Spannung S (2x2) und Voigt-Tangente CC (3x3) an einem Punkt.

    Voigt-Reihenfolge wie tensor4tomatrix.m: [11, 22, 12]. Rueckgabe W = nan
    fuer J <= 0 (physikalisch unzulaessig).
    """
    J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
    if J <= 0.0:
        return np.nan, np.full((2, 2), np.nan), np.full((3, 3), np.nan)
    C = F.T @ F
    Ci = np.linalg.inv(C)
    W = 0.5 * mu * (np.trace(C) - 2.0) - mu * np.log(J) + 0.5 * lam * (J - 1.0) ** 2
    a = mu - lam * (J * J - J)
    S = mu * I2 - a * Ci
    b = lam * (2.0 * J * J - J)
    idx = [(0, 0), (1, 1), (0, 1)]
    CC = np.empty((3, 3))
    for p, (i, j) in enumerate(idx):
        for q, (k, l) in enumerate(idx):
            CC[p, q] = (b * Ci[i, j] * Ci[k, l]
                        + a * (Ci[i, k] * Ci[j, l] + Ci[i, l] * Ci[j, k]))
    return W, S, CC


def energy_stiffness_force_ref(coords, disp, E=E_TRAIN, nu=NU_TRAIN, d=D_TRAIN):
    """Formaenderungsenergie W, innere Kraefte Finte (8) und Tangente Ke (8x8).

    Total Lagrange mit 2x2-Gauss, identische Struktur wie element_quad4_nl.m:
        Finte = sum B' S dV,   Ke = sum (B' CC B + L S L' (x) I) dV
    """
    lam, mu = lame(E, nu)
    Ke = np.zeros((8, 8))
    Finte = np.zeros(8)
    W = 0.0
    ux = disp[:, 0]
    uy = disp[:, 1]
    for (r, s) in GAUSS_POINTS:
        hx, hy, detJ = shape_quad4_ref(coords, r, s)
        F = I2 + np.array([[ux @ hx, ux @ hy], [uy @ hx, uy @ hy]])
        Wp, S, CC = neohooke_point(F, lam, mu)
        Svec = np.array([S[0, 0], S[1, 1], S[0, 1]])
        dV = detJ * d
        B = _bmat(F, hx, hy)
        L = np.stack([hx, hy], axis=1)
        W += Wp * dV
        Finte += (B.T @ Svec) * dV
        Ke += (B.T @ CC @ B + np.kron(L @ S @ L.T, I2)) * dV
    return W, Finte, Ke


def k0_ref(coords, E=E_TRAIN, nu=NU_TRAIN, d=D_TRAIN):
    """Lineare Steifigkeit (Hooke) -- Neo-Hooke linearisiert exakt dazu."""
    return base.k0_ref(coords, material_C(E, nu, CONDITION), d)


# ---------------------------------------------------------------------------
# Zustands-Huelle und Sampler
# ---------------------------------------------------------------------------

def hull_measures(coords, disp):
    """max ||ln U||_F, min J, max J und max |Rotation| [Grad] ueber die GP."""
    ux = disp[:, 0]
    uy = disp[:, 1]
    hmax, jmin, jmax, rmax = 0.0, np.inf, -np.inf, 0.0
    for (r, s) in GAUSS_POINTS:
        hx, hy, _ = shape_quad4_ref(coords, r, s)
        F = I2 + np.array([[ux @ hx, ux @ hy], [uy @ hx, uy @ hy]])
        J = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        jmin = min(jmin, J)
        jmax = max(jmax, J)
        if J <= 0.0:
            return np.inf, jmin, jmax, np.inf
        lam2 = np.linalg.eigvalsh(F.T @ F)
        hmax = max(hmax, float(np.linalg.norm(0.5 * np.log(lam2))))
        rmax = max(rmax, abs(np.degrees(np.arctan2(F[1, 0] - F[0, 1],
                                                   F[0, 0] + F[1, 1]))))
    return hmax, jmin, jmax, rmax


def in_state_hull(coords, disp):
    h, jmin, jmax, _ = hull_measures(coords, disp)
    return h <= H_MAX and jmin >= J_MIN and jmax <= J_MAX


def _affine_U(rng):
    """Rechter Streckungstensor U = exp(H) mit Modus-Mix und log-uniformer Amplitude."""
    mode = rng.choice(list(MODE_P), p=list(MODE_P.values()))
    sgn = rng.choice([-1.0, 1.0])
    if mode == "uniaxial":
        h = np.array([sgn, -sgn * rng.uniform(0.0, 0.6)])
    elif mode == "biaxial":
        h = np.array([sgn, sgn * rng.uniform(0.3, 1.0)])
    elif mode == "shear":
        h = np.array([1.0, -1.0])
    else:
        h = rng.uniform(-1.0, 1.0, 2)
    h = h / max(np.linalg.norm(h), 1e-12)
    amp = H_MAX * np.exp(rng.uniform(np.log(AMP_LOG_MIN), 0.0))
    phi = rng.uniform(0.0, np.pi)
    Q = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    U = Q @ np.diag(np.exp(amp * h)) @ Q.T
    return U, amp


def sample_state(coords_c, rng=np.random):
    """Verschiebungszustand in der Neo-Hooke-Huelle (kanonisch, transl.-projiziert).

    Zustaende ausserhalb der Huelle werden VERWORFEN, nicht skaliert -- sonst
    sammeln sich die Samples an der Huellengrenze.
    """
    if rng.rand() < P_ZERO:
        return np.zeros((4, 2))
    for _ in range(50):
        U, amp = _affine_U(rng)
        th = np.radians(rng.uniform(-ROT_SHAPE_DEG, ROT_SHAPE_DEG))
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Fa = R @ U
        u = coords_c @ (Fa - I2).T
        u = u + rng.normal(0.0, NONAFFINE * amp / np.sqrt(2.0), (4, 2))
        u -= u.mean(axis=0)
        if in_state_hull(coords_c, u):
            return u
    return np.zeros((4, 2))


def amplitude(coords, disp):
    """Skalares Amplitudenmass fuer Auswertungs-Bins: max ||ln U||."""
    return hull_measures(coords, disp)[0]


# ---------------------------------------------------------------------------
# Kette mit analytischer Neo-Hooke-Energie (Gate b)
# ---------------------------------------------------------------------------

def assemble_physical_analytic(coord_e, Ue, Emod=1.0, d=1.0):
    ctx = chain_context(coord_e, Ue)
    _, p, H = energy_stiffness_force_ref(ctx["coords_c"], ctx["z"].reshape(4, 2))
    return chain_assemble(ctx, p, H, Emod, d)


def energy_physical_analytic(coord_e, Ue, Emod=1.0, d=1.0):
    ctx = chain_context(coord_e, Ue)
    W, _, _ = energy_stiffness_force_ref(ctx["coords_c"], ctx["z"].reshape(4, 2))
    return Emod * d * ctx["Lc"] ** 2 * W


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def _state_for(coords_c, rng):
    """Nicht-trivialer Zustand fuer die Gates (u = 0 prueft nichts)."""
    for _ in range(100):
        u = sample_state(coords_c, rng)
        if np.linalg.norm(u) > 1e-3:
            return u
    raise RuntimeError("Kein nicht-trivialer Zustand gefunden.")


def gate_a(n_samples=8, seed=7, verbose=True, strict=True):
    """F = dW/dz, K = dF/dz, K(z=0) = K_lin (Hooke), Skalierungsgesetz."""
    rng = np.random.RandomState(seed)
    worst = dict(dWdz=0.0, dFdz=0.0, k0=0.0, scale=0.0)
    for _ in range(n_samples):
        coords_c = canonicalize_coords(generate_distorted_quad(rng))
        u = corot_state(coords_c, _state_for(coords_c, rng))
        z = u.reshape(-1)
        W, F, K = energy_stiffness_force_ref(coords_c, u)
        fW = lambda zz: energy_stiffness_force_ref(coords_c, zz.reshape(4, 2))[0]
        fF = lambda zz: energy_stiffness_force_ref(coords_c, zz.reshape(4, 2))[1]
        worst["dWdz"] = max(worst["dWdz"], _rel(_fd_grad(fW, z), F))
        worst["dFdz"] = max(worst["dFdz"], _rel(_fd_jac(fF, z), K))
        worst["k0"] = max(worst["k0"], _rel(
            k0_ref(coords_c), energy_stiffness_force_ref(coords_c, np.zeros((4, 2)))[2]))
        Emod, dth, Lc_t = 1000.0, 2.0, 3.7
        Wp, Fp, Kp = energy_stiffness_force_ref(coords_c * Lc_t, u * Lc_t, Emod, NU_TRAIN, dth)
        e = max(abs(Wp - Emod * dth * Lc_t ** 2 * W) / max(abs(Wp), 1e-14),
                _rel(Fp, Emod * dth * Lc_t * F), _rel(Kp, Emod * dth * K))
        worst["scale"] = max(worst["scale"], e)
    ok = (worst["dWdz"] <= 1e-7 and worst["dFdz"] <= 1e-7
          and worst["k0"] <= 1e-10 and worst["scale"] <= 1e-12)
    if verbose:
        print("Gate a (Neo-Hooke) -- Referenz-Targets (Schwellen 1e-7 / 1e-7 / 1e-10 / 1e-12)")
        print(f"    F  vs FD(W)        : {worst['dWdz']:.3e}")
        print(f"    K  vs FD(F)        : {worst['dFdz']:.3e}")
        print(f"    K(z=0) vs K_lin    : {worst['k0']:.3e}")
        print(f"    Skalierungsgesetz  : {worst['scale']:.3e}")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    if strict and not ok:
        raise AssertionError(f"Gate a (Neo-Hooke) fehlgeschlagen: {worst}")
    return ok, worst


def gate_b(n_samples=10, seed=11, verbose=True, strict=True):
    """Kette mit analytischer Neo-Hooke-Energie vs. FD und vs. direkte Referenz."""
    rng = np.random.RandomState(seed)
    worst = dict(fd_F=0.0, fd_K=0.0, oracle_F=0.0, oracle_K=0.0)
    for _ in range(n_samples):
        coord_e = _random_physical_element(rng)
        Emod = rng.uniform(1.0, 1000.0)
        dth = rng.uniform(0.1, 2.0)
        coords_c, Lc, Rc = canonicalization_frame(coord_e)
        u_c = _state_for(coords_c, rng)
        th = rng.uniform(-np.pi / 3, np.pi / 3)
        Rth = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        u_c = (coords_c + u_c) @ Rth.T - coords_c
        U_nodes = (u_c * Lc) @ Rc
        Ue = U_nodes.reshape(-1)

        F_chain, K_chain = assemble_physical_analytic(coord_e, Ue, Emod, dth)
        fW = lambda uu: energy_physical_analytic(coord_e, uu, Emod, dth)
        fF = lambda uu: assemble_physical_analytic(coord_e, uu, Emod, dth)[0]
        h = 1e-6 * Lc
        worst["fd_F"] = max(worst["fd_F"], _rel(_fd_grad(fW, Ue, h), F_chain))
        worst["fd_K"] = max(worst["fd_K"], _rel(_fd_jac(fF, Ue, h), K_chain))

        _, F_ref, K_ref = energy_stiffness_force_ref(coord_e, U_nodes, Emod, NU_TRAIN, dth)
        worst["oracle_F"] = max(worst["oracle_F"], _rel(F_chain, F_ref))
        worst["oracle_K"] = max(worst["oracle_K"], _rel(K_chain, K_ref))
    ok = (worst["fd_F"] <= 1e-7 and worst["fd_K"] <= 1e-7
          and worst["oracle_F"] <= 1e-10 and worst["oracle_K"] <= 1e-10)
    if verbose:
        print("\nGate b (Neo-Hooke) -- Kette mit analytischem W (Schwellen 1e-7 / 1e-10)")
        print(f"    Finte vs FD(W_phys)      : {worst['fd_F']:.3e}")
        print(f"    Ke    vs FD(Finte)       : {worst['fd_K']:.3e}")
        print(f"    Finte vs direkte Referenz: {worst['oracle_F']:.3e}")
        print(f"    Ke    vs direkte Referenz: {worst['oracle_K']:.3e}")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    if strict and not ok:
        raise AssertionError(f"Gate b (Neo-Hooke) fehlgeschlagen: {worst}")
    return ok, worst


ORACLE_FILE = Path(__file__).parent / "nh_oracle.mat"


def gate_a_prime(verbose=True, strict=True):
    """Python-Referenz vs. MATLAB-Element (element_quad4_nl, 'NeoHooke').

    Die Oracle-Datei schreibt export_nh_oracle.m. Fehlt sie, wird das Gate
    uebersprungen (Rueckgabe None) -- das Training verlangt sie trotzdem.
    """
    if not ORACLE_FILE.exists():
        if verbose:
            print("\nGate a' (Neo-Hooke) -- nh_oracle.mat fehlt, zuerst export_nh_oracle.m "
                  "in MATLAB ausfuehren.")
        return None, {}
    md = scipy.io.loadmat(str(ORACLE_FILE))
    coords, Ue = md["coords"], md["Ue"]
    Fm, Km, mat = md["Finte"], md["Ke"], md["mat"]
    worst = dict(F=0.0, K=0.0)
    for i in range(Ue.shape[1]):
        E, nu, d = mat[:, i][:3]
        _, F, K = energy_stiffness_force_ref(coords[:, :, i], Ue[:, i].reshape(4, 2), E, nu, d)
        worst["F"] = max(worst["F"], _rel(F, Fm[:, i]))
        worst["K"] = max(worst["K"], _rel(K, Km[:, :, i]))
    ok = worst["F"] <= 1e-10 and worst["K"] <= 1e-10
    if verbose:
        print(f"\nGate a' (Neo-Hooke) -- Python vs. MATLAB-Element ({Ue.shape[1]} Faelle, "
              f"Schwelle 1e-10)")
        print(f"    Finte: {worst['F']:.3e} | Ke: {worst['K']:.3e}   -> "
              f"{'GRUEN' if ok else 'ROT'}")
    if strict and not ok:
        raise AssertionError(f"Gate a' (Neo-Hooke) fehlgeschlagen: {worst}")
    return ok, worst


def export_oracle_states(path, n=40, seed=5):
    """Zufaellige physische Elemente + Zustaende fuer export_nh_oracle.m."""
    rng = np.random.RandomState(seed)
    coords = np.zeros((4, 2, n))
    Ue = np.zeros((8, n))
    mat = np.zeros((4, n))
    for i in range(n):
        coord_e = _random_physical_element(rng)
        coords_c, Lc, Rc = canonicalization_frame(coord_e)
        u_c = _state_for(coords_c, rng)
        coords[:, :, i] = coord_e
        Ue[:, i] = ((u_c * Lc) @ Rc).reshape(-1)
        mat[:, i] = [rng.uniform(1.0, 1000.0), NU_TRAIN, rng.uniform(0.1, 2.0), 0.0]
    scipy.io.savemat(str(path), dict(coords=coords, Ue=Ue, mat=mat))


def run_gates(verbose=True, strict=True, require_oracle=True):
    ok_a, wa = gate_a(verbose=verbose, strict=strict)
    ok_b, wb = gate_b(verbose=verbose, strict=strict)
    ok_ap, wap = gate_a_prime(verbose=verbose, strict=strict)
    if ok_ap is None:
        ok_ap = not require_oracle
    return ok_a and ok_b and ok_ap, {"a": wa, "b": wb, "a'": wap}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--oracle-states":
        export_oracle_states(Path(__file__).parent / "nh_oracle_states.mat")
        print("nh_oracle_states.mat geschrieben -- jetzt export_nh_oracle.m in MATLAB.")
        raise SystemExit(0)
    print("=== Gates a, b, a' (Neo-Hooke, ohne Netz) ===\n")
    ok, _ = run_gates(strict=False, require_oracle=False)
    print(f"\nGesamt: {'GRUEN' if ok else 'ROT'}")
    raise SystemExit(0 if ok else 1)
