"""Referenzmathematik fuer das nichtlineare DLFE-quad4-Element (Residual-Energie).

Dieses Modul ist die EINZIGE Quelle der Wahrheit fuer:

  * die analytische Referenz (Energie W, innere Kraefte F, Tangente K) --
    Port von sourcecode/elements/quad4/element_quad4_nl.m,
  * die lineare Steifigkeit K0 der kanonischen Geometrie (Port von
    element_quad4_lin.m, nur der Ke-Teil),
  * Geometrie-/Zustands-Sampling (uebernommen aus train_quad4_nl_K_network.py),
  * die Kanonisierungs- + Ko-Rotations-KETTE inklusive exakter erster und
    zweiter Ableitungen (dieselbe Kette rechnet das MATLAB-Element).

Das Trainingsskript (train_quad4_nl_W_network.py) und die Gates importieren
von hier. Direkter Aufruf fuehrt die Gates a und b aus:

    python quad4_nl_ref.py

Gate a: Selbstverifikation der Targets (F = dW/dz, K = dF/dz, K0 = K(z=0),
        Skalierungsgesetz).
Gate b: die komplette Kette mit ANALYTISCHEM W gegen zentrale Differenzen des
        Komposits Ue -> Finte/Ke, plus Oracle-Vergleich gegen das analytische
        Element in physikalischen Koordinaten.

Beide Gates laufen ohne Netz -- sie trennen Ketten-/Referenzfehler von
Netzfehlern, BEVOR ein Netz existiert.

Modell (Residual-Energie mit K0-Split):

    W(c,z)  = 0.5*z' K0(c) z + W_NL(c,z)
    F(c,z)  = K0(c) z + grad_z W_NL
    K(c,z)  = K0(c)   + hess_z W_NL

Konventionen (identisch MATLAB):
    coords : (4,2)  Knotenkoordinaten, Zeile i = [xi, yi]
    disp   : (4,2)  Knotenverschiebungen, Zeile i = [uix, uiy]
    flach  : (8,)   [u1x, u1y, u2x, u2y, ...]   (row-major reshape)
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Feste Trainingskonfiguration (Material ist NICHT frei -- fest eintrainiert)
# ---------------------------------------------------------------------------

NU_TRAIN = 0.3
CONDITION = "planeStrain"
MATERIAL = "StVenant"
E_TRAIN = 1.0
D_TRAIN = 1.0

# Geometrie-Huelle (unveraendert aus der K/F-Variante)
ENV_RATIO_MAX = 4.5
ENV_ANGLE_MIN = 20.0
ENV_ANGLE_MAX = 160.0
ENV_TAPER_MAX = 4.0

# Zustands-Huelle
E_MAX = 0.2      # max ||E_green||_F ueber die Gausspunkte
ROT_MAX = 45.0   # nur Sampling-Formung; zur Laufzeit unbeschraenkt (Ko-Rotation)

AMP_LOG_MIN = 2e-3   # untere Amplitudengrenze (log-uniform)
P_ZERO = 0.12        # Anteil exakt u = 0

_s3 = 1.0 / np.sqrt(3)
GAUSS_POINTS = [(-_s3, -_s3), (_s3, -_s3), (_s3, _s3), (-_s3, _s3)]

TRIU_IJ = [(i, j) for j in range(8) for i in range(j + 1)]   # 36, column-major
KE_N = len(TRIU_IJ)

I2 = np.eye(2)
I8 = np.eye(8)

# Translationsprojektor P = I - (tx tx' + ty ty')/4
_TX = np.array([1.0, 0.0] * 4)
_TY = np.array([0.0, 1.0] * 4)
P_TRANS = I8 - (np.outer(_TX, _TX) + np.outer(_TY, _TY)) / 4.0

# Sigma = kron(I4, [[0, 1], [-1, 0]])
SIGMA8 = np.kron(np.eye(4), np.array([[0.0, 1.0], [-1.0, 0.0]]))


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

def material_C(E=E_TRAIN, nu=NU_TRAIN, condition=CONDITION):
    """Materialmatrix in Voigt-Notation (Evec = [E11, E22, 2*E12])."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    if condition.lower() == "planestrain":
        c11 = lam + 2 * mu
        c12 = lam
    elif condition.lower() == "planestress":
        c11 = lam / (nu - 1) + 2 * mu + 2 * lam
        c12 = lam / (nu - 1) + 2 * lam
    else:
        raise ValueError(condition)
    return np.array([[c11, c12, 0.0], [c12, c11, 0.0], [0.0, 0.0, mu]])


C_MAT = material_C()


# ---------------------------------------------------------------------------
# Geometrie / Kinematik
# ---------------------------------------------------------------------------

def shape_quad4_ref(coords, r, s):
    """Formfunktionsableitungen nach x,y und Jacobideterminante."""
    x = coords[:, 0]
    y = coords[:, 1]
    h_r = 0.25 * np.array([-(1 - s), (1 - s), (1 + s), -(1 + s)])
    h_s = 0.25 * np.array([-(1 - r), -(1 + r), (1 + r), (1 - r)])
    x_r = x @ h_r
    x_s = x @ h_s
    y_r = y @ h_r
    y_s = y @ h_s
    detJ = x_r * y_s - y_r * x_s
    h_x = (y_s * h_r - y_r * h_s) / detJ
    h_y = (-x_s * h_r + x_r * h_s) / detJ
    return h_x, h_y, detJ


def compute_Lc(coords):
    centroid = coords.mean(axis=0)
    return float(np.mean(np.linalg.norm(coords - centroid, axis=1)))


def canonicalize_coords(coords):
    """Schwerpunkt abziehen, durch Lc teilen, Kante 1->2 auf +x drehen."""
    centroid = coords.mean(axis=0)
    Lc = compute_Lc(coords)
    cn = (coords - centroid) / Lc
    e12 = cn[1] - cn[0]
    phi = np.arctan2(e12[1], e12[0])
    c, s = np.cos(phi), np.sin(phi)
    Rc = np.array([[c, s], [-s, c]])
    return cn @ Rc.T


def canonicalization_frame(coords):
    """Wie canonicalize_coords, gibt zusaetzlich Lc und Rc zurueck."""
    centroid = coords.mean(axis=0)
    Lc = compute_Lc(coords)
    cn = (coords - centroid) / Lc
    e12 = cn[1] - cn[0]
    phi = np.arctan2(e12[1], e12[0])
    c, s = np.cos(phi), np.sin(phi)
    Rc = np.array([[c, s], [-s, c]])
    return cn @ Rc.T, Lc, Rc


def detJ_at_gp(coords):
    return np.array([shape_quad4_ref(coords, r, s)[2] for r, s in GAUSS_POINTS])


def detj_ratio(coords):
    dJ = detJ_at_gp(coords)
    return float(dJ.max() / dJ.min())


def interior_angles(coords):
    ang = np.empty(4)
    for i in range(4):
        u = coords[(i - 1) % 4] - coords[i]
        v = coords[(i + 1) % 4] - coords[i]
        cang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15)
        ang[i] = np.degrees(np.arccos(np.clip(cang, -1.0, 1.0)))
    return ang


def in_geometry_hull(coords):
    dJ = detJ_at_gp(coords)
    if np.min(dJ) <= 1e-9:
        return False
    if dJ.max() / dJ.min() > ENV_RATIO_MAX:
        return False
    ang = interior_angles(coords)
    return not (ang.min() < ENV_ANGLE_MIN or ang.max() > ENV_ANGLE_MAX)


def generate_distorted_quad(rng=np.random):
    """Verzerrungs-getriebene Geometrie in der wohlgestellten Huelle."""
    for _ in range(300):
        ax = np.exp(rng.uniform(np.log(0.4), np.log(2.5)))
        logt = rng.uniform(0, 1) * np.log(ENV_TAPER_MAX) * rng.choice([-1.0, 1.0])
        t = np.exp(logt)
        wb, wt, H = ax, ax * t, np.exp(rng.uniform(np.log(0.4), np.log(2.5)))
        coords = np.array([[-wb / 2, -H / 2], [wb / 2, -H / 2],
                           [wt / 2, H / 2], [-wt / 2, H / 2]])
        if rng.rand() < 0.5:
            coords = coords[:, ::-1].copy()
        g1 = rng.uniform(-0.4, 0.4)
        g2 = rng.uniform(-0.4, 0.4)
        coords = coords @ np.array([[1.0, g1], [g2, 1.0]]).T
        coords = coords + rng.uniform(-0.08, 0.08, (4, 2))
        area = 0.5 * np.sum(coords[:, 0] * np.roll(coords[:, 1], -1)
                            - np.roll(coords[:, 0], -1) * coords[:, 1])
        if area < 0:
            coords = coords[::-1].copy()
        if in_geometry_hull(coords):
            return coords
    return None


# ---------------------------------------------------------------------------
# Analytische Referenz: Energie, innere Kraefte, Tangente
# ---------------------------------------------------------------------------

def _bmat(F, hx, hy):
    """Nichtlineare B-Matrix (3x8); B = dEvec/du."""
    B = np.zeros((3, 8))
    B[0, 0::2] = F[0, 0] * hx
    B[1, 0::2] = F[0, 1] * hy
    B[2, 0::2] = F[0, 0] * hy + F[0, 1] * hx
    B[0, 1::2] = F[1, 0] * hx
    B[1, 1::2] = F[1, 1] * hy
    B[2, 1::2] = F[1, 0] * hy + F[1, 1] * hx
    return B


def energy_stiffness_force_ref(coords, disp, C=None, d=D_TRAIN):
    """Formaenderungsenergie W, innere Kraefte Finte (8) und Tangente Ke (8x8).

    Erweiterung von stiffness_force_ref um die Energie:
        W = int 0.5 * Evec' C Evec dV      (StVenant, Total Lagrange)
    Es gilt exakt Finte = dW/du und Ke = dFinte/du (Gate a prueft das).
    """
    C = C_MAT if C is None else C
    Ke = np.zeros((8, 8))
    Finte = np.zeros(8)
    W = 0.0
    ux = disp[:, 0]
    uy = disp[:, 1]
    for (r, s) in GAUSS_POINTS:
        hx, hy, detJ = shape_quad4_ref(coords, r, s)
        gradU = np.array([[ux @ hx, ux @ hy], [uy @ hx, uy @ hy]])
        F = I2 + gradU
        Eg = 0.5 * (F.T @ F - I2)
        Evec = np.array([Eg[0, 0], Eg[1, 1], 2 * Eg[0, 1]])
        Svec = C @ Evec
        S = np.array([[Svec[0], Svec[2]], [Svec[2], Svec[1]]])
        dV = detJ * 1.0 * d          # w = 1
        B = _bmat(F, hx, hy)
        L = np.stack([hx, hy], axis=1)              # (4,2)
        W += 0.5 * float(Evec @ Svec) * dV
        Finte += (B.T @ Svec) * dV
        Ke += (B.T @ C @ B + np.kron(L @ S @ L.T, I2)) * dV
    return W, Finte, Ke


def stiffness_force_ref(coords, disp, C=None, d=D_TRAIN):
    """Kompatibilitaets-Wrapper (nur F und K)."""
    _, F, K = energy_stiffness_force_ref(coords, disp, C, d)
    return K, F


def k0_ref(coords, C=None, d=D_TRAIN):
    """LINEARE Steifigkeit der (kanonischen) Geometrie -- billige Gauss-Schleife.

    Eigenstaendiger Port des Ke-Teils von element_quad4_lin.m: lineare
    B-Matrix, KEINE Doppelknotenschleife, KEIN zustandsabhaengiger Term.
    Fuer StVenant ist das exakt die Tangente bei u = 0 (S(0) = 0 loescht den
    geometrischen Term) -- Gate a prueft genau das gegen die nl-Referenz.
    """
    C = C_MAT if C is None else C
    K0 = np.zeros((8, 8))
    for (r, s) in GAUSS_POINTS:
        hx, hy, detJ = shape_quad4_ref(coords, r, s)
        B = np.zeros((3, 8))
        B[0, 0::2] = hx
        B[1, 1::2] = hy
        B[2, 0::2] = hy
        B[2, 1::2] = hx
        K0 += (B.T @ C @ B) * (detJ * 1.0 * d)
    return K0


def hull_measures(coords, disp):
    """max ||E_green||_F und max lokale Rotation [Grad] ueber die Gausspunkte."""
    ux = disp[:, 0]
    uy = disp[:, 1]
    mE = 0.0
    mR = 0.0
    for (r, s) in GAUSS_POINTS:
        hx, hy, _ = shape_quad4_ref(coords, r, s)
        F = I2 + np.array([[ux @ hx, ux @ hy], [uy @ hx, uy @ hy]])
        Eg = 0.5 * (F.T @ F - I2)
        mE = max(mE, float(np.linalg.norm(Eg)))
        theta = abs(np.degrees(np.arctan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])))
        mR = max(mR, theta)
    return mE, mR


# ---------------------------------------------------------------------------
# Zustands-Sampling und Ko-Rotation (verbatim aus der K/F-Variante)
# ---------------------------------------------------------------------------

def sample_state(coords_c, rng=np.random):
    """Verschiebungszustand in der Zustands-Huelle (kanonisch, transl.-projiziert)."""
    if rng.rand() < P_ZERO:
        return np.zeros((4, 2))
    for _ in range(20):
        G = rng.normal(0, 0.15, (2, 2))                        # affiner Anteil
        u = coords_c @ G.T + rng.normal(0, 0.05, (4, 2))       # + nicht-affin
        u -= u.mean(axis=0)                                    # Translation projizieren
        u *= np.exp(rng.uniform(np.log(AMP_LOG_MIN), 0.0))     # log-uniform
        mE, mR = hull_measures(coords_c, u)
        if mE <= E_MAX and mR <= ROT_MAX:
            return u
        f = 0.9 * min(E_MAX / max(mE, 1e-9), ROT_MAX / max(mR, 1e-9))
        u2 = u * f
        mE2, mR2 = hull_measures(coords_c, u2)
        if mE2 <= E_MAX and mR2 <= ROT_MAX:
            return u2
    return np.zeros((4, 2))


def corot_state(coords_c, u):
    """Zustands-Ko-Rotation: mittlere Starrkoerperrotation exakt herausdrehen."""
    hx, hy, _ = shape_quad4_ref(coords_c, 0.0, 0.0)
    gradU = np.array([[u[:, 0] @ hx, u[:, 0] @ hy],
                      [u[:, 1] @ hx, u[:, 1] @ hy]])
    F = I2 + gradU
    th = np.arctan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])
    ct, st = np.cos(th), np.sin(th)
    uc = (coords_c + u) @ np.array([[ct, -st], [st, ct]]) - coords_c
    return uc - uc.mean(axis=0)


# ---------------------------------------------------------------------------
# Kanonisierungs- + Ko-Rotations-KETTE (identisch im MATLAB-Element)
# ---------------------------------------------------------------------------

def chain_context(coord_e, Ue):
    """Alle Ketten-Groessen fuer ein physikalisches Element.

    coord_e : (4,2) physikalische Knotenkoordinaten
    Ue      : (8,)  physikalische Knotenverschiebungen

    Rueckgabe: dict mit
        chat (8,)   kanonische Koordinaten (flach)
        z    (8,)   kanonischer, ko-rotierter, translations-projizierter Zustand
        Lc, Rc, Tc, theta, Bmat, y, v, gth (8,), Hth (8,8)
    """
    coords_c, Lc, Rc = canonicalization_frame(coord_e)
    xhat = coords_c.reshape(-1)                       # (8,)
    Tc = np.kron(np.eye(4), Rc)                       # (8,8)

    v = P_TRANS @ (Tc @ Ue) / Lc                      # kanonisch + transl.-projiziert

    # theta = atan2(b, a) mit a, b LINEAR in v
    hx0, hy0, _ = shape_quad4_ref(coords_c, 0.0, 0.0)
    a1 = np.empty(8)
    b1 = np.empty(8)
    a1[0::2] = hx0
    a1[1::2] = hy0
    b1[0::2] = -hy0
    b1[1::2] = hx0

    a = 2.0 + float(a1 @ v)
    b = float(b1 @ v)
    r2 = a * a + b * b
    theta = np.arctan2(b, a)

    gth = (a * b1 - b * a1) / r2
    Hth = ((b * b - a * a) * (np.outer(b1, a1) + np.outer(a1, b1))
           + 2.0 * a * b * (np.outer(a1, a1) - np.outer(b1, b1))) / (r2 * r2)

    ct, st = np.cos(theta), np.sin(theta)
    Rm = np.array([[ct, st], [-st, ct]])              # R(-theta)
    Bmat = np.kron(np.eye(4), Rm)                     # (8,8)

    y = xhat + v
    z = P_TRANS @ (Bmat @ y - xhat)

    return dict(chat=xhat, z=z, Lc=Lc, Rc=Rc, Tc=Tc, theta=theta,
                Bmat=Bmat, y=y, v=v, gth=gth, Hth=Hth, a=a, b=b,
                coords_c=coords_c)


def chain_assemble(ctx, p, H, Emod=1.0, d=1.0):
    """Kanonische Ableitungen -> physikalische Finte und Ke.

    p : (8,)   dW/dz   im kanonischen, ko-rotierten Rahmen
    H : (8,8)  d2W/dz2 im kanonischen, ko-rotierten Rahmen
    """
    Bmat = ctx["Bmat"]
    y = ctx["y"]
    gth = ctx["gth"]
    Hth = ctx["Hth"]
    Tc = ctx["Tc"]
    Lc = ctx["Lc"]

    pt = P_TRANS @ p
    BSy = Bmat @ (SIGMA8 @ y)
    Gt = Bmat + np.outer(BSy, gth)                    # dz/dv (ohne P)

    g_v = Gt.T @ pt

    SBp = SIGMA8 @ (Bmat.T @ pt)
    C_v = (-np.outer(SBp, gth) - np.outer(gth, SBp)
           - float(pt @ (Bmat @ y)) * np.outer(gth, gth)
           + float(pt @ BSy) * Hth)
    K_v = Gt.T @ P_TRANS @ H @ P_TRANS @ Gt + C_v

    Finte = Emod * d * Lc * (Tc.T @ (P_TRANS @ g_v))
    Ke = Emod * d * (Tc.T @ (P_TRANS @ K_v @ P_TRANS) @ Tc)
    Ke = 0.5 * (Ke + Ke.T)
    return Finte, Ke


def canonical_model_analytic(chat, z, C=None):
    """'Modell' fuer Gate b: analytische Energie statt Netz.

    Rueckgabe (W, p, H) auf der kanonischen Geometrie mit E = d = 1.
    """
    coords_c = chat.reshape(4, 2)
    disp = z.reshape(4, 2)
    W, F, K = energy_stiffness_force_ref(coords_c, disp, C, 1.0)
    return W, F, K


def assemble_physical_analytic(coord_e, Ue, Emod=1.0, d=1.0):
    """Komplette Kette mit analytischem W (Gate b)."""
    ctx = chain_context(coord_e, Ue)
    _, p, H = canonical_model_analytic(ctx["chat"], ctx["z"])
    return chain_assemble(ctx, p, H, Emod, d)


def energy_physical_analytic(coord_e, Ue, Emod=1.0, d=1.0):
    """W_phys = E*d*Lc^2*What ueber die Kette (fuer FD-Gates)."""
    ctx = chain_context(coord_e, Ue)
    W, _, _ = canonical_model_analytic(ctx["chat"], ctx["z"])
    return Emod * d * ctx["Lc"] ** 2 * W


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def triu_vec(K):
    """8x8 -> 36 (column-major oberes Dreieck, identisch MATLAB find(triu))."""
    return np.array([K[i, j] for (i, j) in TRIU_IJ])


def triu_to_full(vec):
    K = np.zeros((8, 8))
    for v, (i, j) in zip(vec, TRIU_IJ):
        K[i, j] = K[j, i] = v
    return K


def _fd_grad(fun, x, h=1e-6):
    """Zentrale Differenzen des Gradienten einer Skalarfunktion."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        g[i] = (fun(xp) - fun(xm)) / (2 * h)
    return g


def _fd_jac(fun, x, h=1e-6):
    """Zentrale Differenzen der Jacobimatrix einer Vektorfunktion."""
    f0 = fun(x)
    J = np.zeros((len(f0), len(x)))
    for i in range(len(x)):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        J[:, i] = (fun(xp) - fun(xm)) / (2 * h)
    return J


def _rel(a, b):
    """Relativer Fehler in Frobenius-/2-Norm."""
    den = max(np.linalg.norm(b), 1e-14)
    return float(np.linalg.norm(a - b) / den)


def _random_physical_element(rng, scale=None, rot=None, shift=None):
    """Zufaellige Geometrie, frei rotiert/skaliert/verschoben."""
    coords_c = canonicalize_coords(generate_distorted_quad(rng))
    s = rng.uniform(0.3, 5.0) if scale is None else scale
    th = rng.uniform(-np.pi, np.pi) if rot is None else rot
    t = rng.uniform(-10, 10, 2) if shift is None else shift
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return coords_c @ R.T * s + t


# ---------------------------------------------------------------------------
# Gate a -- Selbstverifikation der Referenz-Targets
# ---------------------------------------------------------------------------

def gate_a(n_samples=5, seed=7, verbose=True, strict=True):
    """F = dW/dz, K = dF/dz, k0_ref = K(z=0), Skalierungsgesetz."""
    rng = np.random.RandomState(seed)
    worst = dict(dWdz=0.0, dFdz=0.0, k0=0.0, scale=0.0)

    for _ in range(n_samples):
        coords_c = canonicalize_coords(generate_distorted_quad(rng))
        u = corot_state(coords_c, sample_state(coords_c, rng))
        z = u.reshape(-1)

        W, F, K = energy_stiffness_force_ref(coords_c, u)

        fW = lambda zz: energy_stiffness_force_ref(coords_c, zz.reshape(4, 2))[0]
        fF = lambda zz: energy_stiffness_force_ref(coords_c, zz.reshape(4, 2))[1]

        worst["dWdz"] = max(worst["dWdz"], _rel(_fd_grad(fW, z), F))
        worst["dFdz"] = max(worst["dFdz"], _rel(_fd_jac(fF, z), K))

        K0a = k0_ref(coords_c)
        K0b = energy_stiffness_force_ref(coords_c, np.zeros((4, 2)))[2]
        worst["k0"] = max(worst["k0"], _rel(K0a, K0b))

        # Skalierungsgesetz: W = E*d*Lc^2*What, F = E*d*Lc*Fhat, K = E*d*Khat
        Emod, dth, Lc_t = 1000.0, 2.0, 3.7
        coords_p = coords_c * Lc_t
        Cp = material_C(Emod, NU_TRAIN, CONDITION)
        Wp, Fp, Kp = energy_stiffness_force_ref(coords_p, u * Lc_t, Cp, dth)
        e = max(abs(Wp - Emod * dth * Lc_t ** 2 * W) / max(abs(Wp), 1e-14),
                _rel(Fp, Emod * dth * Lc_t * F),
                _rel(Kp, Emod * dth * K))
        worst["scale"] = max(worst["scale"], e)

    ok = (worst["dWdz"] <= 1e-8 and worst["dFdz"] <= 1e-8
          and worst["k0"] <= 1e-10 and worst["scale"] <= 1e-12)
    if verbose:
        print("Gate a -- Referenz-Targets (Schwellen 1e-8 / 1e-8 / 1e-10 / 1e-12)")
        print(f"    F  vs FD(W)        : {worst['dWdz']:.3e}")
        print(f"    K  vs FD(F)        : {worst['dFdz']:.3e}")
        print(f"    k0_ref vs K(z=0)   : {worst['k0']:.3e}")
        print(f"    Skalierungsgesetz  : {worst['scale']:.3e}")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    if strict and not ok:
        raise AssertionError(f"Gate a fehlgeschlagen: {worst}")
    return ok, worst


# ---------------------------------------------------------------------------
# Gate b -- komplette Kette mit analytischem W
# ---------------------------------------------------------------------------

def gate_b(n_samples=10, seed=11, verbose=True, strict=True):
    """Kette vs. FD des Komposits Ue->W/Finte und vs. analytisches Element."""
    rng = np.random.RandomState(seed)
    worst = dict(fd_F=0.0, fd_K=0.0, oracle_F=0.0, oracle_K=0.0, rigid=0.0)

    for k in range(n_samples):
        coord_e = _random_physical_element(rng)
        Emod = rng.uniform(1.0, 1000.0)
        dth = rng.uniform(0.1, 2.0)

        # Zustand: kanonisch sampeln, dann in physikalische Groessen bringen
        coords_c, Lc, Rc = canonicalization_frame(coord_e)
        u_c = sample_state(coords_c, rng)
        # zusaetzlich um bis zu 60 Grad rotieren (Ko-Rotation muss das schlucken)
        th = rng.uniform(-np.pi / 3, np.pi / 3)
        Rth = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        u_c = (coords_c + u_c) @ Rth.T - coords_c
        U_nodes = (u_c * Lc) @ Rc            # zurueck in physikalische Richtung
        Ue = U_nodes.reshape(-1)

        # (b1) Kette vs. FD des Komposits
        F_chain, K_chain = assemble_physical_analytic(coord_e, Ue, Emod, dth)
        fW = lambda uu: energy_physical_analytic(coord_e, uu, Emod, dth)
        fF = lambda uu: assemble_physical_analytic(coord_e, uu, Emod, dth)[0]
        worst["fd_F"] = max(worst["fd_F"], _rel(_fd_grad(fW, Ue), F_chain))
        worst["fd_K"] = max(worst["fd_K"], _rel(_fd_jac(fF, Ue), K_chain))

        # (b2) Oracle: Kette vs. analytisches Element in PHYSIKALISCHEN Koordinaten
        Cp = material_C(Emod, NU_TRAIN, CONDITION)
        _, F_ref, K_ref = energy_stiffness_force_ref(coord_e, U_nodes, Cp, dth)
        worst["oracle_F"] = max(worst["oracle_F"], _rel(F_chain, F_ref))
        worst["oracle_K"] = max(worst["oracle_K"], _rel(K_chain, K_ref))

        # (b3) reine Starrkoerperbewegung -> z = 0, Finte = 0
        thr = rng.uniform(-np.pi, np.pi)
        Rr = np.array([[np.cos(thr), -np.sin(thr)], [np.sin(thr), np.cos(thr)]])
        U_rigid = (coord_e @ Rr.T + rng.uniform(-5, 5, 2)) - coord_e
        ctx_r = chain_context(coord_e, U_rigid.reshape(-1))
        worst["rigid"] = max(worst["rigid"], float(np.linalg.norm(ctx_r["z"])))

    ok = (worst["fd_F"] <= 1e-8 and worst["fd_K"] <= 1e-8
          and worst["oracle_F"] <= 1e-10 and worst["oracle_K"] <= 1e-10
          and worst["rigid"] <= 1e-12)
    if verbose:
        print("\nGate b -- Kette mit analytischem W (Schwellen 1e-8 / 1e-10 / 1e-12)")
        print(f"    Finte vs FD(W_phys)      : {worst['fd_F']:.3e}")
        print(f"    Ke    vs FD(Finte)       : {worst['fd_K']:.3e}")
        print(f"    Finte vs analyt. Element : {worst['oracle_F']:.3e}")
        print(f"    Ke    vs analyt. Element : {worst['oracle_K']:.3e}")
        print(f"    ||z|| bei Starrkoerper   : {worst['rigid']:.3e}")
        print(f"    -> {'GRUEN' if ok else 'ROT'}")
    if strict and not ok:
        raise AssertionError(f"Gate b fehlgeschlagen: {worst}")
    return ok, worst


def run_gates_a_b(verbose=True, strict=True):
    """Beide Gates -- laeuft bei jedem Trainingsstart automatisch mit."""
    ok_a, wa = gate_a(verbose=verbose, strict=strict)
    ok_b, wb = gate_b(verbose=verbose, strict=strict)
    return ok_a and ok_b, {"a": wa, "b": wb}


if __name__ == "__main__":
    print("=== Gates a und b (ohne Netz) ===\n")
    ok, _ = run_gates_a_b(strict=False)
    print(f"\nGesamt: {'GRUEN' if ok else 'ROT'}")
    raise SystemExit(0 if ok else 1)
