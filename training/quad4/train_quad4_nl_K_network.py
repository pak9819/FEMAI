"""
train_quad4_nl_K_network.py
===========================

Trainiert ein neuronales Netz, das fuer das NICHTLINEARE bilineare
Viereckselement (quad4, Total Lagrange) DIREKT

    * die tangentiale Element-Steifigkeitsmatrix  Ke  (8x8, symmetrisch)  UND
    * den inneren Kraftvektor                      Finte (8x1)

aus Elementgeometrie UND Verschiebungszustand vorhersagt (Option B: "Finte
mitgelernt"). Es ist die nichtlineare Entsprechung von train_quad4_K_network.py
(lineare Ke-Variante) -- mit dem entscheidenden Unterschied, dass die tangente
Steifigkeit hier ZUSTANDSABHAENGIG ist:  Ke = Ke(coord, Ue).

Warum Ke UND Finte
------------------
Der Newton-Loeser braucht das Residuum R = Finte - Fext (also Finte) UND die
Tangente Ke. In Option B liefert EIN Forward-Pass beide Groessen. Das ist der
maximale "Deep-Learned"-Charakter -- die gesamte Elementmechanik (F, E, S, Bu,
Gauss-Integration) entfaellt zur Laufzeit. Preis: Ke und Finte sind nicht
zwingend konsistent (Ke ist nur naeherungsweise die exakte Tangente von Finte),
und der Finte-Fehler geht DIREKT in die konvergierte Loesung ein.

Physikalische Faktorisierungen (exakt, wie linear)
--------------------------------------------------
St.-Venant-Kirchhoff: S = C*E_green, C ist konstant und C ~ E. Damit gilt

    Ke(E, d)    = E * d * K_hat
    Finte(E, d) = E * d * Lc * F_hat

Groessen- UND rotationsinvariant: Geometrie wird auf Schwerpunkt/Lc normiert und
kanonisiert (Kante 1->2 auf +x). Die Verschiebungen werden MITKANONISIERT und
translations-projiziert:

    c_hat = kanonisierte Koordinaten
    u_hat = (Rc * u) / Lc,  danach Mittelwert je Richtung abgezogen

Das MATLAB-Element rechnet exakt zurueck (Tc = blockdiag(Rc x4)):

    Ke    = E*d    * Tc' * K_hat_canon * Tc
    Finte = E*d*Lc * Tc' * F_hat_canon

nu = 0.3 und planeStrain sind fest eintrainiert (Material = StVenant).

Ko-Rotation des Zustands (exakt, Fix 2026-07-15b)
--------------------------------------------------
Fuer objektive Materialien gilt EXAKT: rotiert man den deformierten Zustand
starr um R, rotieren Finte und Ke exakt mit,

    Finte(R o u) = T_R * Finte(u),   Ke(R o u) = T_R * Ke(u) * T_R'.

Deshalb wird die mittlere Zustandsrotation (Polarwinkel von F am Element-
mittelpunkt) VOR dem Netz exakt herausgedreht und die Vorhersage exakt
zurueckrotiert -- dieselbe Trick-Klasse wie die Rc-Kanonisierung der
Geometrie. Das Netz lernt nur rotationsfreie Zustaende; die Zustandsrotation
ist zur Laufzeit UNBESCHRAENKT (Biegeprobleme mit grossen Verdrehungen
liegen damit strukturell in der Huelle, keine ROT_MAX-Grenze mehr).

Strukturelle Nebenbedingungen (exakt)
-------------------------------------
  * Symmetrie: nur die 36 Eintraege des oberen Dreiecks von Ke werden
    ausgegeben (column-major).
  * Kraeftegleichgewicht (Translation): sum_i Finte_i = 0 in x und y
    (folgt aus der Partition of Unity, exakt). Das Netz gibt daher nur die
    6 FREIEN Kraftkomponenten der Knoten 1..3 aus; Knoten 4 folgt aus
        F4x = -(F1x+F2x+F3x),  F4y = -(F1y+F2y+F3y).
  * Der Rotationsmode liegt bei u_hat != 0 NICHT im Nullraum von Ke -- er
    wird bewusst NICHT erzwungen (nur die 2 Translationen, im Element).

    Eingang (16): [c_hat (8);  u_hat (8)]
    Ausgang (42): [Ke_triu (36);  Finte_free (6)]   der KANONISCHEN Groessen

Datensatz
---------
Geometrien: verzerrungs-getrieben (wie Ke-Variante), harte Huelle
(detJ-Verhaeltnis, Innenwinkel). Zustaende: pro Geometrie mehrere
Verschiebungszustaende (affin + nicht-affine Moden), skaliert in eine
wohlgestellte ZUSTANDS-Huelle (max ||E_green|| <= E_MAX). Amplituden
LOG-UNIFORM ueber ~3 Dekaden (Newton-Endphase und realistische Lastniveaus,
||E_green|| ~ 1e-3), u_hat = 0 dicht gesampelt. Jeder Zustand wird vor dem
Training KO-ROTIERT (s.o.) -- eine Rotations-Augmentation ist damit
ueberfluessig (Fixes 2026-07-15, Details in DLFE_quad4_Entwicklung.md Teil III).

Architektur-Sweep / Deployment  -- analog train_quad4_K_network.py.
Das MATLAB-Element rechnet mit GELU und liest die Schichtzahl DYNAMISCH aus
dem Netz (beliebige Tiefe/Breite ohne Code-Aenderung; hier: Hidden 128, Tiefe 4).

Outputs
-------
quad4_nl_K_network.mat   -- Produktivdatei (element_quad4_nl_ai.m)
arch_sweep_nl_K/...       -- Sweep-Varianten, Log, Plots
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

save_dir  = Path(__file__).parent
element_dir = save_dir / ".." / ".." / "sourcecode" / "elements" / "quad4"
sweep_dir = save_dir / "arch_sweep_nl_K"
sweep_dir.mkdir(exist_ok=True)

_stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = sweep_dir / f"quad4_nl_K_sweep_{_stamp}.log"

logger = logging.getLogger("quad4_nl_K_sweep")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_h_console = logging.StreamHandler(); _h_console.setFormatter(logging.Formatter("%(message)s"))
_h_file = logging.FileHandler(log_path, mode="w", encoding="utf-8")
_h_file.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
logger.addHandler(_h_console); logger.addHandler(_h_file)


def log(msg=""):
    logger.info(msg)


log("=== Architektur-Sweep: nl K-Matrix + Finte Netz fuer quad4 (Option B) ===\n")
log(f"Logdatei: {log_path}")
log(f"Device:   {device}")
if device.type == "cuda":
    log(f"GPU:      {torch.cuda.get_device_name(0)}")
log()

# ---------------------------------------------------------------------------
# Festes Material (StVenant, planeStrain, nu = 0.3), E = d = 1
# ---------------------------------------------------------------------------

NU_TRAIN  = 0.3
CONDITION = "planeStrain"
MATERIAL  = "StVenant"
E_TRAIN   = 1.0
D_TRAIN   = 1.0

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------

N_TRAIN_ELEMENTS = 20000     # Geometrien
N_VAL_ELEMENTS   =  2000
STATES_PER_ELEM  =    12     # Verschiebungszustaende je Geometrie (12 statt 8,
                             # Fix 2026-07-15: Zustandsraum um Amplituden-Dekaden
                             # + entkoppelte Rotation gewachsen)
NUM_EPOCHS       =  4000
MINI_BATCH       =  2048     # nur relevant, wenn FULL_BATCH = False
LEARN_RATE       =  3e-3
FULL_BATCH       = True

# Nur EINE Architektur trainieren: GELU, Hidden 128, Tiefe 4 (4 verdeckte
# Schichten). Das MATLAB-Element liest Tiefe/Breite dynamisch aus dem Netz.
HIDDEN_CANDIDATES     = [128]
DEPTH_CANDIDATES      = [4]
ACTIVATION_CANDIDATES = ["GELU"]
ACCURACY_THRESHOLD    = 3.0    # % mittlerer Ke-Frobenius-Fehler (OK-Flag/Pool)

# Deployment: GELU + Tiefe 4 (muss zur oben trainierten Variante passen).
DEPLOY_ACT   = "GELU"
DEPLOY_DEPTH = 4
DEPLOY_BEST  = True

# Loss-Gewicht Finte (in Option B der Genauigkeits- UND Konvergenz-Engpass
# -> etwas hoeher gewichtet als Ke).
LAMBDA_F         = 2.0
# Eval-Floor gesenkt (Fix 2026-07-15): 0.1*RMS verdeckte die Fehler genau in
# dem Kleinlast-Bereich, in dem der Benchmark arbeitet -- die Metrik sah gut
# aus, obwohl der relative Fehler dort > 40 % war.
FINTE_FLOOR_FRAC = 0.02    # nur fuer die Eval-Metrik (relativer Finte-Fehler)

# Geometrie-Huelle (wie Ke-Variante)
ENV_RATIO_MAX = 4.5
ENV_ANGLE_MIN = 20.0
ENV_ANGLE_MAX = 160.0
ENV_TAPER_MAX = 4.0

# Zustands-Huelle (NEU) -- wohlgestellter, trainierter Verzerrungsbereich
E_MAX   = 0.2      # max ||E_green||_F ueber die Gausspunkte
ROT_MAX = 45.0     # max lokale Rotation [Grad] ueber die Gausspunkte

# Schnelllauf-Override (nur Pipeline-Test) -- selbe Architektur, weniger Daten
if os.environ.get("QUAD4_QUICK"):
    N_TRAIN_ELEMENTS = 3000
    STATES_PER_ELEM  = 6
    HIDDEN_CANDIDATES     = [128]
    DEPTH_CANDIDATES      = [4]
    ACTIVATION_CANDIDATES = ["GELU"]

ACTIVATIONS = {
    "Tanh": nn.Tanh, "ReLU": nn.ReLU, "GELU": nn.GELU,
    "SiLU": nn.SiLU, "ELU": nn.ELU, "Softplus": nn.Softplus,
}

_s3 = 1.0 / np.sqrt(3)
GAUSS_POINTS = [(-_s3, -_s3), (_s3, -_s3), (_s3, _s3), (-_s3, _s3)]

TRIU_IJ = [(i, j) for j in range(8) for i in range(j + 1)]   # 36
KE_N    = len(TRIU_IJ)     # 36
FINT_N  = 6                # freie Finte-Komponenten (Knoten 1..3)
N_IN    = 16
N_OUT   = KE_N + FINT_N    # 42

# ---------------------------------------------------------------------------
# Materialmatrix C (StVenant == Hooke-C bei planeStrain, konstant)
# ---------------------------------------------------------------------------

def material_C(E, nu, condition):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    if condition.lower() == "planestrain":
        c11 = lam + 2 * mu; c12 = lam
    elif condition.lower() == "planestress":
        c11 = lam / (nu - 1) + 2 * mu + 2 * lam
        c12 = lam / (nu - 1) + 2 * lam
    else:
        raise ValueError(condition)
    return np.array([[c11, c12, 0.0], [c12, c11, 0.0], [0.0, 0.0, mu]])


C_MAT = material_C(E_TRAIN, NU_TRAIN, CONDITION)

# ---------------------------------------------------------------------------
# Geometrie / Kinematik
# ---------------------------------------------------------------------------

def shape_quad4_ref(coords, r, s):
    x = coords[:, 0]; y = coords[:, 1]
    h_r = 0.25 * np.array([-(1 - s),  (1 - s),  (1 + s), -(1 + s)])
    h_s = 0.25 * np.array([-(1 - r), -(1 + r),  (1 + r),  (1 - r)])
    x_r = x @ h_r; x_s = x @ h_s
    y_r = y @ h_r; y_s = y @ h_s
    detJ = x_r * y_s - y_r * x_s
    h_x = ( y_s * h_r - y_r * h_s) / detJ
    h_y = (-x_s * h_r + x_r * h_s) / detJ
    return h_x, h_y, detJ


def compute_Lc(coords):
    centroid = coords.mean(axis=0)
    return float(np.mean(np.linalg.norm(coords - centroid, axis=1)))


def canonicalize_coords(coords):
    centroid = coords.mean(axis=0)
    Lc = compute_Lc(coords)
    cn = (coords - centroid) / Lc
    e12 = cn[1] - cn[0]
    phi = np.arctan2(e12[1], e12[0])
    c, s = np.cos(phi), np.sin(phi)
    Rc = np.array([[c, s], [-s, c]])
    return cn @ Rc.T


def detJ_at_gp(coords):
    return np.array([shape_quad4_ref(coords, r, s)[2] for r, s in GAUSS_POINTS])


def interior_angles(coords):
    ang = np.empty(4)
    for i in range(4):
        u = coords[(i - 1) % 4] - coords[i]
        v = coords[(i + 1) % 4] - coords[i]
        cang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15)
        ang[i] = np.degrees(np.arccos(np.clip(cang, -1.0, 1.0)))
    return ang


def generate_distorted_quad():
    for _ in range(300):
        ax = np.exp(np.random.uniform(np.log(0.4), np.log(2.5)))
        logt = np.random.uniform(0, 1) * np.log(ENV_TAPER_MAX) * np.random.choice([-1.0, 1.0])
        t = np.exp(logt)
        wb, wt, H = ax, ax * t, np.exp(np.random.uniform(np.log(0.4), np.log(2.5)))
        coords = np.array([[-wb / 2, -H / 2], [wb / 2, -H / 2],
                           [wt / 2,  H / 2], [-wt / 2,  H / 2]])
        if np.random.rand() < 0.5:
            coords = coords[:, ::-1].copy()
        g1 = np.random.uniform(-0.4, 0.4); g2 = np.random.uniform(-0.4, 0.4)
        coords = coords @ np.array([[1.0, g1], [g2, 1.0]]).T
        coords = coords + np.random.uniform(-0.08, 0.08, (4, 2))
        area = 0.5 * np.sum(coords[:, 0] * np.roll(coords[:, 1], -1)
                            - np.roll(coords[:, 0], -1) * coords[:, 1])
        if area < 0:
            coords = coords[::-1].copy()
        dJ = detJ_at_gp(coords)
        if np.min(dJ) <= 1e-9:
            continue
        if (dJ.max() / dJ.min() > ENV_RATIO_MAX):
            continue
        ang = interior_angles(coords)
        if ang.min() < ENV_ANGLE_MIN or ang.max() > ENV_ANGLE_MAX:
            continue
        return coords
    return None

# ---------------------------------------------------------------------------
# Analytische Referenz: tangentiale Ke UND Finte (Port von element_quad4_nl.m)
# ---------------------------------------------------------------------------

def _bmat(F, hx, hy):
    """Nichtlineare B-Matrix (3x8), Spalten (2i,2i+1) = Bui aus element_quad4_nl."""
    B = np.zeros((3, 8))
    B[0, 0::2] = F[0, 0] * hx
    B[1, 0::2] = F[0, 1] * hy
    B[2, 0::2] = F[0, 0] * hy + F[0, 1] * hx
    B[0, 1::2] = F[1, 0] * hx
    B[1, 1::2] = F[1, 1] * hy
    B[2, 1::2] = F[1, 0] * hy + F[1, 1] * hx
    return B


def stiffness_force_ref(coords, disp, C, d=1.0):
    """Ke (8x8) und Finte (8) am Zustand disp; identisch element_quad4_nl.m."""
    Ke = np.zeros((8, 8)); Finte = np.zeros(8)
    ux = disp[:, 0]; uy = disp[:, 1]
    I2 = np.eye(2)
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
        L = np.stack([hx, hy], axis=1)              # (4x2)
        Finte += (B.T @ Svec) * dV
        Ke += (B.T @ C @ B + np.kron(L @ S @ L.T, I2)) * dV
    return Ke, Finte


def hull_measures(coords, disp):
    """max ||E_green||_F und max lokale Rotation [Grad] ueber die Gausspunkte."""
    ux = disp[:, 0]; uy = disp[:, 1]
    I2 = np.eye(2); mE = 0.0; mR = 0.0
    for (r, s) in GAUSS_POINTS:
        hx, hy, _ = shape_quad4_ref(coords, r, s)
        F = I2 + np.array([[ux @ hx, ux @ hy], [uy @ hx, uy @ hy]])
        Eg = 0.5 * (F.T @ F - I2)
        mE = max(mE, float(np.linalg.norm(Eg)))
        theta = abs(np.degrees(np.arctan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])))
        mR = max(mR, theta)
    return mE, mR


# Zustands-Sampling -- Fixes 2026-07-15 (siehe DLFE_quad4_Entwicklung.md, Teil III):
#   1. Amplitude LOG-UNIFORM ueber ~3 Dekaden statt uniform(0.15, 1).
#      Vorher hatten < 1 % der Zustaende ||E_green|| < 0.02 -- der Benchmark
#      arbeitet aber genau dort (median ~0.006, Newton-Endphase noch darunter).
#   2. KEINE Rotations-Augmentation: die Zustandsrotation wird seit dem
#      Ko-Rotations-Fix (15.07.b) EXAKT herauskanonisiert (corot_state, s.u.).
#      Rotierte Kopien eines Zustands ergaeben identische Netzeingaben --
#      Augmentation waere redundant. (Der erste Fix-Versuch superponierte
#      Rotationen als DATEN -- das machte die Zielfunktion so viel haerter,
#      dass das Training kollabierte; Lauf 15.07. 13:13.)
AMP_LOG_MIN = 2e-3     # untere Amplituden-Grenze -> ||E_green|| bis ~1e-3 abgedeckt
P_ZERO      = 0.12     # Anteil exakt u = 0 (Newton-Startzustand jedes Lastschritts)


def sample_state(coords_c):
    """Verschiebungszustand in der Zustands-Huelle (kanonisch, transl.-projiziert)."""
    if np.random.rand() < P_ZERO:
        return np.zeros((4, 2))
    for _ in range(20):
        G = np.random.normal(0, 0.15, (2, 2))          # affiner Anteil
        u = coords_c @ G.T + np.random.normal(0, 0.05, (4, 2))   # + nicht-affin
        u -= u.mean(axis=0)                            # Translation projizieren
        u *= np.exp(np.random.uniform(np.log(AMP_LOG_MIN), 0.0))  # log-uniform
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
    """Zustands-Ko-Rotation (exakt, Fix 2026-07-15b): mittlere Starrkoerper-
    rotation (Polarwinkel von F am Elementmittelpunkt) herausdrehen, danach
    Translation projizieren. Grundlage ist die Objektivitaet von StVenant:

        Finte(R o u) = T_R * Finte(u),   Ke(R o u) = T_R * Ke(u) * T_R'

    Sie gilt exakt fuer JEDEN Winkel -- die Winkelwahl bestimmt nur, wie
    'rotationsfrei' der kanonische Zustand ist (Restrotation an den GPs
    nach corot: median ~1.4 Grad bei Eingangsrotationen bis 45 Grad).
    Das MATLAB-Element nutzt dieselbe Formel und rotiert die Vorhersage
    exakt zurueck. Verifiziert gegen das analytische Element (~1e-15)."""
    hx, hy, _ = shape_quad4_ref(coords_c, 0.0, 0.0)
    gradU = np.array([[u[:, 0] @ hx, u[:, 0] @ hy],
                      [u[:, 1] @ hx, u[:, 1] @ hy]])
    F = np.eye(2) + gradU
    th = np.arctan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])
    ct, st = np.cos(th), np.sin(th)
    uc = (coords_c + u) @ np.array([[ct, -st], [st, ct]]) - coords_c  # Zeilen um -th
    return uc - uc.mean(axis=0)

# ---------------------------------------------------------------------------
# Datensatz
# ---------------------------------------------------------------------------

def build_dataset(n_elements, label=""):
    inputs, targets, skipped = [], [], 0
    n_el = 0
    while n_el < n_elements:
        coords = generate_distorted_quad()
        if coords is None:
            skipped += 1
            continue
        coords_c    = canonicalize_coords(coords)
        coords_flat = coords_c.reshape(-1, order="C")
        for _ in range(STATES_PER_ELEM):
            u = corot_state(coords_c, sample_state(coords_c))  # kanonisch + ko-rotiert
            Ke, Finte = stiffness_force_ref(coords_c, u, C_MAT, D_TRAIN)
            ke_triu   = np.array([Ke[i, j] for (i, j) in TRIU_IJ])
            fint_free = Finte[:6]                       # Knoten 1..3 (Sum=0 -> Knoten4)
            u_flat    = u.reshape(-1, order="C")
            inputs.append(np.concatenate([coords_flat, u_flat]))
            targets.append(np.concatenate([ke_triu, fint_free]))
        n_el += 1
    if skipped:
        log(f"  Verworfene Geometrien ({label}): {skipped}")
    return (np.array(inputs,  dtype=np.float32),
            np.array(targets, dtype=np.float32))


log(f"Material: {MATERIAL}, nu = {NU_TRAIN}, {CONDITION}  |  E = d = 1")
log(f"Zustands-Huelle: ||E_green|| <= {E_MAX} | Zustand ko-rotiert "
    f"(Rotation zur Laufzeit unbeschraenkt) | {STATES_PER_ELEM} Zustaende/Geometrie")
log(f"Generiere Trainingsdaten ({N_TRAIN_ELEMENTS} Geometrien)...")
t0 = time.time()
X_train_np, Y_train_np = build_dataset(N_TRAIN_ELEMENTS, "train")
log(f"  -> {len(X_train_np)} Samples in {time.time()-t0:.1f} s")

log(f"Generiere Validierungsdaten ({N_VAL_ELEMENTS} Geometrien)...")
t0 = time.time()
X_val_np, Y_val_np = build_dataset(N_VAL_ELEMENTS, "val")
log(f"  -> {len(X_val_np)} Samples in {time.time()-t0:.1f} s\n")

# Finte-Targets auf O(1) skalieren (Ke-Targets sind schon O(1)). Ohne das ist
# der zufaellige Init-Ausgang (~O(1)) um Groessenordnungen groesser als die
# kleinen Finte-Targets (~O(0.1)) -> ein per-Sample-relativer Finte-Loss
# explodiert am Anfang (erdrueckt Ke), ein global normierter Loss gibt
# schlechte RELATIVE Finte-Genauigkeit auf leicht belasteten Zustaenden (->
# grosse dU, Newton konvergiert nicht). Mit der Skalierung bekommen BEIDE
# Ausgaenge dieselbe gut konditionierte per-Sample-relative Behandlung.
# Das MATLAB-Element multipliziert die Finte-Ausgabe mit FINT_SCALE zurueck.
FINT_SCALE = max(float(np.sqrt(np.mean((Y_train_np[:, KE_N:] ** 2).sum(axis=1)))), 1e-9)
Y_train_np[:, KE_N:] /= FINT_SCALE
Y_val_np[:,   KE_N:] /= FINT_SCALE
log(f"Finte-Ausgabeskala FINT_SCALE = {FINT_SCALE:.4e} (Targets auf O(1) gebracht)\n")

X_train = torch.tensor(X_train_np).to(device)
Y_train = torch.tensor(Y_train_np).to(device)
X_val   = torch.tensor(X_val_np).to(device)
Y_val   = torch.tensor(Y_val_np).to(device)
N_TRAIN = len(X_train)

# Loss: BEIDE Ausgaenge per-Sample RELATIV (Ke- und skalierte Finte-Targets
# sind jetzt beide O(1) -> balanciert). Der kleine Boden FINT_FLOOR faengt
# Zustaende nahe u=0 ab (dort ist die wahre Finte ~0 -> reiner Relativfehler
# waere singulaer). Per-Sample-relativ optimiert direkt die berichtete
# relative Finte-Metrik ueber ALLE Lastniveaus (wichtig fuer kleine dU).
# Floor-Historie (Details DLFE_quad4_Entwicklung.md, Teil III):
#   (0.2)^2  -- deckelte das komplette Kleinlast-Regime (Benchmark ~0.04*RMS)
#               -> dort effektiv absoluter Loss, 40-700 % relativer Fehler.
#   (0.02)^2 -- KOLLAPS (Lauf 15.07. 13:13): Init-Ausgaben O(1) gegen winzige
#               Nenner -> F-Loss startete ~6, der Gradienten-Schock drueckte
#               den Finte-Head dauerhaft auf 0 (F-Loss 3900 Epochen bei 0.715).
#   (0.05)^2 -- Kompromiss, zusammen mit Klein-Init der Ausgabeschicht (KFNet):
#               sanfter Start UND Relativ-Gewichtung bis 5 % der RMS-Kraft.
FINT_FLOOR   = 2.5e-3   # (0.05)^2 in skalierten Einheiten (RMS ||F_hat_scaled|| = 1)
FINT_FLOOR_T = torch.tensor(FINT_FLOOR, dtype=torch.float32, device=device)


def combined_loss(pred, tgt):
    """Ke und (skalierte) Finte je per-Sample relativ; Finte mit kleinem Boden."""
    pk, pf = pred[:, :KE_N], pred[:, KE_N:]
    tk, tf = tgt[:, :KE_N], tgt[:, KE_N:]
    lk = (((pk - tk) ** 2).sum(dim=1) / (tk ** 2).sum(dim=1).clamp_min(1e-12)).mean()
    lf = (((pf - tf) ** 2).sum(dim=1) / ((tf ** 2).sum(dim=1) + FINT_FLOOR_T)).mean()
    return lk + LAMBDA_F * lf, lk, lf

# ---------------------------------------------------------------------------
# Netzwerk
# ---------------------------------------------------------------------------

class KFNet(nn.Module):
    def __init__(self, hidden, depth, act_name):
        super().__init__()
        Act = ACTIVATIONS[act_name]
        layers = [nn.Linear(N_IN, hidden), Act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), Act()]
        layers += [nn.Linear(hidden, N_OUT)]
        self.net = nn.Sequential(*layers)
        # Klein-Init der Ausgabeschicht (Fix 2026-07-15b): Startvorhersage ~0
        # OHNE Gradienten-Schock. Mit Standard-Init sind die Ausgaben O(1),
        # die skalierten Finte-Targets kleiner Zustaende aber winzig -- der
        # per-Sample-relative Loss startete bei ~6 und drueckte den Finte-Head
        # dauerhaft auf 0 (Kollaps, Lauf 15.07. 13:13).
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.zero_()

    def forward(self, x):
        return self.net(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def forward_macs(hidden, depth):
    return N_IN * hidden + (depth - 1) * hidden * hidden + hidden * N_OUT


def arch_tag(act, h, d):
    return f"{act}_h{h}_d{d}"

# ---------------------------------------------------------------------------
# Training einer Variante
# ---------------------------------------------------------------------------

def train_one(hidden, depth, act_name):
    model = KFNet(hidden, depth, act_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    # patience 40 -> 100 (Fix 2026-07-15b): das Plateau des kollabierten
    # Finte-Heads hatte die LR schon bei Epoche ~1200 auf min_lr gefahren
    # und damit auch das Ke-Training eingefroren.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-5)

    log(f"\n--- {act_name} | Hidden = {hidden:3d} | Tiefe = {depth} "
        f"| Parameter: {count_params(model):,} | MACs: {forward_macs(hidden,depth):,} ---")
    log(f"    Epochs: {NUM_EPOCHS}  |  Batch: {'voll' if FULL_BATCH else MINI_BATCH}  |  LR: {LEARN_RATE}")

    num_batches = 1 if FULL_BATCH else int(np.ceil(N_TRAIN / MINI_BATCH))
    hist_val = []
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        if FULL_BATCH:
            loss, lk, lf = combined_loss(model(X_train), Y_train)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        else:
            idx = torch.randperm(N_TRAIN, device=device)
            for b in range(num_batches):
                sl = idx[b * MINI_BATCH:(b + 1) * MINI_BATCH]
                loss, _, _ = combined_loss(model(X_train[sl]), Y_train[sl])
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        with torch.no_grad():
            vloss, vlk, vlf = combined_loss(model(X_val), Y_val)
        vloss = vloss.item()
        hist_val.append(vloss)
        scheduler.step(vloss)

        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"    Epoch {epoch:4d}/{NUM_EPOCHS} | Val: {vloss:.3e} "
                f"(Ke {vlk.item():.3e} / F {vlf.item():.3e}) | LR: {lr_now:.1e} | "
                f"{time.time()-t_start:.0f} s")

    log(f"    Fertig. Gesamtzeit: {time.time()-t_start:.1f} s")
    return model, hist_val

# ---------------------------------------------------------------------------
# Auswertung (echte relative Frobenius-Metriken)
# ---------------------------------------------------------------------------

def triu_to_full(vec):
    K = np.zeros((8, 8))
    for v, (i, j) in zip(vec, TRIU_IJ):
        K[i, j] = K[j, i] = v
    return K


def fint_full(free6):
    f4x = -(free6[0] + free6[2] + free6[4])
    f4y = -(free6[1] + free6[3] + free6[5])
    return np.array([free6[0], free6[1], free6[2], free6[3],
                     free6[4], free6[5], f4x, f4y])


def evaluate(model):
    model.eval()
    with torch.no_grad():
        Yp = model(X_val).cpu().numpy()
    Yr = Y_val_np

    # Ke relativ (volle 8x8)
    keErr = np.array([np.linalg.norm(triu_to_full(p[:KE_N]) - triu_to_full(r[:KE_N]))
                      / max(np.linalg.norm(triu_to_full(r[:KE_N])), 1e-12)
                      for p, r in zip(Yp, Yr)]) * 100

    # Finte relativ (volle 8), Boden fuer verschwindende Kraefte
    fFull_r = np.array([fint_full(r[KE_N:]) for r in Yr])
    fFull_p = np.array([fint_full(p[KE_N:]) for p in Yp])
    fnorm   = np.linalg.norm(fFull_r, axis=1)
    floor   = FINTE_FLOOR_FRAC * np.sqrt(np.mean(fnorm ** 2))
    fErr    = np.linalg.norm(fFull_p - fFull_r, axis=1) / np.maximum(fnorm, floor) * 100
    big     = fnorm > np.median(fnorm)     # "belastete" Zustaende

    return keErr, fErr, fnorm, big

# ---------------------------------------------------------------------------
# Speichern / Metadaten
# ---------------------------------------------------------------------------

def build_mat_data(model, hidden, depth, act_name):
    md = {
        "input_order":       "canonical_[c_hat(8)_then_u_hat(8)]",
        "normalization":     "centroid_Lc_mean_node_distance",
        "canonicalization":  "edge_n1n2_to_pos_x_after_centroid_Lc",
        "state_input":       "canonical_displacements_translation_projected_u_hat=Rc*u/Lc",
        "output_order":      "Ke_triu(36)_column_major_then_Finte_free(6)_nodes123",
        "finte_constraint":  "sum_i Finte_i = 0 (x,y) -> node4 = -(n1+n2+n3)",
        # Finte-Ausgabe wurde beim Training auf O(1) skaliert -> Element muss
        # die 6 Finte-Ausgaenge mit FINT_SCALE zurueckmultiplizieren.
        "finte_output_scale": float(FINT_SCALE),
        "material":          MATERIAL,
        "nu_train":          float(NU_TRAIN),
        "condition":         CONDITION,
        "E_train":           float(E_TRAIN),
        "d_train":           float(D_TRAIN),
        "hidden":            float(hidden),
        "depth":             float(depth),
        "activation":        act_name,
        "env_ratio_max":     float(ENV_RATIO_MAX),
        "env_angle_min":     float(ENV_ANGLE_MIN),
        "env_angle_max":     float(ENV_ANGLE_MAX),
        "state_Egreen_max":  float(E_MAX),
        # Zustandsrotation wird seit dem Ko-Rotations-Fix EXAKT behandelt ->
        # zur Laufzeit unbeschraenkt (180 = keine Einschraenkung; die
        # Huellen-Anzeige in .._08_ai_nl_check.m liest dieses Feld).
        "state_rot_max_deg": 180.0,
        "state_corotation":  "mean_polar_angle_at_center_removed__u_c=R(-th)(x+u)-x",
    }
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    md["num_linear_layers"] = float(len(linears))
    for i, layer in enumerate(linears):
        md[f"W{i+1}"] = layer.weight.detach().cpu().numpy()
        md[f"b{i+1}"] = layer.bias.detach().cpu().numpy()
    return md


def save_variant(model, hidden, depth, act):
    tag = arch_tag(act, hidden, depth)
    torch.save(model.state_dict(), sweep_dir / f"quad4_nl_K_{tag}.pt")
    scipy.io.savemat(sweep_dir / f"quad4_nl_K_{tag}.mat", build_mat_data(model, hidden, depth, act))


PROD_MAT = save_dir / "quad4_nl_K_network.mat"


def save_production(model, hidden, depth, act):
    # Backup vor dem Ueberschreiben (PROD_MAT ist die einzige Kopie -- nicht
    # git-getrackt, kein Undo sonst moeglich, falls die neue Variante schlechter ist).
    if PROD_MAT.exists():
        backup = sweep_dir / f"quad4_nl_K_network_backup_{_stamp}.mat"
        scipy.io.savemat(backup, scipy.io.loadmat(PROD_MAT))
        log(f"Backup der bisherigen Produktivdatei: {backup.name}")
    scipy.io.savemat(PROD_MAT, build_mat_data(model, hidden, depth, act))

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

grid = [(act, h, d) for act in ACTIVATION_CANDIDATES
        for d in DEPTH_CANDIDATES for h in HIDDEN_CANDIDATES]

log("\n" + "=" * 72)
log(f"  Architektur-Sweep: {len(grid)} Netze "
    f"({ACTIVATION_CANDIDATES} x Tiefe {DEPTH_CANDIDATES} x Breite {HIDDEN_CANDIDATES})")
log("=" * 72)

results = []
t_sweep = time.time()
for n, (act, h, d) in enumerate(grid, start=1):
    log(f"\n>>> Variante {n}/{len(grid)}")
    model, hist_val = train_one(h, d, act)
    keErr, fErr, fnorm, big = evaluate(model)
    save_variant(model, h, d, act)
    results.append({
        'act': act, 'hidden': h, 'depth': d, 'tag': arch_tag(act, h, d), 'model': model,
        'val_hist': hist_val, 'keErr': keErr, 'fErr': fErr, 'fnorm': fnorm, 'big': big,
        'n_params': count_params(model), 'macs': forward_macs(h, d),
        'ke_mean': float(keErr.mean()), 'ke_p95': float(np.percentile(keErr, 95)),
        'f_mean': float(fErr.mean()), 'f_big': float(fErr[big].mean()),
    })

log(f"\nSweep abgeschlossen. Gesamtzeit: {time.time()-t_sweep:.1f} s ({len(grid)} Netze)")

# ---------------------------------------------------------------------------
# Vergleichstabelle
# ---------------------------------------------------------------------------

by_acc = sorted(results, key=lambda r: r['ke_mean'])

log("\n" + "=" * 100)
log("  MODELLVERGLEICH  (sortiert nach mittlerem Ke-Frobenius-Fehler)")
log("-" * 100)
log(f"  {'Rang':>4} | {'Akt':>5} | {'Breite':>6} | {'Tiefe':>5} | {'Param':>8} | "
    f"{'MACs':>8} | {'Ke O':>7} | {'Ke 95':>7} | {'F(bel.)':>8} | {'OK?':>4}")
log("-" * 100)
for rank, r in enumerate(by_acc, start=1):
    ok = r['ke_mean'] < ACCURACY_THRESHOLD
    log(f"  {rank:>4} | {r['act']:>5} | {r['hidden']:>6} | {r['depth']:>5} | "
        f"{r['n_params']:>8,} | {r['macs']:>8,} | {r['ke_mean']:>6.3f}% | {r['ke_p95']:>6.3f}% | "
        f"{r['f_big']:>7.3f}% | {'JA' if ok else 'nein':>4}")
log("=" * 100)

most_accurate = by_acc[0]
log(f"\nGENAUESTE Variante (Ke): {most_accurate['act']}, Breite {most_accurate['hidden']}, "
    f"Tiefe {most_accurate['depth']} (Ke O {most_accurate['ke_mean']:.3f}%, "
    f"F(bel.) {most_accurate['f_big']:.3f}%)")

# ---------------------------------------------------------------------------
# Deployment: billigste GELU/Tiefe-3-Variante unter Schwellwert
# ---------------------------------------------------------------------------
if DEPLOY_BEST:
    pool_ok = [r for r in results if r['act'] == DEPLOY_ACT and r['depth'] == DEPLOY_DEPTH
               and r['ke_mean'] < ACCURACY_THRESHOLD]
    pool_all = [r for r in results if r['act'] == DEPLOY_ACT and r['depth'] == DEPLOY_DEPTH]
    pool = pool_ok if pool_ok else pool_all
    if pool:
        dep = min(pool, key=lambda r: (r['macs'], r['ke_mean']))
        save_production(dep['model'], dep['hidden'], dep['depth'], DEPLOY_ACT)
        log(f"\nDEPLOYT nach {PROD_MAT.name}: {DEPLOY_ACT}, Breite {dep['hidden']}, "
            f"Tiefe {dep['depth']}  (Ke O {dep['ke_mean']:.3f}%, F(bel.) {dep['f_big']:.3f}%)")
    else:
        log(f"\nWARNUNG: keine {DEPLOY_ACT}/Tiefe-{DEPLOY_DEPTH}-Variante -> kein Deployment.")

# Fehler nach Zustands-Amplitude (Newton-Endphase = kleine ||F||)
log("\nFehler nach Belastung (Ke / Finte), Terzil von ||F_hat||:")
best = by_acc[0]
fn = np.linalg.norm(np.array([fint_full(r[KE_N:]) for r in Y_val_np]), axis=1)
q1, q2 = np.percentile(fn, [33, 67])
for name, mask in [("klein (Newton-Ende)", fn <= q1),
                   ("mittel", (fn > q1) & (fn <= q2)),
                   ("gross", fn > q2)]:
    log(f"  {name:>20}: Ke {best['keErr'][mask].mean():.3f}%  "
        f"Finte {best['fErr'][mask].mean():.3f}%")

log("\nHINWEIS: element_quad4_nl_ai.m rechnet mit GELU und liest die Schichtzahl dynamisch "
    "(hier Tiefe 4).\n         Es nutzt dieselbe Kanonisierung + Verschiebungs-Kanonisierung. "
    "Material = StVenant, nu/Zustand fest.")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

act_colors = {a: plt.cm.tab10.colors[i] for i, a in enumerate(ACTIVATION_CANDIDATES)}
depth_ls   = {2: "--", 3: "-", 4: "-."}
depth_mk   = {2: "s", 3: "D", 4: "^"}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("quad4 nl K+Finte Netz: Architektur-Sweep (Option B)", fontsize=13)

ax = axes[0, 0]
for r in results:
    ax.semilogy(r['val_hist'], color=act_colors[r['act']],
                linestyle=depth_ls.get(r['depth'], "-"), alpha=0.7, linewidth=1.0)
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Loss (Ke+F)")
ax.set_title("Trainingsverlauf"); ax.grid(True, which="both", alpha=0.3)

ax = axes[0, 1]
for r in results:
    ax.scatter(r['macs'], r['ke_mean'], s=70, color=act_colors[r['act']],
               marker=depth_mk.get(r['depth'], "o"), edgecolor="k", linewidth=0.4)
ax.axhline(ACCURACY_THRESHOLD, color="k", linestyle=":", linewidth=1.5)
ax.set_xlabel("Forward-MACs"); ax.set_ylabel("Ke O Fehler [%]")
ax.set_title("Pareto: Ke-Genauigkeit vs. Kosten"); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
order = sorted(results, key=lambda r: r['ke_mean'])
xx = np.arange(len(order)); wdt = 0.4
ax.bar(xx - wdt/2, [r['ke_mean'] for r in order], wdt, label="Ke",
       color=[act_colors[r['act']] for r in order])
ax.bar(xx + wdt/2, [r['f_big'] for r in order], wdt, label="Finte (belastet)",
       color=[act_colors[r['act']] for r in order], alpha=0.5)
ax.axhline(ACCURACY_THRESHOLD, color="k", linestyle=":", linewidth=1.5)
ax.set_xticks(xx); ax.set_xticklabels([r['tag'] for r in order], rotation=90, fontsize=6)
ax.set_ylabel("O Fehler [%]"); ax.set_title("Ke- und Finte-Fehler je Architektur")
ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)

ax = axes[1, 1]
b = by_acc[0]
ax.scatter(b['fnorm'], b['fErr'], s=8, alpha=0.3, color=act_colors[b['act']])
ax.set_xlabel("||F_hat|| (Belastung)"); ax.set_ylabel("Finte-Fehler [%]")
ax.set_title("Finte-Fehler ueber Belastung (beste Variante)")
ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plot_path = sweep_dir / f"quad4_nl_K_sweep_{_stamp}.png"
plt.savefig(plot_path, dpi=150)
log(f"\nPlot gespeichert: {plot_path}")
log(f"Log  gespeichert: {log_path}")
log("\nFertig.")
