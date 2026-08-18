"""
train_quad4_K_network.py
========================

Trainiert ein neuronales Netz, das die *gesamte* Element-Steifigkeitsmatrix
Ke (8x8) des bilinearen Viereckselements (quad4) approximiert -- nicht mehr
nur die B-Matrix-Ableitungen.

Konzept (Deep Learned Finite Elements -- Variante "ganze Steifigkeitsmatrix")
-----------------------------------------------------------------------------
Klassisch wird die Elementsteifigkeit durch Gauss-Integration aufgebaut:

    Ke = int_Ve B' C B dV   (8x8, symmetrisch)

Hier lernt das Netz diese Matrix direkt aus der Elementgeometrie. Es werden
keine Formfunktionsableitungen und keine Gauss-Schleife mehr benoetigt -- ein
einziger Forward-Pass liefert die komplette Steifigkeitsmatrix.

Zwei physikalische Eigenschaften machen das Problem gut lernbar:

  1. GROESSEN- UND ROTATIONSINVARIANZ (2D):
     In der ebenen Elastizitaet ist Ke unabhaengig von der absoluten
     Elementgroesse. Skaliert man ein Element gleichmaessig (x -> L*x), so
     gilt  B -> B/L,  dV -> L^2 dV,  also  Ke -> Ke (unveraendert).
     Deshalb wird die Geometrie auf Schwerpunkt und charakteristische Laenge
     Lc normalisiert. ZUSAETZLICH wird die Rotation entfernt (Kanonisierung):
     das Element wird so gedreht, dass die Kante Knoten1->Knoten2 auf der
     +x-Achse liegt. Dadurch bilden alle rotierten Varianten einer Form auf
     DIESELBE Eingabe ab -- die Eingangsmannigfaltigkeit verliert eine
     Dimension, das Netz muss Rotationsequivarianz nicht mehr aus Daten lernen.
     Das MATLAB-Element dreht die Vorhersage exakt zurueck:
        Ke_hat = Tc' * Khat_canon * Tc ,  Tc = blockdiag(Rc, Rc, Rc, Rc)
     wobei Rc die Kanonisierungs-Drehung ist (verifiziert, Fehler ~1e-16).

  2. LINEARITAET IN E*d:
     Ke skaliert linear mit dem E-Modul und der Dicke d:
        Ke(E, d) = E * d * Khat
     Das Netz lernt daher die dimensionslose Form-Steifigkeit Khat
     (trainiert mit E = 1, d = 1). In MATLAB wird mit dem tatsaechlichen
     E*d multipliziert. Querkontraktionszahl nu und der ebene Zustand
     (planeStrain/planeStress) sind hingegen *fest* ins Netz eintrainiert.

Symmetrie:
    Ke ist symmetrisch -> nur die 36 Eintraege des oberen Dreiecks
    (inkl. Diagonale, Spalten-weise/column-major) werden ausgegeben.
    Die Symmetrie ist damit strukturell exakt garantiert.

    Eingang  (8):  [x1_c, y1_c, x2_c, y2_c, x3_c, y3_c, x4_c, y4_c]
                   (kanonisierte Koordinaten: Schwerpunkt/Lc + Kante 1->2 -> +x)
    Ausgang (36):  oberes Dreieck von Khat der KANONISCHEN Form (column-major)

Datensatz (verzerrungs-getrieben):
    Statt nur leicht gestoerter Rechtecke wird der Formraum breit mit
    VERZERRTEN Elementen gefuellt (Trapez/Taper, Scherung, Seitenverhaeltnis,
    Jitter), begrenzt durch eine harte, wohlgestellte Huelle (ENV_*: detJ-
    Verhaeltnis, Innenwinkel). Das verbessert die Generalisierung auf verzerrte
    Elemente (siehe examples/FEMSolid_ex_quad4_06_ai_patch_distortion.m).

Architektur-Sweep (Neuronen x Tiefe x Aktivierung)
--------------------------------------------------
Da der Forward-Pass im FEM-Code *pro Element* laeuft, soll das Netz so klein
wie moeglich sein, ohne zu viel Genauigkeit zu verlieren. Dieses Skript testet
in EINEM Lauf ein konfigurierbares Gitter aus drei Achsen:

  * HIDDEN_CANDIDATES     -- Neuronen je verdeckter Schicht (Netz-"Breite")
  * DEPTH_CANDIDATES      -- Anzahl verdeckter Schichten   (Nichtlinearitaet)
  * ACTIVATION_CANDIDATES -- Aktivierungsfunktion

Alle Varianten werden auf demselben Datensatz trainiert und am Ende verglichen
(Genauigkeit vs. Kosten = Parameter / Forward-MACs). Empfohlen wird die
*billigste* Variante unter dem Frobenius-Schwellwert (ACCURACY_THRESHOLD); die
*genaueste* Variante wird zusaetzlich ausgewiesen.

Jede Variante wird separat im Unterordner arch_sweep/ abgelegt. Zusaetzlich
wird (DEPLOY_BEST) die billigste GELU-Variante unter dem Schwellwert direkt
als produktive Datei quad4_K_network.mat gespeichert -- Training und MATLAB-
Einsatz sind damit in EINEM Lauf konsistent.

WICHTIG fuer das MATLAB-Element:
    element_quad4_lin_ai.m rechnet mit  GELU  und  genau 3 verdeckten
    Schichten  und nutzt DIESELBE Kanonisierung. Ein Gewinner mit anderer
    Aktivierung oder Tiefe laeuft dort NICHT korrekt, bevor das Element
    angepasst wurde (die Hidden-Groesse wird dynamisch gelesen).

Logging:
    Saemtliche Konsolenausgaben sowie der finale Modellvergleich werden
    zusaetzlich in eine Logdatei unter arch_sweep/ geschrieben.

Outputs (alle in arch_sweep/)
-----------------------------
quad4_K_<act>_h<H>_d<D>.pt      -- PyTorch state dict je Variante
quad4_K_<act>_h<H>_d<D>.mat     -- Gewichte + Metadaten fuer MATLAB je Variante
quad4_arch_sweep_<zeit>.log     -- vollstaendiges Log (Print-Ausgaben + Vergleich)
quad4_arch_sweep_<zeit>.png     -- Vergleichsplots
"""

import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Ausgabeverzeichnis und Logging einrichten
#   -> alles, was geprintet wird, landet zusaetzlich in der Logdatei.
# ---------------------------------------------------------------------------

save_dir  = Path(__file__).parent
element_dir = save_dir / ".." / ".." / "sourcecode" / "elements" / "quad4"
sweep_dir = save_dir / "arch_sweep"
sweep_dir.mkdir(exist_ok=True)

_stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = sweep_dir / f"quad4_arch_sweep_{_stamp}.log"

logger = logging.getLogger("quad4_arch_sweep")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_fmt_console = logging.Formatter("%(message)s")
_fmt_file    = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
_h_console = logging.StreamHandler()
_h_console.setFormatter(_fmt_console)
_h_file = logging.FileHandler(log_path, mode="w", encoding="utf-8")
_h_file.setFormatter(_fmt_file)
logger.addHandler(_h_console)
logger.addHandler(_h_file)


def log(msg=""):
    """Ausgabe auf Konsole UND in die Logdatei (ersetzt print)."""
    logger.info(msg)


log("=== Architektur-Sweep K-Matrix Network fuer quad4-Element ===\n")
log(f"Logdatei: {log_path}")
log(f"Device:   {device}")
if device.type == "cuda":
    log(f"GPU:      {torch.cuda.get_device_name(0)}")
log()

# ---------------------------------------------------------------------------
# Festes Material (nu und ebener Zustand sind ins Netz eintrainiert)
# E und d werden analytisch herausfaktorisiert (Training mit E = d = 1).
# ---------------------------------------------------------------------------

NU_TRAIN   = 0.3
CONDITION  = "planeStrain"      # 'planeStrain' oder 'planeStress'
E_TRAIN    = 1.0
D_TRAIN    = 1.0

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------

N_TRAIN_ELEMENTS   = 20000
N_VAL_ELEMENTS     =  2000
NUM_EPOCHS         =  4000
MINI_BATCH         =   512     # nur relevant, wenn FULL_BATCH = False
LEARN_RATE         =  3e-3

# --- Voll-Batch-Training (grosser Geschwindigkeitshebel) -------------------
# Das Netz ist winzig (wenige tausend Parameter) und der komplette Datensatz
# liegt auf der GPU. Die reine Rechenzeit pro Schritt ist damit
# vernachlaessigbar -- das Training war OVERHEAD-gebunden: ~40 Mini-Batches
# je Epoche bedeuteten ~40 Kernel-Launches PLUS ein loss.item() (GPU->CPU-
# Sync) pro Schritt. Voll-Batch macht genau EINEN Optimierer-Schritt je Epoche
# (exakter Gradient) und synchronisiert nur einmal pro Epoche. Dadurch faellt
# die Laufzeit drastisch. Weil pro Epoche nun 1 statt ~40 Schritte laufen, sind
# etwas mehr Epochen noetig (NUM_EPOCHS hoch) und eine etwas groessere Lernrate
# sinnvoll -- unterm Strich trotzdem ein Bruchteil der Zeit.
FULL_BATCH         = True

# --- Architektur-Sweep: drei Achsen (oben leicht erweiterbar) --------------
# HINWEIS: Die Gesamtzahl der trainierten Netze ist
#          len(HIDDEN) * len(DEPTH) * len(ACTIVATION).
#          Jedes Netz wird NUM_EPOCHS lang trainiert -> Laufzeit beachten.
HIDDEN_CANDIDATES     = [16, 32, 64]          # Neuronen je Schicht (Breite)
DEPTH_CANDIDATES      = [2, 3]                 # Anzahl verdeckter Schichten
ACTIVATION_CANDIDATES = ["Tanh", "GELU", "ReLU"]   # Aktivierungsfunktion
ACCURACY_THRESHOLD    = 3.0                    # % mittlerer Frobenius-Fehler

# --- Deployment ------------------------------------------------------------
# Nach dem Sweep die billigste GELU-Variante (das MATLAB-Element rechnet mit
# GELU) unter dem Schwellwert direkt als Produktivdatei quad4_K_network.mat
# ablegen -- so ist die Pipeline (Training -> MATLAB) in einem Lauf komplett.
DEPLOY_BEST = True

# --- Verzerrungs-Huelle des Datensatzes ------------------------------------
# Der Datensatz wird gezielt mit VERZERRTEN Elementen (Trapez/Taper, Scherung,
# Seitenverhaeltnis, Jitter) gefuellt -- statt nur leicht gestoerter Rechtecke.
# Eine harte Huelle begrenzt die Verzerrung auf ein noch WOHLGESTELLTES Gebiet:
# jenseits davon (detJ nahe 0) waechst die wahre Ke ins Singulaere und ist
# weder lern- noch sinnvoll nutzbar. Diese Huelle ist der dokumentierte
# Gueltigkeitsbereich des Netzes.
ENV_RATIO_MAX = 4.5     # max. detJ(max)/detJ(min) ueber die Gausspunkte
ENV_ANGLE_MIN = 20.0    # min. Innenwinkel [Grad]
ENV_ANGLE_MAX = 160.0   # max. Innenwinkel [Grad]
ENV_TAPER_MAX = 4.0     # max. Taper-Faktor der Trapezform (obere/untere Kante)

# --- Schnelllauf-Override (nur zum schnellen Verifizieren der Pipeline) -----
# Setzt man die Umgebungsvariable QUAD4_QUICK, wird nur EINE (GELU-)Variante
# kurz trainiert. Fuer den produktiven Sweep die Variable NICHT setzen.
import os
if os.environ.get("QUAD4_QUICK"):
    HIDDEN_CANDIDATES     = [32]
    DEPTH_CANDIDATES      = [3]
    ACTIVATION_CANDIDATES = ["GELU"]

# Verfuegbare Aktivierungen (weitere bei Bedarf hier ergaenzen)
ACTIVATIONS = {
    "Tanh":     nn.Tanh,
    "ReLU":     nn.ReLU,
    "GELU":     nn.GELU,
    "SiLU":     nn.SiLU,
    "ELU":      nn.ELU,
    "Softplus": nn.Softplus,
}

# 2x2 Gauss-Integration (Gewichte = 1)
_s3 = 1.0 / np.sqrt(3)
GAUSS_POINTS  = [(-_s3, -_s3), (_s3, -_s3), (_s3, _s3), (-_s3, _s3)]
GAUSS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# Indizes des oberen Dreiecks (column-major, identisch zu MATLAB find(triu(true(8))))
TRIU_IJ = [(i, j) for j in range(8) for i in range(j + 1)]   # 36 Paare
N_OUT = len(TRIU_IJ)

# ---------------------------------------------------------------------------
# Materialmatrix C (ebene Elastizitaet, Hooke)
# ---------------------------------------------------------------------------

def material_C(E, nu, condition):
    """3x3-Elastizitaetsmatrix fuer ebene Probleme (identisch material_elasticity.m)."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu  = E / (2 * (1 + nu))
    if condition.lower() == "planestress":
        c11 = lam / (nu - 1) + 2 * mu + 2 * lam
        c12 = lam / (nu - 1) + 2 * lam
    elif condition.lower() == "planestrain":
        c11 = lam + 2 * mu
        c12 = lam
    else:
        raise ValueError(f"Unbekannter Zustand: {condition}")
    return np.array([[c11, c12, 0.0],
                     [c12, c11, 0.0],
                     [0.0, 0.0, mu]])


C_MAT = material_C(E_TRAIN, NU_TRAIN, CONDITION)

# ---------------------------------------------------------------------------
# Analytische Referenz: Formfunktionsableitungen und Element-Steifigkeit
# ---------------------------------------------------------------------------

def shape_quad4_ref(coords, r, s):
    """Physikalische Formfunktionsableitungen h_x, h_y und detJ (isoparametrisch)."""
    x = coords[:, 0]
    y = coords[:, 1]

    h_r = 0.25 * np.array([-(1 - s),  (1 - s),  (1 + s), -(1 + s)])
    h_s = 0.25 * np.array([-(1 - r), -(1 + r),  (1 + r),  (1 - r)])

    x_r = x @ h_r;  x_s = x @ h_s
    y_r = y @ h_r;  y_s = y @ h_s

    detJ = x_r * y_s - y_r * x_s

    h_x = ( y_s * h_r - y_r * h_s) / detJ
    h_y = (-x_s * h_r + x_r * h_s) / detJ

    return h_x, h_y, detJ


def stiffness_ref(coords, C, d):
    """Analytische Element-Steifigkeitsmatrix Ke (8x8) per 2x2-Gauss-Integration."""
    Ke = np.zeros((8, 8))
    for (r, s), w in zip(GAUSS_POINTS, GAUSS_WEIGHTS):
        h_x, h_y, detJ = shape_quad4_ref(coords, r, s)
        B = np.zeros((3, 8))
        B[0, 0::2] = h_x
        B[1, 1::2] = h_y
        B[2, 0::2] = h_y
        B[2, 1::2] = h_x
        Ke += (B.T @ C @ B) * detJ * d * w
    return Ke


def compute_Lc(coords):
    centroid = coords.mean(axis=0)
    return float(np.mean(np.linalg.norm(coords - centroid, axis=1)))


def canonicalize_coords(coords):
    """Groessen- UND rotations-invariante Normalisierung ("Feature-Kanonisierung").

    Zuerst wie bisher auf Schwerpunkt und charakteristische Laenge Lc normieren.
    Zusaetzlich wird die Rotation aus der Eingabe entfernt: das Element wird so
    gedreht, dass die Kante Knoten1->Knoten2 auf der +x-Achse liegt. Dadurch
    bilden alle rotierten Varianten einer Form auf DIESELBE Eingabe ab -- die
    Eingangsmannigfaltigkeit verliert eine (Rotations-)Dimension, das Netz muss
    Rotationsequivarianz nicht mehr aus den Daten lernen. Das MATLAB-Element
    dreht die vorhergesagte Steifigkeit anschliessend exakt zurueck
    (Ke = Tc' * Khat_canon * Tc, Tc = blockdiag(Rc)).
    """
    centroid = coords.mean(axis=0)
    Lc = compute_Lc(coords)
    cn = (coords - centroid) / Lc
    e12 = cn[1] - cn[0]
    phi = np.arctan2(e12[1], e12[0])
    c, s = np.cos(phi), np.sin(phi)
    Rc = np.array([[c, s], [-s, c]])          # Drehung um -phi (Kante 1->2 -> +x)
    return cn @ Rc.T


def detJ_at_gp(coords):
    """detJ an allen Gausspunkten."""
    return np.array([shape_quad4_ref(coords, r, s)[2] for r, s in GAUSS_POINTS])


def interior_angles(coords):
    """Innenwinkel [Grad] der vier Ecken (Reihenfolge = Knotennummerierung)."""
    ang = np.empty(4)
    for i in range(4):
        u = coords[(i - 1) % 4] - coords[i]
        v = coords[(i + 1) % 4] - coords[i]
        cang = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-15)
        ang[i] = np.degrees(np.arccos(np.clip(cang, -1.0, 1.0)))
    return ang


def generate_distorted_quad():
    """Gezielt VERZERRTES, aber wohlgestelltes quad4-Element.

    Aufgebaut aus vier Verzerrungsquellen -- so wird der Formraum weit ueber
    "leicht gestoerte Rechtecke" hinaus abgedeckt (Trapeze, Scherung), ohne ins
    Entartete zu laufen:
      1. Seitenverhaeltnis  ax, ay   (log-uniform)
      2. Trapez-Taper t             (obere Kante t-fach breiter/schmaler als
                                     untere; |log t| gleichverteilt -> dichter
                                     Verzerrungs-Schwanz)
      3. Scherung g1, g2            (Parallelogramm/Skew)
      4. Knoten-Jitter             (kleine Zusatzstoerung)
    Eine harte Huelle (ENV_*) verwirft Elemente ausserhalb des wohlgestellten
    Gebiets (detJ-Verhaeltnis, Innenwinkel).
    """
    for _ in range(300):
        ax = np.exp(np.random.uniform(np.log(0.4), np.log(2.5)))
        ay = np.exp(np.random.uniform(np.log(0.4), np.log(2.5)))

        # Trapez-Taper: obere Kante um Faktor t skaliert; |log t| gleichverteilt,
        # damit auch stark tapernde Formen dicht vorkommen (nicht nur um t=1).
        logt = np.random.uniform(0, 1) * np.log(ENV_TAPER_MAX) * np.random.choice([-1.0, 1.0])
        t = np.exp(logt)
        wb, wt, H = ax, ax * t, ay
        coords = np.array([[-wb / 2, -H / 2], [wb / 2, -H / 2],
                           [wt / 2,  H / 2], [-wt / 2,  H / 2]])
        if np.random.rand() < 0.5:                 # Taper auch in die andere Achse
            coords = coords[:, ::-1].copy()

        g1 = np.random.uniform(-0.4, 0.4)          # Scherung
        g2 = np.random.uniform(-0.4, 0.4)
        coords = coords @ np.array([[1.0, g1], [g2, 1.0]]).T
        coords = coords + np.random.uniform(-0.08, 0.08, (4, 2))   # Jitter

        # CCW sicherstellen (positive Flaeche)
        area = 0.5 * np.sum(coords[:, 0] * np.roll(coords[:, 1], -1)
                            - np.roll(coords[:, 0], -1) * coords[:, 1])
        if area < 0:
            coords = coords[::-1].copy()

        dJ = detJ_at_gp(coords)
        if np.min(dJ) <= 1e-9:
            continue
        ratio = dJ.max() / dJ.min()
        ang = interior_angles(coords)
        if ratio > ENV_RATIO_MAX or ang.min() < ENV_ANGLE_MIN or ang.max() > ENV_ANGLE_MAX:
            continue
        return coords
    return None


# ---------------------------------------------------------------------------
# Datensatz (einmalig fuer alle Varianten)
# ---------------------------------------------------------------------------

def build_dataset(n_elements, label=""):
    inputs, targets, skipped = [], [], 0
    while len(inputs) < n_elements:
        coords = generate_distorted_quad()
        if coords is None:
            skipped += 1
            continue
        # Rotations- + groesseninvariante Kanonisierung: das Netz lernt Ke der
        # KANONISCH orientierten Form; das Element dreht spaeter exakt zurueck.
        coords_c = canonicalize_coords(coords)
        Ke = stiffness_ref(coords_c, C_MAT, D_TRAIN)

        coords_flat = coords_c.reshape(-1, order="C")
        ke_triu = np.array([Ke[i, j] for (i, j) in TRIU_IJ])

        inputs.append(coords_flat)
        targets.append(ke_triu)

    if skipped:
        log(f"  Verworfene Elemente ({label}): {skipped}")

    return (np.array(inputs,  dtype=np.float32),
            np.array(targets, dtype=np.float32))


log(f"Material: nu = {NU_TRAIN}, {CONDITION}  |  E = d = 1 (analytisch herausfaktorisiert)")
log(f"Generiere Trainingsdaten ({N_TRAIN_ELEMENTS} Elemente)...")
t0 = time.time()
X_train_np, Y_train_np = build_dataset(N_TRAIN_ELEMENTS, "train")
log(f"  -> {len(X_train_np)} Samples in {time.time()-t0:.1f} s")

log(f"Generiere Validierungsdaten ({N_VAL_ELEMENTS} Elemente)...")
t0 = time.time()
X_val_np, Y_val_np = build_dataset(N_VAL_ELEMENTS, "val")
log(f"  -> {len(X_val_np)} Samples in {time.time()-t0:.1f} s\n")

X_train = torch.tensor(X_train_np).to(device)
Y_train = torch.tensor(Y_train_np).to(device)
X_val   = torch.tensor(X_val_np).to(device)
Y_val   = torch.tensor(Y_val_np).to(device)

N_TRAIN = len(X_train)

# Per-Sample-relativer Frobenius-Loss:
#   L = mean_b  ||pred_b - tgt_b||^2 / ||tgt_b||^2
# Jedes Element wird durch seine EIGENE Ke-Norm normiert. Dadurch dominieren
# grosse (stark verzerrte) Matrizen den Loss nicht, kleine werden nicht
# ignoriert -- optimiert direkt die berichtete relative Frobenius-Metrik. Das
# ersetzt die fruehere globale Per-Dimension-Normierung (Y_std).
def rel_frob_loss(pred, tgt):
    num = ((pred - tgt) ** 2).sum(dim=1)
    den = (tgt ** 2).sum(dim=1).clamp_min(1e-12)
    return (num / den).mean()

# ---------------------------------------------------------------------------
# Netzwerk-Klasse   8 -> (FC-Act) x depth -> FC(36)
#   hidden : Neuronen je verdeckter Schicht
#   depth  : Anzahl verdeckter Schichten (Nichtlinearitaet)
#   act    : Aktivierungsfunktion (Name aus ACTIVATIONS)
# ---------------------------------------------------------------------------

class KMatrixNet(nn.Module):
    def __init__(self, hidden, depth, act_name):
        super().__init__()
        Act = ACTIVATIONS[act_name]
        layers = [nn.Linear(8, hidden), Act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), Act()]
        layers += [nn.Linear(hidden, N_OUT)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def forward_macs(hidden, depth):
    """Multiply-Accumulate-Operationen eines Forward-Pass (1 Element).
    Proxy fuer die Kosten pro Elementaufruf im FEM-Code."""
    macs = 8 * hidden                       # Eingangsschicht
    macs += (depth - 1) * hidden * hidden    # verdeckte Schichten
    macs += hidden * N_OUT                   # Ausgangsschicht
    return macs


def arch_tag(act_name, hidden, depth):
    return f"{act_name}_h{hidden}_d{depth}"

# ---------------------------------------------------------------------------
# Training einer Variante
# ---------------------------------------------------------------------------

def train_one(hidden, depth, act_name):
    model = KMatrixNet(hidden, depth, act_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=40, min_lr=1e-5)

    n_params = count_params(model)
    macs     = forward_macs(hidden, depth)
    log(f"\n--- {act_name} | Hidden = {hidden:3d} | Tiefe = {depth} "
        f"| Parameter: {n_params:,} | Forward-MACs: {macs:,} ---")
    arch = "-".join([f"FC({hidden})-{act_name}"] * depth + [f"FC({N_OUT})"])
    log(f"    Architektur: FC(8)-{arch}")
    batch_info = "voll" if FULL_BATCH else str(MINI_BATCH)
    log(f"    Epochs: {NUM_EPOCHS}  |  Batch: {batch_info}  |  LR: {LEARN_RATE}")

    num_batches = 1 if FULL_BATCH else int(np.ceil(N_TRAIN / MINI_BATCH))
    loss_train_hist, loss_val_hist = [], []
    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()

        if FULL_BATCH:
            # Ein exakter Gradientenschritt auf dem gesamten Datensatz --
            # keine Mini-Batch-Schleife, kein Per-Schritt-Sync.
            pred = model(X_train)
            loss = rel_frob_loss(pred, Y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_t = loss.detach()
        else:
            idx = torch.randperm(N_TRAIN, device=device)
            acc = torch.zeros((), device=device)          # auf der GPU akkumulieren
            for b in range(num_batches):
                sl = idx[b * MINI_BATCH : (b + 1) * MINI_BATCH]
                loss = rel_frob_loss(model(X_train[sl]), Y_train[sl])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc += loss.detach()                       # kein .item() im Hot-Path
            train_loss_t = acc / num_batches

        model.eval()
        with torch.no_grad():
            val_loss_t = rel_frob_loss(model(X_val), Y_val)

        # Nur EINMAL pro Epoche synchronisieren (statt pro Mini-Batch).
        val_loss = val_loss_t.item()
        loss_train_hist.append(train_loss_t.item())
        loss_val_hist.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 100 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            log(f"    Epoch {epoch:4d}/{NUM_EPOCHS} | "
                f"Train: {loss_train_hist[-1]:.3e} | "
                f"Val: {val_loss:.3e} | "
                f"LR: {lr_now:.1e} | "
                f"Zeit: {time.time()-t_start:.0f} s")

    log(f"    Fertig. Gesamtzeit: {time.time()-t_start:.1f} s")
    return model, loss_train_hist, loss_val_hist

# ---------------------------------------------------------------------------
# Auswertung
# ---------------------------------------------------------------------------

def triu_to_full(vec):
    K = np.zeros((8, 8))
    for v, (i, j) in zip(vec, TRIU_IJ):
        K[i, j] = K[j, i] = v
    return K


def evaluate(model):
    model.eval()
    with torch.no_grad():
        Y_pred = model(X_val).cpu().numpy()
    Y_ref = Y_val_np

    fro_err = np.array([np.linalg.norm(triu_to_full(p) - triu_to_full(r))
                        for p, r in zip(Y_pred, Y_ref)])
    fro_ref = np.array([np.linalg.norm(triu_to_full(r)) for r in Y_ref])
    rel_fro = fro_err / fro_ref * 100

    rmse_per_dim    = np.sqrt(np.mean((Y_pred - Y_ref) ** 2, axis=0))
    rms_ref_per_dim = np.sqrt(np.mean(Y_ref ** 2, axis=0))
    nrmse_per_dim   = rmse_per_dim / np.clip(rms_ref_per_dim, 1e-9, None) * 100

    return rel_fro, nrmse_per_dim, Y_pred

# ---------------------------------------------------------------------------
# Speichern (jede Variante separat in arch_sweep/ -- NICHT die Produktivdatei)
# ---------------------------------------------------------------------------

def build_mat_data(model, hidden, depth, act_name):
    """Gewichte + Metadaten fuer den MATLAB-Import (identisch fuer Sweep/Deploy)."""
    mat_data = {
        "input_order":     "row_major_xy_pairs_[x1_y1_x2_y2_x3_y3_x4_y4]",
        "normalization":   "centroid_Lc_mean_node_distance",
        # Feature-Kanonisierung: MATLAB muss die Eingabe GENAUSO drehen
        # (Kante 1->2 auf +x) und die Vorhersage zuruecktransformieren.
        "canonicalization": "edge_n1n2_to_pos_x_after_centroid_Lc",
        "output_order":    "upper_triangle_column_major_of_Khat_8x8",
        "nu_train":        float(NU_TRAIN),
        "condition":       CONDITION,
        "E_train":         float(E_TRAIN),
        "d_train":         float(D_TRAIN),
        "hidden":          float(hidden),
        "depth":           float(depth),
        "activation":      act_name,
        # dokumentierte Gueltigkeits-/Verzerrungshuelle des Datensatzes
        "env_ratio_max":   float(ENV_RATIO_MAX),
        "env_angle_min":   float(ENV_ANGLE_MIN),
        "env_angle_max":   float(ENV_ANGLE_MAX),
    }
    # Alle linearen Schichten generisch exportieren (W1/b1, W2/b2, ...)
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    mat_data["num_linear_layers"] = float(len(linears))
    for i, layer in enumerate(linears):
        mat_data[f"W{i+1}"] = layer.weight.detach().cpu().numpy()
        mat_data[f"b{i+1}"] = layer.bias.detach().cpu().numpy()
    return mat_data


def save_variant(model, hidden, depth, act_name):
    tag = arch_tag(act_name, hidden, depth)
    torch.save(model.state_dict(), sweep_dir / f"quad4_K_{tag}.pt")
    scipy.io.savemat(sweep_dir / f"quad4_K_{tag}.mat",
                     build_mat_data(model, hidden, depth, act_name))


PROD_MAT = element_dir / "quad4_K_network.mat"


def save_production(model, hidden, depth, act_name):
    """Schreibt die gewaehlte Variante als Produktivdatei (vom MATLAB-Element geladen)."""
    scipy.io.savemat(PROD_MAT, build_mat_data(model, hidden, depth, act_name))

# ---------------------------------------------------------------------------
# Sweep ueber das gesamte Gitter
# ---------------------------------------------------------------------------

grid = [(act, h, d)
        for act in ACTIVATION_CANDIDATES
        for d in DEPTH_CANDIDATES
        for h in HIDDEN_CANDIDATES]

log("\n" + "=" * 72)
log(f"  Architektur-Sweep: {len(ACTIVATION_CANDIDATES)} Aktivierungen "
    f"x {len(DEPTH_CANDIDATES)} Tiefen x {len(HIDDEN_CANDIDATES)} Breiten "
    f"= {len(grid)} Netze")
log(f"  Aktivierungen: {ACTIVATION_CANDIDATES}")
log(f"  Tiefen:        {DEPTH_CANDIDATES}")
log(f"  Breiten:       {HIDDEN_CANDIDATES}")
log("=" * 72)

results = []
t_sweep = time.time()
for n, (act, h, d) in enumerate(grid, start=1):
    log(f"\n>>> Variante {n}/{len(grid)}")
    model, train_hist, val_hist = train_one(h, d, act)
    rel_fro, nrmse, Y_pred = evaluate(model)
    save_variant(model, h, d, act)
    results.append({
        'act':        act,
        'hidden':     h,
        'depth':      d,
        'tag':        arch_tag(act, h, d),
        'model':      model,
        'train_hist': train_hist,
        'val_hist':   val_hist,
        'rel_fro':    rel_fro,
        'nrmse':      nrmse,
        'Y_pred':     Y_pred,
        'n_params':   count_params(model),
        'macs':       forward_macs(h, d),
        'mean_e':     float(rel_fro.mean()),
        'p95_e':      float(np.percentile(rel_fro, 95)),
    })

log(f"\nSweep abgeschlossen. Gesamtzeit: {time.time()-t_sweep:.1f} s "
    f"({len(grid)} Netze)")

# ---------------------------------------------------------------------------
# Vergleichstabelle (nach Genauigkeit sortiert) + Empfehlungen
# ---------------------------------------------------------------------------

by_acc = sorted(results, key=lambda r: r['mean_e'])

log("\n" + "=" * 86)
log("  MODELLVERGLEICH  (sortiert nach mittlerem Frobenius-Fehler)")
log("-" * 86)
log(f"  {'Rang':>4} | {'Aktivierung':>11} | {'Breite':>6} | {'Tiefe':>5} | "
    f"{'Param':>7} | {'MACs':>7} | {'Ø Fro':>8} | {'95.Pz':>7} | {'OK?':>4}")
log("-" * 86)
for rank, r in enumerate(by_acc, start=1):
    ok = r['mean_e'] < ACCURACY_THRESHOLD
    log(f"  {rank:>4} | {r['act']:>11} | {r['hidden']:>6} | {r['depth']:>5} | "
        f"{r['n_params']:>7,} | {r['macs']:>7,} | {r['mean_e']:>7.3f}% | "
        f"{r['p95_e']:>6.3f}% | {'JA' if ok else 'nein':>4}")
log("=" * 86)

# Empfehlung: billigste (wenigste Forward-MACs) Variante unter dem Schwellwert.
acceptable = [r for r in results if r['mean_e'] < ACCURACY_THRESHOLD]
most_accurate = by_acc[0]

if acceptable:
    cheapest = min(acceptable, key=lambda r: (r['macs'], r['mean_e']))
    log(f"\nEMPFEHLUNG (billigste Variante < {ACCURACY_THRESHOLD}% Fehler):")
    log(f"  -> {cheapest['act']}, Breite {cheapest['hidden']}, "
        f"Tiefe {cheapest['depth']}  "
        f"({cheapest['n_params']:,} Param, {cheapest['macs']:,} MACs, "
        f"Ø {cheapest['mean_e']:.3f}% Fehler)")
else:
    log(f"\nKeine Variante unter {ACCURACY_THRESHOLD}% Schwellwert.")

log(f"\nGENAUESTE Variante (unabhaengig von den Kosten):")
log(f"  -> {most_accurate['act']}, Breite {most_accurate['hidden']}, "
    f"Tiefe {most_accurate['depth']}  "
    f"({most_accurate['n_params']:,} Param, {most_accurate['macs']:,} MACs, "
    f"Ø {most_accurate['mean_e']:.3f}% Fehler)")

# ---------------------------------------------------------------------------
# Deployment in die Produktivdatei quad4_K_network.mat
#   Das MATLAB-Element rechnet mit GELU -> nur GELU-Varianten sind einsetzbar.
#   Deployt wird die billigste GELU-Variante unter dem Schwellwert (sonst die
#   genaueste GELU-Variante ueberhaupt).
# ---------------------------------------------------------------------------
if DEPLOY_BEST:
    gelu_ok = [r for r in results if r['act'] == 'GELU' and r['mean_e'] < ACCURACY_THRESHOLD]
    gelu_all = [r for r in results if r['act'] == 'GELU']
    pool = gelu_ok if gelu_ok else gelu_all
    if pool:
        dep = min(pool, key=lambda r: (r['macs'], r['mean_e']))
        save_production(dep['model'], dep['hidden'], dep['depth'], 'GELU')
        log(f"\nDEPLOYT nach {PROD_MAT.name}:  GELU, Breite {dep['hidden']}, "
            f"Tiefe {dep['depth']}  ({dep['n_params']:,} Param, "
            f"Ø {dep['mean_e']:.3f}% Fehler)")
    else:
        log("\nWARNUNG: keine GELU-Variante trainiert -> kein Deployment.")

# Achsen-Analyse: bestes Mittel je Aktivierung / je Tiefe / je Breite
log("\nMittlerer Fehler je Aktivierung (ueber alle Breiten/Tiefen):")
for act in ACTIVATION_CANDIDATES:
    sub = [r['mean_e'] for r in results if r['act'] == act]
    log(f"  {act:>11}:  Ø {np.mean(sub):.3f}%  (best {np.min(sub):.3f}%)")

log("\nMittlerer Fehler je Tiefe (Nichtlinearitaet):")
for d in DEPTH_CANDIDATES:
    sub = [r['mean_e'] for r in results if r['depth'] == d]
    log(f"  Tiefe {d}:  Ø {np.mean(sub):.3f}%  (best {np.min(sub):.3f}%)")

log("\nMittlerer Fehler je Breite (Neuronen):")
for h in HIDDEN_CANDIDATES:
    sub = [r['mean_e'] for r in results if r['hidden'] == h]
    log(f"  Breite {h:>3}:  Ø {np.mean(sub):.3f}%  (best {np.min(sub):.3f}%)")

log("\nHINWEIS: element_quad4_lin_ai.m ist auf GELU + 3 verdeckte Schichten "
    "festgelegt und nutzt\n         dieselbe Kanonisierung (Kante 1->2 -> +x). "
    "Ein Gewinner mit anderer Aktivierung/Tiefe\n         muss vor dem Einsatz "
    "dort angepasst werden; die Hidden-Groesse wird dynamisch gelesen.")

# ---------------------------------------------------------------------------
# Vergleichsplots
# ---------------------------------------------------------------------------

act_colors    = {a: plt.cm.tab10.colors[i] for i, a in enumerate(ACTIVATION_CANDIDATES)}
depth_ls      = {1: ":", 2: "--", 3: "-", 4: "-."}
depth_marker  = {1: "o", 2: "s", 3: "D", 4: "^"}

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("quad4 K-Matrix Network: Architektur-Sweep "
             "(Neuronen x Tiefe x Aktivierung)", fontsize=13)

# (a) Validierungs-Loss-Verlauf aller Varianten
ax = axes[0, 0]
for r in results:
    ax.semilogy(r['val_hist'], color=act_colors[r['act']],
                linestyle=depth_ls.get(r['depth'], "-"), alpha=0.7, linewidth=1.0)
ax.set_xlabel("Epoch"); ax.set_ylabel("Val MSE Loss")
ax.set_title("Trainingsverlauf (Validierung)")
ax.grid(True, which="both", alpha=0.3)
# zwei kompakte Legenden: Farbe = Aktivierung, Linienstil = Tiefe
h_act = [plt.Line2D([], [], color=act_colors[a], label=a) for a in ACTIVATION_CANDIDATES]
h_dep = [plt.Line2D([], [], color="k", linestyle=depth_ls.get(d, "-"),
                    label=f"Tiefe {d}") for d in DEPTH_CANDIDATES]
leg1 = ax.legend(handles=h_act, fontsize=8, loc="upper right", title="Aktivierung")
ax.add_artist(leg1)
ax.legend(handles=h_dep, fontsize=8, loc="lower left", title="Tiefe")

# (b) Genauigkeit ueber Breite, je (Aktivierung, Tiefe) eine Linie
ax = axes[0, 1]
for act in ACTIVATION_CANDIDATES:
    for d in DEPTH_CANDIDATES:
        sub = sorted([r for r in results if r['act'] == act and r['depth'] == d],
                     key=lambda r: r['hidden'])
        if not sub:
            continue
        ax.plot([r['hidden'] for r in sub], [r['mean_e'] for r in sub],
                color=act_colors[act], linestyle=depth_ls.get(d, "-"),
                marker=depth_marker.get(d, "o"), markersize=5,
                label=f"{act} d{d}")
ax.axhline(ACCURACY_THRESHOLD, color="k", linestyle=":", linewidth=1.5)
ax.set_xlabel("Neuronen je Schicht (Breite)")
ax.set_ylabel("Ø Frobenius-Fehler [%]")
ax.set_title("Genauigkeit vs. Breite")
ax.set_xticks(HIDDEN_CANDIDATES)
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# (c) Pareto: Genauigkeit vs. Kosten (Forward-MACs)
ax = axes[1, 0]
for r in results:
    ax.scatter(r['macs'], r['mean_e'], s=70,
               color=act_colors[r['act']], marker=depth_marker.get(r['depth'], "o"),
               edgecolor="k", linewidth=0.4, zorder=3)
ax.axhline(ACCURACY_THRESHOLD, color="k", linestyle=":", linewidth=1.5,
           label=f"Schwellwert {ACCURACY_THRESHOLD}%")
ax.set_xlabel("Forward-MACs pro Element (Kosten-Proxy)")
ax.set_ylabel("Ø Frobenius-Fehler [%]")
ax.set_title("Pareto: Genauigkeit vs. Kosten  (links-unten = besser)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# (d) Balken: mittlerer Fehler je Variante (sortiert), eingefaerbt nach Aktivierung
ax = axes[1, 1]
order = sorted(results, key=lambda r: r['mean_e'])
labels = [r['tag'] for r in order]
ax.bar(range(len(order)), [r['mean_e'] for r in order],
       color=[act_colors[r['act']] for r in order])
ax.axhline(ACCURACY_THRESHOLD, color="k", linestyle=":", linewidth=1.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, rotation=90, fontsize=6)
ax.set_ylabel("Ø Frobenius-Fehler [%]")
ax.set_title("Fehler je Architektur (sortiert)")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plot_path = sweep_dir / f"quad4_arch_sweep_{_stamp}.png"
plt.savefig(plot_path, dpi=150)
log(f"\nPlot gespeichert: {plot_path}")
log(f"Log  gespeichert: {log_path}")
log("\nFertig.")
