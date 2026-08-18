# Deep Learned Finite Elements: Das quad4-Element (linear und nichtlinear)

**Aktueller Stand** (zuletzt inhaltlich geändert: 2026-08-18). Dieses Dokument
beschreibt den KI-basierten Ansatz für das bilineare Viereckselement `quad4`
in **zwei Varianten**:

* **linear** — ein neuronales Netz sagt die Element-Steifigkeitsmatrix `Ke`
  direkt aus der Geometrie voraus,
* **nichtlinear** (Total Lagrange, St.-Venant-Kirchhoff) — ein Netz lernt die
  **Formänderungsenergie** aus Geometrie **und** Verschiebungszustand;
  tangentiale Steifigkeit `Ke` und innerer Kraftvektor `Finte` entstehen
  daraus durch Differentiation (Residual-Energie mit analytischem K₀-Split).

Der Fokus liegt bewusst auf der **angewandten Mathematik der KI** (Abschnitt 1)
und erst danach auf der konkreten Anwendung auf das Element (Abschnitte 2–3)
und dessen Implementierung (Abschnitt 4). Die klassische FEM-Theorie (B-Matrix,
Gauss-Integration, Total-Lagrange-Kinematik) wird als bekannt vorausgesetzt und
nur soweit erwähnt, wie sie erklärt, *was* das Netz ersetzt.

Weitere Dokumente:

* `DLFE_quad4_Entwicklung.md` — **der Weg hierher**: welche Probleme auftraten
  (44 %-Falle, Verteilungs-Lücke, Trainingskollaps), der ursprüngliche Plan
  (Option A vs. B), das Debug-Protokoll vom 15.07. und der Neuentwurf als
  Residual-Energie-Netz (Teil IV).
* `DLFE_quad4_nl_plan.md` — Plan und Umsetzungsstand des Residual-Energie-
  Netzes inklusive vollständiger Gate-Tabelle.
* `FEMSolid_quad4_benchmark_structures.md` — die 10 linearen
  Benchmark-Strukturen im Detail.

Zugehörige Dateien:

| Datei | Rolle |
|-------|-------|
| `training/quad4/train_quad4_K_network.py` | Training linear, schreibt `quad4_K_network.mat` |
| `training/quad4/train_quad4_nl_W_network.py` | Training nichtlinear, schreibt `quad4_nl_W_network.mat` |
| `training/quad4/quad4_nl_ref.py` | Referenzmathematik (Energie/F/K, K₀, Kette) + Gates a/b |
| `training/quad4/generate_newton_trajectories.m` | Trainingsdaten aus echten Newton-Läufen |
| `sourcecode/elements/quad4/element_quad4_lin_ai.m` | MATLAB-Element linear: ein Forward-Pass → `Ke` |
| `sourcecode/elements/quad4/element_quad4_nl_ai.m` | MATLAB-Element nichtlinear (Signatur wie `element_quad4_nl.m`) |
| `sourcecode/elements/quad4/quad4_nl_ai_energy.m` | Kette: Kanonisierung, Ko-Rotation, exakte 1./2. Ableitungen |
| `sourcecode/elements/quad4/quad4_nl_ai_model.m` | Modell: K₀-Split + Residual-Netz (Wert, Gradient, Hessian) |
| `sourcecode/elements/quad4/quad4_K_network.mat`, `quad4_nl_W_network.mat` | deployte Netze (Gewichte + Metadaten) |
| `examples/FEMSolid_ex_quad4_03_ai.m` | direkter Ke-Vergleich analytisch vs. gelernt (linear) |
| `examples/FEMSolid_ex_quad4_05_ai_benchmark.m` | 10 Strukturen, FEM vs. KI (linear) |
| `examples/FEMSolid_ex_quad4_06_ai_patch_distortion.m` | Verzerrungs-Patch-Test (linear) |
| `examples/FEMSolid_ex_quad4_07_ai_nl_benchmark.m` | 5 nichtlineare Strukturen, FEM vs. KI (Gates f/g/h) |
| `examples/FEMSolid_ex_quad4_08_ai_nl_check.m` | Einzelelement-Check ohne Solver (Hüllen-Diagnose, nichtlinear) |
| `examples/FEMSolid_ex_quad4_09_ai_nl_consistency.m` | Konsistenz- und Ketten-Verifikation per FD (Gates d/e) |

> Abgelöst, aber noch vorhanden: `train_quad4_nl_K_network.py` und
> `quad4_nl_K_network.mat` (Zwei-Kopf-Variante). Sie werden von keinem Element
> mehr gelesen.

---

## 1. Mathematische Grundlagen des KI-Ansatzes

### 1.1 Das Netz als Funktionsapproximator

Beide Varianten verwenden ein einfaches **vollverbundenes Feedforward-Netz**
(Multilayer Perceptron, MLP): eine Verkettung affiner Abbildungen und
elementweiser Nichtlinearitäten,

```
a⁽⁰⁾ = x
z⁽ˡ⁾ = W⁽ˡ⁾ a⁽ˡ⁻¹⁾ + b⁽ˡ⁾                 (affine Schicht l)
a⁽ˡ⁾ = σ(z⁽ˡ⁾)     für l = 1..L-1          (verdeckte Schichten, elementweise)
a⁽ᴸ⁾ = z⁽ᴸ⁾                                 (Ausgabeschicht: linear, keine Aktivierung)
```

`x ∈ ℝⁱⁿ` ist der Eingabevektor (Geometrie- bzw. Geometrie+Verschiebungs-Merkmale,
s. u.), `a⁽ᴸ⁾ ∈ ℝᵒᵘᵗ` die Netzausgabe. Die Gewichte `W⁽ˡ⁾` und Biase `b⁽ˡ⁾` sind
die einzigen freien Parameter; ihre Anzahl (Breite × Tiefe) bestimmt die
Kapazität des Netzes. Ein MLP mit nichtlinearer Aktivierung ist ein
**universeller Funktionsapproximator** — die Grundannahme ist, dass sich die
(im Fall des linearen Elements exakt polynomiale, im nichtlinearen Fall glatte,
aber komplizierte) Abbildung Geometrie → `Ke` bzw. (Geometrie, Zustand) →
`(Ke, Finte)` durch ein hinreichend großes MLP beliebig genau approximieren
lässt.

### 1.2 Aktivierungsfunktion: GELU

Beide Netze verwenden **GELU** (Gaussian Error Linear Unit) in der exakten
Form

```
GELU(z) = z · Φ(z) = 0.5 · z · (1 + erf(z / √2))
```

mit `Φ` = Verteilungsfunktion der Standardnormalverteilung. GELU ist glatt
(beliebig oft differenzierbar, im Gegensatz zu ReLU) und in der Praxis für
kleine, glatte Regressionsprobleme wie dieses robuster als ReLU/Tanh
(empirisch durch einen Architektur-Sweep über Aktivierung × Breite × Tiefe
bestätigt). Die exakte `erf`-Form ist wichtig, weil MATLAB zur Laufzeit
**dieselbe** Formel verwendet wie PyTorch beim Training (`nn.GELU()`,
Default `approximate='none'`) — nur so stimmen trainiertes Netz und
Forward-Pass im Element exakt überein (gegen `.pt` verifiziert).

### 1.3 Trainingsziel: Verlustfunktion und Optimierung

Beide Netze werden **überwacht** (supervised regression) trainiert: ein
Datensatz `{(xᵢ, yᵢ)}` aus Eingaben und analytisch berechneten Zielwerten wird
erzeugt (Abschnitte 2.4/3.6), und die Netzparameter werden so optimiert, dass
die Vorhersage `ŷᵢ = f_θ(xᵢ)` den Zielwert `yᵢ` möglichst gut trifft.

Zentrale Besonderheit gegenüber einem naiven MSE-Loss: die Zielgrößen
(`Ke`-Einträge, `Finte`-Komponenten) unterscheiden sich zwischen Samples um
Größenordnungen (z. B. `Finte → 0` für kleine Verschiebungen). Ein absoluter
mittlerer quadratischer Fehler würde daher fast ausschließlich die großen
Zustände optimieren. Stattdessen wird ein **pro Sample relativer Verlust**
verwendet:

```
L = mean_i ( ‖ŷᵢ − yᵢ‖² / (‖yᵢ‖² + ε) )
```

Jedes Sample wird durch die **eigene** Norm geteilt — das optimiert direkt
den später berichteten relativen Fehler, unabhängig vom Lastniveau. Der
kleine Boden `ε` verhindert eine Singularität bei `yᵢ → 0` (Nulllast).

Die Optimierung erfolgt mit **Adam** (adaptiver stochastischer Gradientenabstieg
mit Momentum-Schätzung erster und zweiter Ordnung) und einer
**Plateau-Lernraten-Reduktion** (`ReduceLROnPlateau`: Lernrate wird halbiert,
wenn der Validierungsverlust über mehrere hundert Epochen nicht mehr fällt).
Da die Datensätze klein genug sind, um komplett in den GPU-Speicher zu passen,
wird **Voll-Batch-Training** verwendet (ein Optimierungsschritt pro Epoche über
den gesamten Datensatz) — das ist für ein derart kleines Netz deutlich
schneller als Mini-Batches, weil Letztere bei so wenigen Parametern durch den
Python/CUDA-Overhead pro Schritt dominiert würden (gemessen linear:
~450 s → ~22 s).

### 1.4 Ausnutzung bekannter Invarianzen: Kanonisierung

Der methodisch wichtigste Kunstgriff ist **nicht** die Netzarchitektur, sondern
die **Vorverarbeitung der Eingabe**. Die Zielfunktion (Geometrie → `Ke` bzw.
(Geometrie, Zustand) → `(Ke, Finte)`) besitzt bekannte Symmetrien:

* **Translationsinvarianz** — nur die *Form*, nicht die Lage im Raum, ist
  relevant.
* **Skaleninvarianz** — in der ebenen Elastizität ist `Ke` unabhängig von der
  absoluten Elementgröße (nur von der Form).
* **Rotationsäquivarianz** — dreht man das Element um `R`, transformiert sich
  `Ke` exakt mit: `Ke(R∘Geometrie) = T_R · Ke(Geometrie) · T_Rᵀ`.

Anstatt dem Netz zu überlassen, diese Symmetrien **approximativ aus Daten** zu
lernen (was Rotationsaugmentation und mehr Kapazität erfordern würde, ohne die
Symmetrie je exakt zu erfüllen), werden sie **exakt vorab herausgerechnet**:
Jede Eingabe wird auf einen **kanonischen Repräsentanten** ihrer
Äquivalenzklasse abgebildet (Schwerpunkt subtrahieren, durch charakteristische
Länge teilen, auf eine feste Referenzorientierung drehen), das Netz lernt nur
noch auf kanonischen Eingaben, und die Vorhersage wird anschließend **exakt**
zurücktransformiert. Dieses Prinzip — bekannte Invarianzen der Zielfunktion in
eine deterministische Vor-/Nachverarbeitung statt in die Lernaufgabe zu
verlagern — ist der Kern dessen, was in Abschnitt 1.5 als „harte
Nebenbedingung" bezeichnet wird, und reduziert zugleich die effektive
Eingabedimension (weniger unabhängige Freiheitsgrade → weniger Trainingsdaten
nötig, bessere Generalisierung).

### 1.5 Harte vs. weiche Nebenbedingungen

Bei beiden Elementen werden physikalische Eigenschaften auf zwei
unterschiedliche Arten in das Modell eingebracht:

| Art | Beispiel | Umsetzung |
|---|---|---|
| **Harte Nebenbedingung** (strukturell exakt, nicht gelernt) | Symmetrie von `Ke`; Kräftegleichgewicht von `Finte`; Translationsnullraum von `Ke`; Rotations-/Skaleninvarianz | Architektur/Vor-Nachverarbeitung erzwingt die Eigenschaft **exakt**, unabhängig von den Gewichten |
| **Weiche Nebenbedingung** (approximativ, über den Verlust) | Genauigkeit der `Ke`-/`Finte`-Werte selbst; (nichtlinear) Konsistenz zwischen `Ke` und `Finte` | Nur über die Trainingsdaten und den Verlust angenähert, mit Restfehler |

Wo immer eine Eigenschaft **exakt aus der Struktur des Problems folgt**
(Symmetrie, Gleichgewicht, Invarianz), wird sie hart erzwungen — das eliminiert
genau die Fehlerart, die numerisch am gefährlichsten ist (siehe Abschnitt 2.1,
Starrkörper-Projektion). Nur Eigenschaften, die selbst für die *exakte*
Lösung nicht in geschlossener Form vorliegen (der tatsächliche Zahlenwert von
`Ke`/`Finte`), werden dem Lernprozess überlassen.

---

## 2. Das lineare quad4-Element

Datei: `train_quad4_K_network.py` (Training) / `element_quad4_lin_ai.m`
(Laufzeit) / `quad4_K_network.mat` (deploytes Netz, ~16 KB).

### 2.1 Zu lernende Größe

Klassisch: `Ke = ∫_Ve Bᵀ C B dV` (8×8, symmetrisch, per Gauss-Integration).
Das Netz ersetzt diese Integration durch einen einzigen Forward-Pass:

```
Ke = f_θ(Geometrie)
```

**Warum das gutartig ist:** `Ke` hängt bei linearer Elastizität *nicht* vom
Lastzustand ab — die Zielfunktion ist zeitunabhängig, muss also nur **einmal**
pro Element ausgewertet werden.

**Kritischer Punkt (harte vs. weiche Nebenbedingung, Abschnitt 1.5):** Die
analytische `Ke` besitzt einen *exakten* Nullraum aus drei Starrkörpermoden
(2 Translationen + 1 Rotation). Ein gelerntes `Ke` mit nur 1–2 % Frobenius-Fehler
erfüllt das lediglich näherungsweise — das erzeugt beim Lösen des
Gleichungssystems eine **spurious Verspannung**, die sich massiv verstärkt
(gemessen: 44 % Verschiebungsfehler bei 1–2 % Matrixfehler). Da die drei Moden
aus der Geometrie exakt bekannt sind, wird der Nullraum **hart** erzwungen:

```
Ke ← P · Ke · P ,     P = I − Q·Qᵀ
```

mit `Q` = orthonormale Basis der drei Moden (hier sogar ohne SVD berechenbar,
da die Moden bereits paarweise orthogonal sind — reine Spaltennormierung).
Ergebnis: Verschiebungsfehler 44 % → **~1,3 % Mittel** (Median 0,6 %).
Ohne diese Projektion ist der Ansatz unbrauchbar.

### 2.2 Kanonisierung der Geometrie

```
centroid = Mittelwert der 4 Knoten
Lc       = mittlerer Abstand der Knoten zum Schwerpunkt
ĉ        = Rc · (coord − centroid) / Lc
```

`Rc` ist die Drehmatrix, die die Kante Knoten1→Knoten2 auf die `+x`-Achse
dreht. Rücktransformation exakt (verifiziert ~1e-16):

```
Ke = Tc' · Khat_canon · Tc ,   Tc = blockdiag(Rc, Rc, Rc, Rc)
```

Zusätzlich wird die **Linearität in `E·d`** ausgenutzt: `Ke(E,d) = E·d·Khat`.
Trainiert wird mit `E = d = 1`; das Element multipliziert mit den realen
Werten. `E` und Dicke `d` sind damit **frei**, Querkontraktionszahl `ν = 0.3`
und ebener Zustand (`planeStrain`) sind **fest** eintrainiert.

### 2.3 Netzarchitektur und Ein-/Ausgabe

```
Eingang  (8):  [x1_c, y1_c, ..., x4_c, y4_c]     kanonisierte Knotenkoordinaten
Ausgang (36):  oberes Dreieck (column-major) von Khat
```

**Architektur:** GELU, 3 verdeckte Schichten à 32 Neuronen
(`FC(8)→32→32→32→FC(36)`), ≈ 4 700 Parameter. Die volle symmetrische 8×8 wird
aus den 36 Werten per Index-Tabelle (ein Gather, identisch zu MATLAB
`find(triu(true(8)))`) rekonstruiert — Symmetrie ist damit strukturell exakt,
nicht nur approximativ gelernt.

### 2.4 Trainingsdatensatz

`generate_distorted_quad()` erzeugt gezielt **verzerrte** (nicht nur
rechteckige) Vierecke: zufälliges Seitenverhältnis, Trapez-Taper, Scherung,
Knoten-Jitter. Eine harte Hülle definiert den Gültigkeitsbereich und verwirft
(Rejection Sampling), was sie verletzt:

* `detJ > 0` an allen Gausspunkten,
* `max(detJ)/min(detJ) ≤ 4.5`,
* alle Innenwinkel ∈ `[20°, 160°]`.

Umfang: 20 000 Trainings- / 2 000 Validierungsgeometrien. Jede Geometrie wird
vor der Zielwert-Berechnung kanonisiert (2.2); die Targets `Ke` kommen aus
einem Python-Port des analytischen Elements.

### 2.5 Verlustfunktion

Pro-Sample-relativer Frobenius-Verlust (Abschnitt 1.3), ohne weitere
Sonderbehandlung — die Zielgrößen (`Ke`-Einträge) sind bereits alle von
ähnlicher Größenordnung, im Gegensatz zum nichtlinearen Fall (Abschnitt 3.7).

### 2.6 Training-Pipeline

`train_quad4_K_network.py` in einem Lauf:

1. Verzerrungs-getriebenen Datensatz erzeugen (20 000 train / 2 000 val,
   wohlgestellte Hülle).
2. Architektur-Sweep (Breite × Tiefe × Aktivierung), Voll-Batch.
3. Jede Variante nach `arch_sweep/` speichern, Vergleichslog + Plots.
4. Billigste GELU-Variante unter dem Fehler-Schwellwert nach
   `quad4_K_network.mat` deployen.

Schnelllauf zum Pipeline-Test: Umgebungsvariable `QUAD4_QUICK` setzen
(nur eine GELU-Variante).

### 2.7 Ergebnis

* **Validierung** (harter, verzerrter Satz): Ø **0,35 %** Frobenius-Fehler
  (95. Perzentil 0,78 %).
* **Verschiebungs-Benchmark** (10 Strukturen, FEM vs. KI): **~1,3 %** Mittel
  dank Starrkörper-Projektion. Assemblierung je nach Struktur ~Faktor 2
  schneller als das analytische Element (Abschnitt 4.4).
* **Patch-Test** (relativer Frobenius je Element, r = Verschiebung des
  Mittelknotens):

  | r | E1 | E2 | E3 | E4 |
  |------|------|------|-------|------|
  | 0.00 | 0.10 | 0.10 | 0.10 | 0.10 |
  | 0.30 | 0.22 | 0.24 | 0.38 | 0.26 |
  | 0.50 | 0.38 | 0.62 | 1.20 | 0.41 |
  | 0.60 | 0.48 | 0.88 | 4.83 | 0.51 |
  | 0.70 | 0.60 | 1.25 | 11.68 | 0.65 |

  E3 wandert am schnellsten aus der Trainingshülle (max. Innenwinkel ~179° bei
  r≈0.7, near-degeneriert) und divergiert dort erwartungsgemäß — außerhalb des
  Gültigkeitsbereichs.

**Gültigkeitsbereich:** `ν = 0,3`, `planeStrain` fest; `E`, `d` frei;
Geometrie nur innerhalb der Verzerrungshülle aus 2.4 (jenseits davon,
detJ→0, wird die wahre `Ke` singulär — bewusst ausgeschlossen).

---

## 3. Das nichtlineare quad4-Element (Residual-Energie-Netz)

Dateien: `training/quad4/train_quad4_nl_W_network.py` (Training) /
`element_quad4_nl_ai.m` + `quad4_nl_ai_energy.m` + `quad4_nl_ai_model.m`
(Laufzeit) / `quad4_nl_W_network.mat` (deploytes Netz).
Materialgesetz: St.-Venant-Kirchhoff, Total Lagrange.

> **Modellwechsel (2026-08-18).** Die frühere Variante („Option B": zwei
> unabhängige Netzköpfe für `Ke` und `Finte`, `quad4_nl_K_network.mat`) ist
> **abgelöst**. Sie lieferte richtige Lösungen, aber `Ke` war nur
> näherungsweise die Jacobi-Matrix von `Finte` (gemessen 7,1 %), wodurch
> Newton nur linear konvergierte (Kontraktionsrate 0,902) und das
> Iterationslimit erreichte. Das aktuelle Modell lernt stattdessen ein
> **Energiepotential**; `Finte` und `Ke` entstehen durch Differentiation, die
> Konsistenz gilt damit **per Konstruktion**. Der Weg dorthin steht in
> `DLFE_quad4_Entwicklung.md`, Teil IV; die Historie der Zwei-Kopf-Variante
> in Teil III.

### 3.0 Das Modell in einer Formel

```
What(ĉ, z) = ½·zᵀ K̂₀(ĉ) z  +  Ŵ_NL(ĉ, z)
F̂ = ∂What/∂z  = K̂₀z + ∇Ŵ_NL          K̂ = ∂²What/∂z² = K̂₀ + ∇²Ŵ_NL
```

* `K̂₀(ĉ)` — **exakte lineare Steifigkeit** der kanonischen Geometrie,
  analytisch pro Elementaufruf berechnet (4-Gausspunkt-Schleife mit linearer
  B-Matrix, ohne Doppelknotenschleife). **Kein Cache**: `assemble.m` bleibt
  zustandslos, Solver-Signaturen unangetastet.
* `Ŵ_NL` — kleines neuronales Netz (GELU-MLP, **skalarer** Ausgang), das
  **nur die nichtlineare Energieabweichung** lernt. Für StVenant ist `What`
  exakt ein Polynom 4. Grades in `z`; der K₀-Split entfernt den quadratischen
  Term, das Netz sieht nur den kubisch/quartischen Rest.

Vier Eigenschaften folgen **strukturell**, nicht aus dem Training:

| Eigenschaft | Grund | gemessen |
|---|---|---|
| `Ke = ∂Finte/∂Ue` (Konsistenz) | beides aus einem Potential | 8,8e-10 |
| `Ke` symmetrisch | Hessematrix eines Skalars | Maschinengenauigkeit |
| `Finte = 0` bei Starrkörperbewegung | Ko-Rotation ⇒ `z = 0`, Subtraktionsform ⇒ `F̂(ĉ,0) = 0` | 1,4e-15 |
| `Σᵢ Finte_i = 0`, Translationsnullraum von `Ke` | Translationsprojektor `P` in der Kette | exakt |

Der praktische Effekt: Newton braucht mit dem KI-Element **dieselbe
Iterationszahl wie mit dem analytischen Element** — der Kernbefund des
Neuentwurfs.

**Was das für die Fehlermetriken bedeutet.** Die Konsistenz verschiebt, welcher
Fehler überhaupt zählt. `Ke` ist exakt die Jacobi-Matrix der *gelernten*
`Finte`-Funktion; Newton konvergiert also quadratisch gegen die Lösung von
„gelernte `Finte` = `Fext`". Damit gilt:

* Der **`Finte`-Fehler bestimmt die Lösung** (`dU`, `dVM`) — er ist die
  kritische Größe.
* Der **`Ke`-Fehler gegenüber der wahren Tangente beeinflusst die Konvergenz
  nicht mehr**. Er ist eine Qualitätsanzeige der zweiten Ableitung (und über
  den Sobolev-Verlust ein Regularisierer, der auch `Finte` verbessert), aber
  kein Konvergenzrisiko.

Bei der Zwei-Kopf-Variante war das anders: dort ging der Ke-Fehler *doppelt*
ein — über die Lösung und über die Inkonsistenz in die Konvergenzrate.

Warum der K₀-Split mehr ist als eine Beschleunigung: er trifft genau das
Regime, an dem die Vorgängerversion scheiterte. Bei kleinen Amplituden
(Newton-Endphase) geht der gelernte Anteil gegen null, und `Finte` wird von
`K̂₀z` exakt dominiert — dort, wo die Zwei-Kopf-Variante 40–700 % relativen
Fehler hatte.

### 3.1 Unterschied zum linearen Fall

Der Newton-Löser braucht in **jeder Iteration jedes Lastschritts** aus jedem
Element zwei Größen, die jetzt beide vom **Verschiebungszustand** abhängen:

```
R(U) = Fint(U) − λ·Fext = 0        (Gleichgewicht/Residuum)
dU   = −Ke(U)⁻¹ · R(U)              (Newton-Korrektur)
```

1. `Finte(coord, Ue)` — bildet das Residuum,
2. `Ke(coord, Ue)` — die tangentiale Steifigkeit für die Korrektur.

Beide entstehen aus **einer** Modellauswertung (Energie + Ableitungen). Das
bedeutet gegenüber linear:

* die Eingabe verdoppelt sich (Geometrie **und** Verschiebung, je 8 Werte),
* das Modell wird **pro Element pro Newton-Iteration** ausgewertet (kein
  Caching möglich); dafür entfällt zur Laufzeit die teure
  Doppelknotenschleife des analytischen Elements,
* `Ke` und `Finte` sind **exakt konsistent** (gemeinsames Potential) — Newton
  bekommt die echte Jacobi-Matrix seines Residuums und konvergiert wieder mit
  der Rate des analytischen Elements. Es bleibt: der `Finte`-Fehler geht
  **direkt in die konvergierte Lösung** ein (das Modell definiert, wo
  „Gleichgewicht" ist), weshalb die Genauigkeit bei kleinen Amplituden
  entscheidend ist — genau dort greift der K₀-Split (Abschnitt 3.0).

Nach Ausnutzen aller Invarianzen (Translation, Skala, Rotation von Geometrie
**und** Zustand, `E·d`-Faktorisierung) bleiben von naiv 16 Eingangsdimensionen
**effektiv ~10**: ~5 für die Elementform + ~5 für den Verschiebungszustand —
deutlich härter als linear (~5), aber für ein kleines Netz machbar.

### 3.2 Erweiterte Kanonisierung: der Verschiebungszustand

Zusätzlich zur Geometrie-Kanonisierung (2.2, identisch übernommen) wird der
Verschiebungszustand mitkanonisiert:

```
û = (Rc · u_Knoten) / Lc,   danach Mittelwert je Richtung abgezogen
```

* **Mitrotieren** (`Rc·u`): Referenz- und Momentankonfiguration müssen im
  selben kanonischen System stehen (Objektivität).
* **Mitskalieren** (`/Lc`): Unter `x → L·x, u → L·u` bleibt der
  Verschiebungsgradient `gradU` (und damit `E_green`, `S`, `Ke`) exakt
  unverändert — ein kleines und ein großes Element mit „derselben" Deformation
  landen auf derselben Netzeingabe.
* **Translation abziehen:** Eine Starrkörperverschiebung ändert `gradU` nicht
  und wird exakt herausprojiziert.

### 3.3 Ko-Rotation des Zustands (Objektivität)

Eine Invarianz bleibt: die **Starrkörperrotation des deformierten Zustands**
(insbesondere bei Biegung der dominante Anteil von `û`). Für ein objektives
Material gilt exakt:

```
Finte(R∘u) = T_R · Finte(u),      Ke(R∘u) = T_R · Ke(u) · T_Rᵀ
```

Diese Rotation wird — wie die Geometrie-Rotation in 2.2/3.2 — **exakt**
herausgerechnet statt über Datenaugmentation approximiert:

1. Mittlere Zustandsrotation messen (Polarwinkel von `F̄ = I + gradU` am
   Elementmittelpunkt): `θ = atan2(F̄₂₁ − F̄₁₂, F̄₁₁ + F̄₂₂)`.
2. Zustand zurückdrehen: `û_corot = R(−θ)·(x̂+û) − x̂`, Translation erneut
   abziehen.
3. Netz auf `[ĉ; û_corot]` auswerten.
4. Vorhersage exakt zurückrotieren: `F̂ = T_θ·F̂_corot`, `K̂ = T_θ·K̂_corot·T_θᵀ`.

Diese Identität gilt für **jeden** Winkel `θ` — die Wahl bestimmt nur, wie
„rotationsfrei" der kanonische Zustand ist (Restrotation an den Gausspunkten
nach Ko-Rotation: median ~1.4°, selbst bei 45° Eingangsrotation). Training und
Element verwenden **dieselbe Formel** (`corot_state` in Python, identischer
Code in MATLAB) — verifiziert gegen das analytische Element auf ~1e-15, die
komplette Element-Kette (Kanonisierung + Ko-Rotation + Rückrotation +
Skalierung) per Oracle-Test auf ~1e-13.

Damit verliert der Zustandsraum die Rotationsdimension vollständig, und
Zustandsrotation ist zur Laufzeit **unbeschränkt** (Metadatum
`state_rot_max_deg = 180`) — Biegeprobleme mit großen Verdrehungen liegen
strukturell in der Hülle.

### 3.4 Exakte Faktorisierung

Für St.-Venant-Kirchhoff ist `S = C·E_green` mit konstantem `C ∝ E`, daher
exakt:

```
W(E, d)     = E · d · Lc² · What(ĉ, z)
Finte(E, d) = E · d · Lc  · (Kette aus What)
Ke(E, d)    = E · d       · (Kette aus What)
```

Trainiert wird mit `E = d = 1`; Materialgesetz (StVenant), `ν = 0,3` und
`planeStrain` sind **fest** eintrainiert und werden beim Laden **hart**
geprüft (`error`, kein Warning — ein Mismatch ergäbe stumm falsche Physik).

### 3.5 Modellein-/ausgabe und strukturelle Garantien

```
Eingang (16):  [ĉ (8);  z (8)]      kanonisiert, ko-rotiert, transl.-projiziert
Ausgang  (1):  Ŵ_NL                 SKALAR (Residual-Energie)
```

**Architektur:** GELU, 3 verdeckte Schichten à 48 Neuronen
(`FC(16)→48→48→48→FC(1)`), ≈ 5 600 Parameter / 5 424 MACs je Forward-Pass.
Zum Vergleich: die abgelöste Zwei-Kopf-Variante brauchte 4×128 und ~57 000
Parameter. Der K₀-Split macht die Zielfunktion so viel leichter, dass ein
Zehntel der Kapazität genügt — und das schlägt direkt auf die
Assemblierungszeit durch.

Die Eingaben werden vor dem Netz affin normalisiert
(`c̃ = (ĉ−c_mean)/c_std`, `z̃ = z/z_scale`); die Ableitungen werden exakt
zurücktransformiert. Wichtig: **`z` bekommt nur eine Skalierung, keinen
Offset** — sonst wäre `z̃(z=0) ≠ 0` und die strukturelle Nullform gebrochen.

**Strukturelle Nullform** (Subtraktionsform, im Training identisch):

```
Ŵ_NL = f(c̃, z̃) − f(c̃, 0) − ∇_z̃ f(c̃, 0)ᵀ·z̃
```

liefert `Ŵ_NL(ĉ,0) = 0` und `∇Ŵ_NL(ĉ,0) = 0` **exakt, unabhängig von den
Gewichten**. Kosten: ein zusätzlicher Forward + Reverse-Gradient bei `z̃ = 0`
pro Aufruf (kein zweiter Hessian). Die härtere Variante, zusätzlich
`½z̃ᵀ∇²f(c̃,0)z̃` abzuziehen (womit `K_T(ĉ,0) = K̂₀` exakt würde), wurde
bewusst **nicht** umgesetzt: ohne Cache kostet sie einen zweiten vollen
Hessian pro Aufruf — genau die Laufzeit, die der Ansatz gewinnen soll.

Was daraus **strukturell exakt** folgt (harte Nebenbedingung, Abschnitt 1.5):

1. **Konsistenz `Ke = ∂Finte/∂Ue`** — beide sind Ableitungen desselben
   Potentials. Das ist die zentrale Verbesserung gegenüber der
   Zwei-Kopf-Variante.
2. **Symmetrie von `Ke`** — Hessematrix eines Skalars.
3. **Kräftegleichgewicht `Σᵢ Finte_i = 0` und Translationsnullraum von `Ke`** —
   der Translationsprojektor `P` steckt in der Kette; eine nachträgliche
   Projektion `P·Ke·P` und die Knoten-4-Rekonstruktion der Vorgängerversion
   entfallen ersatzlos.
4. **`Finte = 0` bei reiner Starrkörperbewegung** — Ko-Rotation liefert
   `z = 0`, die Subtraktionsform `F̂(ĉ,0) = 0` (gemessen 1,4e-15).

Was **weich** bleibt (gelernt, nicht garantiert): `K_NL(ĉ,0) ≈ 0`, also dass
die Tangente im unbelasteten Zustand exakt `K̂₀` ist. Das Residual-Target ist
dort exakt null und wird supervidiert; geprüft wird es hart in Gate e4
(`Ke(u→0)` gegen `element_quad4_lin`).

> Dass Gate e4 wirklich nur diesen gelernten Rest misst, lässt sich zeigen:
> bei `u = 0` ist `θ = 0` und `B = I`, aber `∇θ ≠ 0`, sodass
> `G̃ = I + (Σx̂)·∇θᵀ` von der Identität abweicht. Der Zusatzterm fällt jedoch
> **exakt** heraus, denn `Σx̂` ist genau der Rotationsmode der kanonischen
> Geometrie und `K̂₀·(Rotationsmode) = 0`. Damit gilt
> `G̃ᵀ·P·K̂₀·P·G̃ = K̂₀`, und es bleibt `Ke(u→0) = K_linear + O(K_NL(ĉ,0))`.

### 3.6 Trainingsdatensatz

Ein Sample = **eine Geometrie × ein Verschiebungszustand**, Umfang ~240 000
Trainingssamples. Der Mix ist bewusst dreiteilig:

| Anteil | Quelle | Zweck |
|---|---|---|
| 70 % | synthetisch (`generate_distorted_quad` + `sample_state`) | breite, gleichmäßige Abdeckung des Zustandsraums |
| 20 % | **Newton-Trajektorien** aus echten FEM-Rechnungen | genau die Zustände, die der Löser später besucht |
| 10 % | Active-Learning-Slot | gezieltes Nachsampeln in Fehler-Zellen (Runde 1: synthetisch gefüllt) |

Die **Newton-Trajektorien** (`generate_newton_trajectories.m`) rechnen ~25
zufällige Strukturen mit dem *analytischen* Element und zeichnen in **jeder**
Newton-Iteration die Zustände einer Element-Stichprobe auf — auch die nicht
konvergierten Iterierten, denn genau die besucht der Löser. Das schließt die
Verteilungs-Lücke, die im Juli der Kernfehler war
(`DLFE_quad4_Entwicklung.md`, III.2).

> **Leakage-Sperre:** Die 5 Benchmark-Strukturen aus
> `FEMSolid_ex_quad4_07_ai_nl_benchmark.m` sind **held-out** und dürfen nie
> Trajektorien-Quelle sein — sonst misst der Benchmark Memorierung statt
> Generalisierung. Der Generator nutzt ausschließlich zufällige Geometrien,
> Lagerungen und Lasten; drei Strukturen sind zusätzlich als reine
> Validierungs-Strukturen reserviert.

* **Geometrien:** identisch zu 2.4 (verzerrungsgetrieben, gleiche Hülle).
  Jede Geometrie wird kanonisiert — das Netz sieht nur kanonische Formen.
* **Zustände** (`sample_state`): Newton-Iterierte sind
  **Nicht-Gleichgewichtszustände** — der Zustandsraum muss generisch
  abgedeckt werden, nicht nur physikalische Lösungen. Pro Geometrie:
  12 % exakt `û = 0` (Referenzzustand, an dem jeder Lastschritt startet);
  sonst affiner Anteil (`u = ĉ·Gᵀ`, `G ~ N(0, 0.15)`, deckt konstante
  Dehnung und Scherung ab) plus nicht-affiner Anteil (`N(0, 0.05)` pro
  Knoten, deckt Biege-/Hourglass-Moden ab), Translation projiziert, Amplitude
  **log-uniform** über ~3 Dekaden skaliert (`× exp(uniform(ln 2e-3, 0))`) —
  von `‖E_green‖ ~ 1e-3` (Newton-Endphase) bis zur Hüllengrenze. Jeder
  Zustand wird abschließend ko-rotiert (3.3); eine Rotations-Augmentation
  gibt es bewusst **nicht** (rotierte Kopien ergäben nach der Ko-Rotation
  exakt dieselbe Netzeingabe).
* **Zustands-Hülle:** `‖E_green‖_F ≤ 0,2` über alle Gausspunkte
  (`E_MAX = 0.2`). Rotation ist seit der Ko-Rotation **keine** Hüllengrenze
  mehr. Die Benchmark-Lasten müssen damit nur noch die Dehnungs-Hülle
  einhalten — das prüft `…_08_ai_nl_check.m` gezielt.
* **Targets:** `Ŵ`, `F̂`, `K̂` aus einem Python-Port des analytischen Elements
  (`energy_stiffness_force_ref`: 2×2-Gauss, `gradU → F → E_green → S`,
  Energiedichte `½·Evecᵀ·C·Evec`, nichtlineare B-Matrix + geometrischer Term —
  Zeile für Zeile äquivalent zu `element_quad4_nl.m`), gerechnet direkt auf
  der kanonischen Geometrie mit `E = d = 1`. Daraus die **Residual-Targets**
  `Ŵ_NL = Ŵ − ½zᵀK̂₀z`, `F̂_NL = F̂ − K̂₀z`, `K̂_NL = K̂ − K̂₀`.
  Die Targets prüfen sich **selbst** (Gate a): `F̂ = ∂Ŵ/∂z`, `K̂ = ∂F̂/∂z`
  und `k0_ref = K̂(z=0)` per zentraler Differenzen in fp64 — ein Fehler in der
  neuen Referenz-Energie würde sonst stumm inkonsistente Sobolev-Targets
  erzeugen.

Die log-uniforme Amplitudenverteilung ist wichtig: eine uniforme Verteilung
deckt das für den Newton-Löser besonders relevante Kleinlast-Regime
(Endphase der Iteration) kaum ab, weil dort die *absolute* Anzahl an Samples
verschwindet, obwohl gerade dort hohe *relative* Genauigkeit gebraucht wird.

### 3.7 Sobolev-Verlust: Residuen lernen, Totalgrößen messen

Trainiert wird auf allen drei Ableitungsstufen gleichzeitig (**Sobolev-
Training**), denn `Ŵ`, `F̂` und `K̂` hängen über Differentiation zusammen —
ein Output-Rescaling wie das frühere `FINT_SCALE` ist prinzipbedingt nicht
mehr möglich, die Balance läuft ausschließlich über Gewichte und Floors.

Der entscheidende Kniff: gelernt werden die **Residuen** (`Ŵ_NL`, `F̂_NL`,
`K̂_NL`), normiert wird aber durch die Normen der **Totalgrößen**. Weil
`K̂₀z` exakt ist, gilt `ΔF_tot = ΔF_NL` und `ΔK_tot = ΔK_NL` — der Verlust
optimiert damit direkt den relativen Fehler der Größen, die der Newton-Löser
tatsächlich sieht:

```
L = λ_W·mean( (ΔŴ_NL)²   / (Ŵ_tot²      + floorW²) )
  + λ_F·mean( ‖ΔF̂_NL‖²   / (‖F̂_tot‖²    + floorF²) )
  + λ_K·mean( ‖ΔK̂_NL‖²_F /  ‖K̂_tot‖²_F )
```

`λ_W = 0,1`, `λ_F = λ_K = 1`; `floorW/F = 0,05·RMS` der jeweiligen
Totalgröße. Der **K-Term braucht keinen Floor**, weil `K̂_tot → K̂₀ ≠ 0`
auch für `z → 0` — der Nenner kann nicht verschwinden. Genau diese
Singularität hatte die Vorgängerversion zum Balancieren gezwungen und im Juli
den Trainingskollaps ausgelöst.

Der Kollaps-Modus vom 15.07. ist damit **dreifach** entschärft: Klein-Init
der Ausgabeschicht, strukturelle Nullform (Startvorhersage ist exakt 0 —
und das ist für kleine Zustände die *richtige* Antwort) und der floorfreie
K-Nenner.

**Trainings-Setup:** Adam, LR 1e-3, `ReduceLROnPlateau` (Patience 20),
Gradient-Clipping 1.0, Mini-Batch 4096 mit dem K-Term auf einem rotierenden
1024er-Subsample (Speicher), Early Stopping. Ableitungen im Training über
`torch.func` (`grad`, `jacfwd(grad)`, forward-over-reverse), fp32 ohne AMP
(höhere Ableitungen). Architektur-Sweep über Breite × Tiefe optional
(`QUAD4_SWEEP`), Artefakte nach `arch_sweep_nl_W/`; Schnelllauf:
`QUAD4_QUICK`.

**Metriken:** relative Fehler von `F` und `K` als **Totalgrößen**, berichtet
mit `mean, P50, P90, P95, P99` — jeweils gesamt, je Amplituden-Terzil, je
Verzerrungs-Bin und getrennt für synthetische und Trajektorien-Daten.
Perzentile sind kein Luxus: Newton wird vom **schlechtesten** Element
limitiert (die Kontraktionsrate hängt am dominanten Eigenwert), nicht vom
mittleren.

### 3.8 Verifikation und Ergebnis

Die Verifikation läuft als **Leiter**, bei der jedes Gate das nächste
freischaltet — entscheidend ist, dass die ersten beiden Stufen *ohne Netz*
laufen und damit Ketten- von Lernfehlern trennen:

| Gate | Was geprüft wird | Ergebnis |
|---|---|---|
| a | Referenz-Targets in sich: `F̂=∂Ŵ/∂z`, `K̂=∂F̂/∂z`, `k0_ref=K̂(z=0)`, Skalierungsgesetz | 1,4e-9 / 6,0e-11 / 0 / 9,0e-16 |
| b | Kette mit **analytischer** Energie vs. FD des Komposits und vs. analytisches Element | 2,4e-9 / **2,4e-13** |
| c | Trainingsskript: autograd-`K_NL` vs. FD(`F_NL`), fp64 | 1,6e-9 |
| d | MATLAB-Modell vs. Python-Oracle-Vektoren (fp64) | 4,0e-14 |
| e1/e2 | **physische** Elemente: `Finte` vs. FD(`W_phys`), `Ke` vs. FD(`Finte`) in globalen Koordinaten | 9,2e-10 / 8,8e-10 |
| e3 | `Finte` bei reiner Starrkörperbewegung | 1,4e-15 |
| e4 | `Ke(u→0)` vs. lineares Element (weiche Nebenbedingung) | 0,81 % max / 0,29 % Mittel |
| f/g/h | Benchmark: Genauigkeit (mean + P99), Hülle, Speedup | s. u. |

**Validierungsgenauigkeit des deployten Netzes** (relative Total-Fehler, das
was Newton sieht; 27 432 Validierungssamples):

| Datenquelle | eF mean | eF P99 | eK mean | eK P99 |
|---|---|---|---|---|
| gesamt | **0,39 %** | 2,77 % | **0,60 %** | 4,29 % |
| nur synthetisch | 0,43 % | 2,93 % | 0,66 % | 4,52 % |
| nur Newton-Trajektorien | 0,14 % | 0,23 % | 0,17 % | 0,38 % |

Damit ist das Go-Kriterium (mean < 2 %, P99 < 5 %) erfüllt. Aufschlussreich
ist die letzte Zeile: auf **echten Solver-Zuständen** ist der Fehler rund
dreimal kleiner als auf der breiten synthetischen Verteilung. Der
Zustandsraum, den Newton tatsächlich besucht, ist die leichtere Teilmenge —
die synthetischen Daten decken bewusst auch Zustände ab, die nie vorkommen.

Gate e2 ist der Kern: `Ke` ist auf **8,8e-10** exakt die Jacobi-Matrix von
`Finte` — die 7,1-%-Inkonsistenz der Vorgängerversion existiert
konstruktionsbedingt nicht mehr. Damit ist der offene Punkt aus Teil III
(Newton-Kontraktionsrate 0,902, Iterationslimit) **geschlossen**: Newton
konvergiert wieder mit der Rate des analytischen Elements.

Die Gates a–e prüfen **Struktur**, nicht Lernqualität — sie gelten für jedes
Netz dieser Form. Was vom Training abhängt, ist allein die Genauigkeit der
gelernten Energieabweichung (Gates e4 und f) sowie, über die Netzgröße, der
Speedup (Gate h).

### 3.9 Benchmark: 5 nichtlineare Strukturen (Stand 2026-08-18)

Netz h48/d3, relatives Konvergenzkriterium `tolR = 1e-6·‖Fext‖` für **beide**
Backends:

| Struktur | it FEM | it KI | Finte mean/P99 | Ke mean/P99 | dU | dVM |
|---|---|---|---|---|---|---|
| Kragträger (Endquerlast) | 23 | **23** | 0,14 / 0,18 % | 0,13 / 0,15 % | 0,09 % | 0,10 % |
| Kragträger (Eigengewicht) | 22 | **22** | 0,14 / 0,19 % | 0,14 / 0,19 % | 0,11 % | 0,12 % |
| Tiefer Kragträger (Querlast) | 18 | **18** | 0,12 / 0,16 % | 0,13 / 0,15 % | 0,07 % | 0,08 % |
| Scheibe (Scherung) | 12 | **12** | 0,13 / 0,18 % | 0,13 / 0,14 % | 0,04 % | 0,04 % |
| Kragträger (Axialzug) | 19 | **19** | 0,23 / 0,23 % | 0,31 / 0,31 % | 0,14 % | 0,05 % |

**Die Iterationszahlen sind in allen fünf Fällen identisch** — die
strukturelle Konsistenz wirkt genau wie vorhergesagt. Zum Vergleich: die
Zwei-Kopf-Variante erreichte in denselben Strukturen das Iterationslimit
(150 = 6 Lastschritte × 25). Verschiebungsfehler 0,04–0,14 %, also rund eine
Größenordnung besser als die 0,44–7,8 % der Vorgängerversion.

**Speedup (Gate h):**

| Struktur | Assemblierung | Gesamtlösung |
|---|---|---|
| Kragträger (Endquerlast) | 1,39× | 1,10× |
| Kragträger (Eigengewicht) | 1,41× | 1,53× |
| Tiefer Kragträger (Querlast) | 1,31× | 1,34× |
| Scheibe (Scherung) | 1,42× | 1,31× |
| Kragträger (Axialzug) | 1,54× | 1,64× |

Die Gesamtlösung ist auf **5 von 5** Strukturen schneller (Median der
Assemblierung: 1,41×).

> **Woher der Speedup kommt — und woher nicht.** Das Kostenmodell zeigt, dass
> das Netz **arithmetisch teurer** ist als das analytische Element (~120 000
> gegenüber ~3 600 MACs je Aufruf). Gemessen ist es trotzdem schneller, weil
> das analytische Element seine Zeit in der **interpretierten
> Doppelknotenschleife** verbringt (4 Gausspunkte × 16 Knotenpaare mit
> winzigen Matrizen), während das Netz aus wenigen dichten Matrixprodukten
> besteht. Der Vorteil stammt also aus der Ausführungsform, nicht aus weniger
> Rechenarbeit — in einer vektorisierten oder kompilierten
> Referenzimplementierung könnte das analytische quad4 gewinnen. Das
> eigentliche Argument für den Ansatz bleibt die Extrapolation: bei teuren
> Elementen (3D, viele Gausspunkte, komplexe Materialgesetze) wächst **nur**
> die FEM-Seite, die Netz-Seite bleibt konstant.

Ausführliche Gate-Tabelle mit Schwellwerten: `DLFE_quad4_nl_plan.md`,
Abschnitt „Umsetzungsstand". Die Vorgeschichte (Verteilungs-Lücke,
Trainingskollaps, Zwei-Kopf-Variante) steht in `DLFE_quad4_Entwicklung.md`,
Teile III und IV.

---

## 4. Implementierung in MATLAB

### 4.1 Dateiaufteilung und gemeinsames Muster

| Datei | Rolle |
|---|---|
| `element_quad4_lin_ai.m` | lineares Element (ein Forward-Pass → `Ke`) |
| `element_quad4_nl_ai.m` | nichtlineares Element: Signatur wie `element_quad4_nl.m`, Metadaten-Guards, OOD-Proxy, analytisches `Fbe` |
| `quad4_nl_ai_energy.m` | **Kette**: Kanonisierung, Ko-Rotation, exakte 1./2. Ableitungen, Rückskalierung |
| `quad4_nl_ai_model.m` | **Modell**: `K̂₀`-Schleife + Residual-Netz (Wert, Gradient, Hessian) |

Die Trennung Modell / Kette ist bewusst: das Modell ist ohne Solver direkt
gegen die Python-Oracle-Vektoren im `.mat` testbar (Gate d), die Kette wurde
mit *analytischer* Energie geprüft, bevor überhaupt ein Netz existierte
(Gate b).

Gemeinsames Muster aller KI-Elemente:

1. **Einmaliges Laden** der Netzgewichte beim ersten Aufruf
   (`persistent`-Variablen) — kein Neuladen der `.mat`-Datei pro Element.
2. **Konsistenzprüfung** (nur beim ersten Aufruf pro Session). Beim
   nichtlinearen Element sind **Material, `ν` und ebener Zustand harte
   `error`** statt Warnungen: sie sind eintrainiert und nicht
   herausfaktorisiert, ein Mismatch ergäbe stumm falsche Physik. Ebenso hart:
   `model_form`, `activation`, Kanonisierungs- und Ko-Rotations-Strings sowie
   die Ein-/Ausgabegrößen des Netzes (16 Eingänge, **skalarer** Ausgang).
3. **Kanonisierung** der Eingabe (Abschnitte 2.2/3.2–3.3).
4. **Modellauswertung** (linear: ein Forward-Pass; nichtlinear: Energie,
   Gradient und Hessian, s. 4.2).
5. **Rücktransformation** in die physikalische Orientierung und Skalierung.
6. **Symmetrisierung** `0.5·(Ke+Ke')` gegen Rundungsreste.

Lastvektoren bleiben analytisch: `Fbe` per kleiner Gauss-Schleife (nur wenn
eine Last anliegt), `Fte` wie im analytischen Element bzw. `= 0` (nichtlinear).

**Kein Cache, kein Zustand:** `K̂₀` und der Null-Zustands-Pass werden pro
Aufruf neu gerechnet. Das hält `assemble.m` zustandslos und die
Solver-Signaturen unangetastet — eine bewusste Entscheidung zugunsten
einfacher Architektur, bezahlt mit einem kleinen, gemessenen Laufzeitanteil.

**OOD-Prüfung:** ein billiger Proxy (`‖E_green‖` am Elementmittelpunkt — der
Verschiebungsgradient liegt für die Ko-Rotation ohnehin vor) warnt **einmal
pro Session**, wenn ein Zustand die trainierte Hülle verlässt. Die volle
4-Gausspunkt-Hüllenprüfung bleibt bewusst aus dem Hot-Path; dafür gibt es
`FEMSolid_ex_quad4_08_ai_nl_check.m`.

### 4.2 Netzableitungen (Hot-Path, nichtlinear)

Das nichtlineare Element braucht nicht nur den Funktionswert, sondern
Gradient **und** Hessematrix des Netzes. Beides wird mit handkodierten
Rekurrenzen berechnet — tiefen-agnostisch über `num_linear_layers`, sodass
eine andere Netzgröße keinen Code-Eingriff erfordert:

```
Forward :  z_l = W_l·a_{l-1} + b_l,   a_l = gelu(z_l)
Reverse :  r_{l-1} = W_lᵀ·(gelu'(z_l) ⊙ r_l),          Seed r_L = 1
Fwd-over-Reverse (alle 8 z-Richtungen als Matrix-Batch, keine Schleife):
           ṙ_{l-1} = W_lᵀ·( gelu'(z_l) ⊙ ṙ_l + gelu''(z_l) ⊙ ż_l ⊙ r_l )
```

GELU und seine Ableitungen in exakter erf-Form (identisch zu PyTorch
`nn.GELU()`):

```
gelu(x)   = x·Φ(x)
gelu'(x)  = Φ(x) + x·φ(x)
gelu''(x) = (2 − x²)·φ(x)          Φ = ½(1+erf(x/√2)),  φ = e^{−x²/2}/√(2π)
```

Der Null-Zustands-Pass (Subtraktionsform) läuft **ohne** Tangenten — er
braucht nur Wert und Gradient, nicht den Hessian. Verifiziert gegen PyTorch
auf fp32-Präzision (Gate d, 1,3e-7).

### 4.3 Rücktransformation über die Kette

Die Kette rechnet die kanonischen Ableitungen `p = ∂Ŵ/∂z`, `H = ∂²Ŵ/∂z²`
exakt in physikalische Größen um. Mit `G̃ = B + (B·Σ·y)·∇θᵀ` (das ist
`∂z/∂v` ohne Projektor):

```
g_v   = G̃ᵀ·P·p
C_v   = −(ΣBᵀp̃)∇θᵀ − ∇θ(ΣBᵀp̃)ᵀ − (p̃ᵀBy)∇θ∇θᵀ + (p̃ᵀBΣy)∇²θ
K_v   = G̃ᵀ·P·H·P·G̃ + C_v
Finte = E·d·Lc·Tcᵀ·P·g_v          Ke = E·d·Tcᵀ·P·K_v·P·Tc
```

`C_v` ist der Beitrag der **θ-Abhängigkeit** — der Ko-Rotationswinkel hängt
vom Zustand ab und wird nicht eingefroren. Das Einfrieren wäre nur bis
O(Netzfehler) korrekt und würde die FD-Gates reißen (Details:
`DLFE_quad4_Entwicklung.md`, IV.3).

Was gegenüber der Vorgängerversion **entfällt**: `recon_map`-Gather
(Symmetrie ist jetzt strukturell), Knoten-4-Rekonstruktion (steckt in `P`),
`finte_output_scale` (kein separater Kraftkopf mehr) und die nachträgliche
Projektion `P·Ke·P` (ebenfalls in der Kette).

### 4.4 Performance-Aspekte

Da das Element **pro Newton-Iteration pro Element** (nichtlinear) bzw.
**einmal pro Element** (linear) aufgerufen wird, wurde der Hot-Path gezielt von
vermeidbarer Arbeit befreit:

| Maßnahme | Wirkung |
|---|---|
| Gewichte `persistent`, einmalig aufbereitet | keine Struct-/Reshape-Zugriffe pro Aufruf |
| Hessian über alle 8 Richtungen als Matrix-Batch | keine Richtungsschleife |
| Null-Zustands-Pass ohne Tangenten | spart den zweiten Hessian |
| Konsistenzcheck nur 1× pro Session | keine String-Vergleiche im Hot-Path |
| OOD-Proxy statt 4-GP-Hüllenprüfung | nutzt den ohnehin berechneten `gradU` |
| Lastschleife nur bei tatsächlicher Last | spart die `shape_quad4`-Auswertungen sonst |
| `orth()`/SVD → Spaltennormierung (linear) | keine SVD mehr pro Element |

**Zwei Ebenen sind zu unterscheiden** — und beim nichtlinearen Element ist die
zweite die entscheidende:

* **pro Assemblierung:** das Energiemodell kostet ~20 Forward-Äquivalente
  (Hessian) plus die `K̂₀`-Schleife. Linear ist die KI-Variante ~2× schneller
  als die klassische Routine; nichtlinear ist der Vorsprung kleiner, weil der
  Hessian bezahlt werden muss.
* **pro Lösung** (Assemblierungen × Newton-Iterationen): hier zahlt sich die
  strukturelle Konsistenz aus. Die Vorgängerversion brauchte wegen der
  inkonsistenten Tangente ein Vielfaches an Iterationen (Iterationslimit); das
  aktuelle Element konvergiert mit **derselben Iterationszahl wie das
  analytische**. Genau deshalb misst der Benchmark beide Ebenen getrennt
  (Gate h1/h2).

> Einordnung: Ein 4-Knoten-Element ist analytisch ohnehin billig. Der
> DLFE-Ansatz spielt seinen Vorteil grundsätzlich erst bei *teuren* Elementen
> aus (komplexe Materialgesetze, viele Gausspunkte, nichtlinear, 3D). Der
> Benchmark gibt deshalb neben den gemessenen Zeiten ein **Kostenmodell**
> (MACs Netz-Kette + `K̂₀` vs. FLOPs der Gauss-Schleife) aus — nur damit ist
> die Extrapolation auf teure Elemente nachvollziehbar.

---

## 5. Offene Punkte und nächste Schritte

**Erledigt mit dem Residual-Energie-Netz (2026-08-18):** Konsistenz
`Ke = ∂Finte/∂Ue` (strukturell statt gelernt), `F̂(ĉ,0) = 0` (strukturell),
modell-ehrliches relatives Konvergenzkriterium im Benchmark. Die
Newton-Konvergenzrate ist damit kein offener Punkt mehr.

**Nichtlinear — verbleibende Punkte:**

1. **Genauigkeit auf 2 % in jedem Terzil** (mean) **und 5 % im P99** — der
   verbleibende Engpass ist reines Training, keine Struktur. Kontingenzliste
   in `DLFE_quad4_nl_plan.md`, Phase 6: zuerst eine **Active-Learning-Runde**
   (Fehler-Zellen aus Verzerrung × Amplitude gezielt nachsampeln), dann
   λ_K-Schedule, dann breitere Netze.
2. **`K_NL(ĉ,0) = 0` strukturell** statt weich (Gate e4): dazu müsste
   zusätzlich `½zᵀ∇²f(ĉ,0)z` abgezogen werden. Ohne Cache kostet das einen
   zweiten Hessian pro Aufruf — lohnt sich erst zusammen mit einem
   Precompute (siehe 3.). Zwischenlösung, falls e4 limitiert: Zusatzterm
   `μ·‖K_NL(ĉ,0)‖²/‖K̂₀‖²` im Verlust.
3. **Optionaler Precompute** (bewusst zurückgestellt): `K̂₀`, `f(ĉ,0)` und
   `∇f(ĉ,0)` sind bei Total Lagrange über die gesamte Simulation konstant.
   Ein Precompute beim Modellaufbau würde ~15–25 % der Elementkosten sparen
   und Punkt 2 gratis machen — verlangt aber ein zusätzliches Feld im
   Element-Dispatch (`assemble.m` reicht `opts` bereits durch, die Signaturen
   blieben unverändert). Aktuell bewusst **nicht** umgesetzt: `assemble.m`
   bleibt zustandslos.
4. **Aitken-Relaxation** in `newton.m` — mit konsistenter Tangente nicht mehr
   nötig; nur relevant, falls wieder ein Quasi-Newton-Fall auftritt.

**Linear (nice-to-have):**

* **Eigenstruktur-Loss:** die kleinen Nicht-null-Eigenwerte treiben den
  Verschiebungsfehler; ein Zusatzterm darauf statt reiner Frobenius-Norm.
* **Voller Architektur-Sweep** mit dem verzerrten Datensatz — der zuletzt
  deployte Lauf war ein GELU-Schnelllauf zur Verifikation.

---

## 6. Zusammenfassung

Beide Varianten ersetzen einen analytischen, auf Gauss-Integration und
B-Matrizen basierenden Berechnungsschritt durch ein kleines, einmal
trainiertes MLP. Der methodische Kern liegt **nicht** in der Netzarchitektur
(ein einfaches MLP reicht), sondern in vier Prinzipien — und alle vier sagen
dasselbe: *was exakt bekannt ist, wird nicht gelernt.*

* **Exakte Vorverarbeitung bekannter Invarianzen** (Translation, Skala,
  Rotation, im nichtlinearen Fall zusätzlich Objektivität des
  Verschiebungszustands) statt deren approximativem Erlernen.
* **Strukturelle statt gelernte Erzwingung** exakt bekannter Eigenschaften
  der Zielgröße (Symmetrie, Gleichgewicht, Nullraum).
* **Ein Potential statt getrennter Ausgänge** (nichtlinear): weil `Finte` und
  `Ke` Ableitungen derselben Energie sind, ist ihre Konsistenz per
  Konstruktion exakt — statt über einen Strafterm angenähert, was
  nachweislich nicht funktionierte.
* **Analytische Abspaltung des dominanten Anteils** (K₀-Split): das Netz
  lernt nur die nichtlineare Energieabweichung. Das macht die Zielfunktion
  leichter (kleineres, schnelleres Netz) und das kritische
  Kleinamplituden-Regime exakt.

Der Preis für jede dieser Entscheidungen ist ein Stück Rechenarbeit zur
Laufzeit (Kanonisierung, Ko-Rotations-Kettenregel, `K̂₀`-Schleife,
Null-Zustands-Pass) — und genau deshalb misst der Benchmark den Speedup auf
zwei Ebenen und legt ein Kostenmodell daneben.

Die verbleibende Fehlerquelle ist dort am größten, wo eine Eigenschaft *nicht*
hart erzwungen werden kann, weil sie selbst für die exakte Lösung nicht in
geschlossener Form vorliegt: beim linearen Element die tatsächlichen
`Ke`-Werte, beim nichtlinearen Element der Wert der Energieabweichung `Ŵ_NL`
— und damit über `Finte` die Lage des Gleichgewichts. Das ist eine reine
Frage der Trainingsqualität, keine strukturelle Grenze mehr.
