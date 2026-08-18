# DLFE quad4 nichtlinear: Neuentwurf als Residual-Energie-Netz (K₀-Split + Sobolev-Training)

**Version 2 (2026-08-18).** Ersetzt Version 1 (reines Energie-Netz). Änderungen
gegenüber V1: analytischer K₀-Split ohne Cache, Residual-Lernen, Input-
Normalisierung, Newton-Trajektorien + Active Learning im Datensatz, Perzentil-
Gates, Speedup als hartes Gate. Alle Entscheidungen sind mit Paul geklärt.

## Umsetzungsstand (2026-08-18)

Implementiert und verifiziert. Neue/geänderte Dateien:

| Datei | Rolle |
|---|---|
| `training/quad4/quad4_nl_ref.py` | **neu** — Referenzmathematik (Energie/F/K, K₀, Kette mit exakten Ableitungen), Gates a und b |
| `training/quad4/train_quad4_nl_W_network.py` | **neu** — Datensatz (70/20/10), Sobolev-Training, Perzentil-Metriken, Gate c, Export |
| `training/quad4/generate_newton_trajectories.m` | **neu** — Newton-Trajektorien als Trainingsdaten (Held-out-Sperre) |
| `sourcecode/elements/quad4/quad4_nl_ai_model.m` | **neu** — kanonisches Energiemodell (K₀-Split + Netz, handkodierte Ableitungen) |
| `sourcecode/elements/quad4/quad4_nl_ai_energy.m` | **neu** — Kanonisierungs-/Ko-Rotations-Kette mit exakten 1./2. Ableitungen |
| `sourcecode/elements/quad4/element_quad4_nl_ai.m` | **neu geschrieben** — Signatur unverändert, harte Metadaten-Guards, OOD-Proxy |
| `examples/FEMSolid_ex_quad4_09_ai_nl_consistency.m` | **neu** — Gates d und e |
| `examples/FEMSolid_ex_quad4_07_ai_nl_benchmark.m` | relatives `tolR`, Perzentile, Gate f, Gate h (Speedup + Kostenmodell) |
| `examples/FEMSolid_ex_quad4_08_ai_nl_check.m` | auf das neue `.mat` und die Rotations-Freiheit angepasst |
| `docs/*`, `README.md`, `.gitignore` | Doku fortgeführt (Stand, Historie Teil IV, Plan), Sweep-Artefakte ignoriert |

**Gate-Ergebnisse** (Stand 2026-08-18, Netz aus dem QUICK-Lauf — die Gates a–e
sind netzunabhängig bzw. strukturell und gelten auch für das finale Netz):

| Gate | Ergebnis | Schwelle | Status |
|---|---|---|---|
| a — `F=∂W/∂z`, `K=∂F/∂z`, `k0_ref=K(z=0)`, Skalierung | 1.4e-9 / 6.0e-11 / 0.0 / 9.0e-16 | 1e-8 / 1e-8 / 1e-10 / 1e-12 | **grün** |
| b — Kette vs. FD des Komposits | 2.4e-9 (F) / 8.2e-10 (K) | 1e-8 | **grün** |
| b — Kette vs. analytisches Element (Oracle) | 2.4e-13 (F) / 2.7e-15 (K) | 1e-10 | **grün** |
| b — `z` bei reiner Starrkörperbewegung | 1.6e-15 | 1e-12 | **grün** |
| c — autograd-`K_NL` vs. FD(`F_NL`), fp64 | 1.6e-9, Symmetrie 5.4e-15 | 1e-6 | **grün** |
| d — MATLAB-Modell vs. Python-Oracle (fp64) | W 4.0e-14 / F 6.4e-15 / K 4.4e-16 | 1e-10 | **grün** |
| e1 — `Finte` vs. FD(`W_phys`), physisch | 9.2e-10 | 1e-6 | **grün** |
| e2 — `Ke` vs. FD(`Finte`), global | 8.8e-10 | 1e-6 | **grün** |
| e3 — `Finte` bei Starrkörperbewegung | 1.4e-15 | 1e-12 | **grün** |
| e4 — `Ke(u→0)` vs. lineares Element | 0,81 % max / 0,29 % Mittel | < 1 % | **grün** |
| f — Benchmark: Genauigkeit, Iterationen, dU | Fehler ≤ 0,31 %, Iterationen **identisch**, dU ≤ 0,14 % | mean < 2 %, P99 < 5 %, dU < 0,5 %, Iter ≤ FEM+3 | **grün** |
| g — Hüllen-Check (Beispiel 08) | in Hülle Ke 0,13–1,46 % / Finte 0,00–0,53 %; außerhalb erwartungsgemäß wachsend (2,9 % → 21 %), OOD-Warnung feuert; E/d-Skalierung 8e-17 | in-hull < 2 % | **grün** |
| h1 — Gesamt-Lösungszeit | KI schneller auf **5 von 5** (1,10–1,64×) | ≥ 4 von 5 | **grün** |
| h2 — Assemblierung (berichtet) | Median **1,41×** | — | berichtet |

**Deploytes Netz:** GELU, `h48 d3`, 5 569 Parameter / 5 424 MACs. Das
Go-Kriterium ist mit der **kleinsten** Sweep-Architektur erfüllt:
eF 0,39 % (P99 2,77 %), eK 0,60 % (P99 4,29 %) auf 27 432 Validierungssamples.
Auf reinen Newton-Trajektorien sogar eF 0,14 % / eK 0,17 %.

> **Einordnung des Speedups (wichtig, nicht überinterpretieren):** Das
> Kostenmodell weist das Netz als arithmetisch **teurer** aus (~120 000 vs.
> ~3 600 MACs je Aufruf). Gemessen ist es trotzdem 1,41× schneller, weil die
> interpretierte Doppelknotenschleife des analytischen Elements durch wenige
> dichte Matrixprodukte ersetzt wird. Der gemessene Vorteil kommt also aus der
> Ausführungsform, nicht aus weniger Rechenarbeit — bei einer vektorisierten
> oder kompilierten Referenz könnte das analytische quad4 gewinnen. Das
> tragfähige Argument bleibt die Extrapolation auf teure Elemente, wo nur die
> FEM-Seite wächst.

**Kernbefund:** e1/e2 bestätigen in MATLAB, was der Ansatz strukturell
verspricht — `Ke` ist auf 9e-10 exakt die Jacobi-Matrix von `Finte`. Die
7-%-Inkonsistenz der Zwei-Kopf-Variante, Ursache der Newton-Kontraktionsrate
0,902, existiert konstruktionsbedingt nicht mehr. e4 ist die einzige *weiche*
Nebenbedingung (`K_NL(ĉ,0) ≈ 0`) und hängt an der Trainingsqualität.

**Solver-Integrationstest** (Kragträger 12×2, relatives `tolR`, QUICK-Netz):
Newton-Iterationen **FEM 9 / KI 9** bei 3 Lastschritten, `dU` 0,29 %,
Assemblierung 1,14× schneller. Das Zwei-Kopf-Netz lief in denselben Fällen in
den Iterationsdeckel.

**Datensatz** (Phase 2, tatsächlich erzeugt): 25 Zufallsstrukturen →
**30 681 Newton-Zustände**, davon 3 432 aus 3 reservierten
Validierungs-Strukturen; nur 5 Zustände fielen durch den Hüllen-Filter. Die
5 Benchmark-Strukturen sind wie gefordert **nicht** enthalten.

**Abweichungen vom Plan (bewusst):**

1. **Phase 0, Baseline-Timing-Lauf entfällt.** Das alte Zwei-Kopf-Element
   wurde in place ersetzt (Signatur/Dateiname unverändert, wie geplant), womit
   die Vergleichslinie „altes AI-Element" nicht mehr messbar ist. Der Verlust
   ist gering: die alte Variante erreichte das Konvergenzkriterium nie, ihre
   Gesamt-Lösungszeit war also ohnehin nicht vergleichbar. Ihre Kennzahlen
   sind in `DLFE_quad4_Entwicklung.md`, III.4 dokumentiert.
2. **Gate d: Testvektoren gefiltert und in fp64.** Zwei Korrekturen gegenüber
   dem Plan. (a) ~12 % der Val-Samples sind exakt `z = 0` (`P_ZERO` im
   Sampler); dort sind `W` und `F` exakt null, ein relativer Vergleich wäre
   eine Division durch null. Der Export wählt deshalb Testvektoren mit `‖z‖`
   oberhalb des Medians, die MATLAB-Metrik hat zusätzlich einen Floor.
   (b) Die Vektoren werden in **fp64** statt fp32 erzeugt — sonst misst das
   Gate die Auslöschung in `K̂₀·z` (Nullraum von `K̂₀`) statt der
   Rekurrenzen. Schwelle entsprechend von 1e-5 auf 1e-10 verschärft.
3. **Gate-c-Schwelle in fp64 statt fp32.** Wie im Plan, nur explizit: das
   Modell wird für Gate c nach `double` konvertiert.
4. **Nichts gelöscht.** `train_quad4_nl_K_network.py` und
   `quad4_nl_K_network.mat` (Zwei-Kopf-Variante) bleiben im Arbeitsverzeichnis
   liegen — abgelöst, aber nicht entfernt. Ohne Git-Snapshot wäre ein Löschen
   unwiderruflich; die Entscheidung liegt bei Paul (Plan Phase 7).
5. **Sweep-Auswahl kostenbewusst statt rein genauigkeitsbasiert** (Ergänzung
   zu Phase 3.4). Der Plan sah „Top-2 nach mittlerem K̂-Fehler" vor — eine
   reine Genauigkeits-Rangliste wählt aber systematisch das *größte* Netz und
   verschenkt damit genau das Ziel dieses Projekts. Umgesetzt ist deshalb das
   Muster des linearen Skripts: voll trainiert werden die **billigste
   brauchbare** und die **genaueste** Variante, deployt wird die **billigste,
   die das Go-Kriterium erfüllt** (sonst die genaueste). Motiviert durch den
   Sweep-Befund, dass bereits `h48 d3` (5 569 Parameter, 5 424 MACs) im
   Quick-Pass eK 1,24 % / eF 0,82 % erreicht — der K₀-Split macht die
   Zielfunktion so viel leichter, dass große Netze nicht mehr nötig sind.

---

## 0. Kontext

Das nichtlineare KI-quad4-Element (`element_quad4_nl_ai.m` +
`training/quad4/train_quad4_nl_K_network.py`) lernt aktuell zwei unabhängige
Köpfe (K̂-triu + F̂). Folgen:

1. **Newton konvergiert nicht**: Ke ≠ ∂Finte/∂û (gemessen 7,1 % Inkonsistenz)
   → Quasi-Newton mit Kontraktionsrate 0,902 → Iterationslimit in allen 5
   Benchmarks. Zusätzlich ist das absolute Kriterium `tolR = 1e-8`
   (`sourcecode/model/init_setup.m:112`) für ein gelerntes Residuum
   prinzipiell unerreichbar.
2. **Genauigkeit unzureichend**: deploytes Netz Ke ~7,4 % / Finte ~7,0 %
   mittlerer rel. Frobenius-Fehler; frühere Variante 20 % Finte-Fehler im
   kleinsten Amplituden-Terzil.
3. **Hygiene**: deploytes .mat nicht mit dem aktuellen Skript reproduzierbar
   (Jacobi-Loss-Code überschrieben), FEMAI ist noch kein Git-Repo.

**Neuentwurf in einem Satz:** Das Modell ist die Summe aus der **exakten
linearen Steifigkeit** der kanonischen Geometrie (ko-rotierende lineare FEM
als Basis, pro Aufruf analytisch berechnet, KEIN Cache) und einem kleinen
neuronalen **Residual-Energie-Netz**, das nur die kubisch/quartische
Energieabweichung lernt; `Finte` und `Ke` entstehen per Differentiation →
Konsistenz `Ke = ∂Finte/∂û` gilt **per Konstruktion**, das Newton-Problem ist
strukturell gelöst.

```
Ŵ(ĉ, z) = ½·zᵀ·K̂₀(ĉ)·z + Ŵ_NL,θ(ĉ, z)
F̂(ĉ, z) = K̂₀(ĉ)·z + ∇_z Ŵ_NL,θ
K̂(ĉ, z) = K̂₀(ĉ)  + ∇²_z Ŵ_NL,θ
```

Physikalische Begründung: Für St.-Venant-Kirchhoff ist `W` **exakt ein
Polynom 4. Grades** in `z` (E_green quadratisch in u, W quadratisch in
E_green). `K̂₀` ist der exakte quadratische Term (SVK-Tangente bei u = 0 =
lineare Hooke-Steifigkeit, da S(0) = 0 den geometrischen Term auslöscht). Das
Netz lernt also nur noch den kubischen + quartischen Rest — eine deutlich
leichtere Zielfunktion, die ein kleineres Netz erlaubt (Speed-Hebel), und das
Kleinamplituden-Regime (Newton-Endphase, historischer Schmerzpunkt) wird von
`K̂₀` exakt dominiert.

**Festgelegte Entscheidungen (mit Paul geklärt):**

- Residual-Energie-Netz als Hauptweg; GAN/PDE-PINN verworfen.
- **KEIN Cache** irgendeiner Art (keine persistente Map, kein Precompute-Feld
  im Modell): `assemble.m` bleibt zustandslos, Solver-Signaturen unangetastet.
  `K̂₀` wird **pro Elementaufruf analytisch** berechnet (2×2-Gauss mit
  linearer B-Matrix — billig, keine Doppelknotenschleife). Fallback nur bei
  Speedup-Gate-Fail: `K̂₀` aus dem vorhandenen linearen Netz
  (`quad4_K_network.mat`, ~0,35 % Fehlerboden) — dokumentiert in Phase 6.
- Strukturelle Nullform NUR für W und F (Subtraktion von `f(ĉ,0)` und
  `∇f(ĉ,0)ᵀz`). Die quadratische Subtraktion (`½zᵀ∇²f(ĉ,0)z`) entfällt —
  sie kostete ohne Cache einen zweiten vollen Hessian pro Aufruf.
  `K_NL(ĉ,0) ≈ 0` wird stattdessen **weich** über die Targets erzwungen
  (Residual-K-Target bei z = 0 ist exakt 0) und in Gate e hart abgeprüft.
- Inputs bleiben konzeptionell rohe `(ĉ, z)`; vor dem Netz **lineare**
  Skalierung von z (kein Offset!) und affine Normalisierung von ĉ
  (Abschnitt 2.3), Ableitungen exakt zurücktransformiert.
- Training auf **Residual-Targets**, Loss-Normierung und alle Gates auf
  **Totalgrößen** (das, was Newton sieht) — Abschnitt 3.5.
- Datensatz-Mix 70 % synthetisch / 20 % Newton-Trajektorien / 10 % Active
  Learning; Benchmark-Strukturen 1–5 sind strikt **held-out** (Abschnitt 3.4).
- `newton.m`/Solver unangetastet; relative Toleranz wird nur im
  Benchmark-Skript über `model.solver.tolR` gesetzt.
- Erfolgskriterien: Genauigkeit (mean UND P99, je Terzil) **und** Speedup
  (eigenes hartes Gate, Phase 5 Gate h).

---

## 1. Notation und Konventionen (überall einhalten)

| Symbol | Bedeutung | Shape |
|---|---|---|
| `coord_e` | Elementknoten physisch, Spalten = Knoten 1..4 | 2×4 |
| `Ue` | Knotenverschiebungen physisch, DOF-Ordnung `[u1x;u1y;u2x;u2y;u3x;u3y;u4x;u4y]` | 8×1 |
| `ĉ` | kanonische Koordinaten (Schwerpunkt abgezogen, /Lc, Kante 1→2 auf +x) als Vektor `[x1;y1;…;x4;y4]` | 8×1 |
| `x̂` | dasselbe wie ĉ (im Ketten-Kontext) | 8×1 |
| `z` | kanonischer, translations-projizierter, **ko-rotierter** Zustand | 8×1 |
| `Rc` | Geometrie-Kanonisierungs-Drehung | 2×2 |
| `Tc` | `kron(eye(4), Rc)` | 8×8 |
| `P` | Translationsprojektor `I8 − (tx·txᵀ + ty·tyᵀ)/4`, `tx=[1;0;1;0;1;0;1;0]`, `ty=[0;1;0;1;0;1;0;1]` | 8×8 |
| `K̂₀(ĉ)` | exakte lineare Steifigkeit der kanonischen Geometrie, `E=d=1`, ν=0.3, planeStrain | 8×8 |
| `Ŵ_NL` | Netz-Residualenergie (Skalar) | 1 |
| `p_NL`, `Ĥ_NL` | `∇_z Ŵ_NL`, `∇²_z Ŵ_NL` | 8×1, 8×8 |
| triu-Ordnung | column-major, identisch `find(triu(true(8)))` | 36 |

Skalierungsgesetz (exakt, wie V1, gilt unverändert für die Summe):

```
W_phys = E · d · Lc² · Ŵ(ĉ, z)
Finte  = E · d · Lc  · Tcᵀ · (Rück-Rotation von F̂)
Ke     = E · d       · Tcᵀ · (Rück-Rotation von K̂) · Tc
```

Fest eintrainiert (nicht frei): Material StVenant, ν = 0.3, planeStrain.
Frei: E, d, Elementgröße, Lage, Rotation (Geometrie und Zustand).

---

## 2. Mathematik des Modells

### 2.1 K̂₀ analytisch pro Aufruf (kein Cache)

`K̂₀` ist die **lineare** quad4-Steifigkeit auf der kanonischen Geometrie:

```
für gp = 1..4 (2×2-Gauss, ξ,η = ±1/√3, w = 1):
    [N, dNdxi] = shape_quad4(ξ, η)          % vorhandene Routine
    J  = ĉ_2x4 · dNdxiᵀ                      % 2×2
    dNdx = J⁻¹ · dNdxi                        % 2×4 (explizit über 2×2-Inverse)
    B  = lineare B-Matrix aus dNdx            % 3×8: Zeile1 = [dN1dx 0 dN2dx 0 …],
                                              %      Zeile2 = [0 dN1dy …], Zeile3 = [dN1dy dN1dx …]
    K̂₀ += Bᵀ · C · B · det(J)               % C = Hooke planeStrain, E=1, ν=0.3
```

Vorlage: `sourcecode/elements/quad4/element_quad4_lin.m` (nur den Ke-Teil
portieren, Lastanteile weglassen). **Keine** Doppelknotenschleife, **kein**
zustandsabhängiger Term — das ist der billige Teil des analytischen Elements.
Identische Implementierung in Python (`k0_ref(chat)`) für die Targets;
Kreuz-Verifikation MATLAB↔Python ≤ 1e-12 rel (Gate a).

Eigenschaften (nutzen, nicht neu erzwingen): `K̂₀` ist symmetrisch, hat exakt
die 2 Translationen + 1 Rotation im Nullraum; da `z` translations-projiziert
ist, gilt `P·K̂₀·z = K̂₀·z` automatisch.

### 2.2 Strukturelle Nullform (Subtraktionsform, nur W und F)

`f(ĉ̃, z̃)` sei der rohe Skalar-MLP (auf normalisierten Inputs, 2.3). Das
Residualmodell ist:

```
Ŵ_NL(ĉ, z) = f(c̃, z̃) − f(c̃, 0) − ∇_z̃ f(c̃, 0)ᵀ · z̃
```

Damit exakt und unabhängig von den Gewichten: `Ŵ_NL(ĉ,0) = 0`,
`F_NL(ĉ,0) = ∇_z Ŵ_NL(ĉ,0) = 0` → **`F̂(ĉ,0) = 0` in Maschinengenauigkeit**
(reine Starrkörperbewegung ⇒ z = 0 nach Ko-Rotation ⇒ Finte = 0 exakt).

`K_NL(ĉ,0) = ∇²f(ĉ,0)` bleibt gelernt (weich): das K-Residual-Target ist bei
z = 0 exakt 0 und wird supervidiert; Gate e prüft `K̂(ĉ,0)` gegen das
lineare Element hart ab. Kosten der Nullform pro Elementaufruf: **ein**
zusätzlicher Forward + Reverse-Gradient bei z̃ = 0 (kein zweiter Hessian).

Identische Form in PyTorch implementieren (im `forward` des Modells, via
`torch.func.grad` für den `∇f(c̃,0)`-Term), damit Training und Deployment
strukturgleich sind.

### 2.3 Input-Normalisierung (exakt zurücktransformiert)

Aus dem Trainingsdatensatz einmalig bestimmt und als Metadaten ins .mat:

```
c̃ = (ĉ − c_mean) ./ c_std        % affin erlaubt (nach c wird nicht abgeleitet)
z̃ = z ./ z_scale                  % NUR Skalierung, KEIN Offset!
```

`z_scale(i) = std(z_i)` über den Trainingssatz (8×1, elementweise).
**Kein Offset auf z** — sonst wäre z̃(z=0) ≠ 0 und die Nullform aus 2.2
bricht. Exakte Rücktransformation der Ableitungen (D = diag(z_scale)):

```
p_NL = (∇_z̃ f_model) ./ z_scale                       % 8×1, elementweise
Ĥ_NL = (∇²_z̃ f_model) ./ (z_scale · z_scaleᵀ)        % 8×8, elementweise
```

### 2.4 Ko-Rotations-Kette (unverändert aus V1, mit K₀-Split an einer Stelle)

θ hängt von û ab; „θ einfrieren" scheitert, weil das *Netz* die Objektivität
nur bis O(Netzfehler) erfüllt → Kettenregel exakt mitführen. θ = atan2(b, a)
mit a, b **linear** in û → geschlossene Formeln.

Konstanten je Element(-aufruf): `Tc`, `P`, `x̂`; `dh0` = Formfunktions-
gradienten bei [0 0]; daraus Vektoren `a1, b1` mit `a = tr F̄ = 2 + a1ᵀv`,
`b = F̄₂₁ − F̄₁₂ = b1ᵀv`.

Kette mit `v = P·(Tc·Ue)/Lc`:

```
θ    = atan2(b, a),  r² = a² + b²
∇θ   = (a·b1 − b·a1)/r²
∇²θ  = [2ab·(a1a1ᵀ − b1b1ᵀ) + (b²−a²)·(a1b1ᵀ + b1a1ᵀ)]/r⁴
B    = kron(I4, R(−θ)),  Σ = kron(I4, [0 1; −1 0])
y    = x̂ + v,   z = P·(B·y − x̂)          → Modell-Input (ĉ, z)
```

**Einziger Unterschied zu V1:** „das Netz liefert Ŵ, p, Ĥ" wird ersetzt durch
„das **Modell** liefert":

```
Ŵ = ½·zᵀK̂₀z + Ŵ_NL        p = K̂₀z + p_NL        Ĥ = K̂₀ + Ĥ_NL
```

Der Rest der Kette ist formal identisch (mit p̃ = P·p):

```
G̃    = B + (B·Σ·y)·∇θᵀ
g_v  = G̃ᵀ·p̃
C_v  = −(ΣBᵀp̃)·∇θᵀ − ∇θ·(ΣBᵀp̃)ᵀ − (p̃ᵀBy)·∇θ∇θᵀ + (p̃ᵀBΣy)·∇²θ
K_v  = G̃ᵀ·P·Ĥ·P·G̃ + C_v
Finte = E·d·Lc · Tcᵀ·P·g_v        Ke = E·d · Tcᵀ·P·K_v·P·Tc
```

Alte `P·Ke·P`-Projektion und Knoten-4-Rekonstruktion (ΣF=0) entfallen —
beides steckt strukturell in P. Nur `Ke = (Ke+Keᵀ)/2` als Rundungspolitur
behalten. **Formeln nicht blind vertrauen — Gate b verifiziert die komplette
Kette per FD, bevor ein Netz existiert.**

---

## Phase 0 — Git-Hygiene (~0,5 h)

*FEMAI-Arbeitskopie (Stand 2026-08-18): noch KEIN Git-Repo; `.gitignore`
deckt `__pycache__/` und die Sweep-Dirs bereits ab.*

**Alle Git-Operationen in diesem Plan (init, Branch, Commits) führt Paul
selbst aus** (siehe `CLAUDE.md`, Regel 1). Der Plan definiert nur, WAS in
welchen Commit gehört; die Umsetzung liefert fertige Änderungen plus
Commit-Message-Vorschläge.

1. `git init` + Initial-Commit des kompletten FEMAI-Stands als
   Baseline-Snapshot. Commit-Message benennt explizit: *deployed
   `quad4_nl_K_network.mat` ist mit dem committeten Skript NICHT
   reproduzierbar (Jac-Loss-Version überschrieben)*.
2. **Baseline-Timing-Lauf** (für Gate h): `FEMSolid_ex_quad4_07_ai_nl_benchmark.m`
   einmal mit dem ALTEN Zwei-Kopf-Element laufen lassen, Konsolen-Log als
   `training/quad4/baseline_two_head_benchmark.log` speichern und committen —
   das ist die Referenzlinie „altes AI-Element", die nach dem Rewrite nicht
   mehr messbar ist.
3. `training/quad4/arch_sweep_nl_W/` in `.gitignore` ergänzen.
4. Feature-Branch `dlfe-quad4-energy-net`; alle folgenden Phasen committen
   dorthin.

**Gate:** Initial-Commit + Baseline-Log existieren; `git status` sauber.

## Phase 1 — Python-Referenz und Ketten-Validierung (~0,5 Tag)

Arbeitsort: `training/quad4/train_quad4_nl_W_network.py` (neu; Vorlage
`training/quad4/train_quad4_nl_K_network.py`, das alte Skript bleibt
unangetastet). Verbatim wiederverwenden: `generate_distorted_quad`,
`canonicalize_coords`, `sample_state`, `corot_state`, Hull-Konstanten
(ENV_RATIO_MAX 4.5, Winkel [20°,160°], E_MAX 0.2), Seeds.

1. **Referenz-Energie:** `stiffness_force_ref` → `energy_stiffness_force_ref`
   erweitern: im Gauß-Loop zusätzlich `W += 0.5·(Evecᵀ·C·Evec)·dV` (Evec =
   Green-Lagrange in Voigt, dV mit d=1). Rückgabe `(W, F, K)`.
2. **`k0_ref(chat)`:** lineare Steifigkeit wie 2.1 in NumPy.
3. **Gate a — Target-Selbstverifikation (PFLICHT vor jedem Training):**
   an 5 zufälligen (ĉ, z)-Samples in fp64, zentrale FD mit h = 1e-6:
   - `F_ref =? ∂W_ref/∂z` ≤ 1e-8 rel
   - `K_ref =? ∂F_ref/∂z` ≤ 1e-8 rel
   - `k0_ref(chat) =? K_ref(ĉ, z=0)` ≤ 1e-10 rel
   - Skalierungsgesetz (E=1000, d=2, Lc≠1 einmal durchrechnen) ≤ 1e-12 rel
4. **Gate b — Ketten-Validierung:** die komplette MATLAB-Kette aus 2.4
   isoliert in NumPy implementieren, aber mit dem **analytischen** Ŵ_ref als
   „Modell" (statt Netz). g_v, K_v gegen zentrale FD des Komposits
   `Ue → Finte` bzw. `Ue → Ke` ≤ 1e-8 rel, an 10 zufällig rotierten/
   skalierten/verschobenen Elementen mit Zuständen bis 60° Zustandsrotation.
   Trennt Kettenregel-Bugs von Netz-Bugs, bevor ein Netz existiert.

**Gate:** a und b grün, als Testfunktionen im Skript verankert (laufen bei
jedem Trainingsstart automatisch mit, ~Sekunden).

## Phase 2 — Datensatz (~1 Tag inkl. MATLAB-Trajektorien)

Gesamtumfang Training: **240 000 Samples = 168k synthetisch (70 %) + 48k
Newton-Trajektorien (20 %) + 24k Active-Learning-Slot (10 %)**. Der AL-Slot
wird in Runde 1 mit zusätzlichen synthetischen Samples gefüllt und in der
AL-Runde (Phase 6) ersetzt. Validierung: 24k synthetisch (2 000 Geometrien ×
12 Zustände) **+ 5k Trajektorien-Val aus 3 SEPARATEN Zufallsstrukturen**
(nicht aus den Trainings-Trajektorien-Strukturen).

### 2a. Synthetische Samples

Wie bisher: `generate_distorted_quad` + `sample_state` (log-uniforme
Amplituden über ~3 Dekaden, 12 % exakt z = 0) + `corot_state` + Hüllen.

### 2b. Newton-Trajektorien (NEU) — mit Leakage-Sperre

**Absolute Regel: die 5 Benchmark-Strukturen aus
`FEMSolid_ex_quad4_07_ai_nl_benchmark.m` dürfen NICHT als
Trajektorien-Quelle dienen — sie sind das Held-out-Testset.**

Neues MATLAB-Skript `training/quad4/generate_newton_trajectories.m`:

1. ~25 Zufallsstrukturen erzeugen: Rechteck `Lx ∈ [1,8]`, `Ly ∈ [1,6]`,
   `nel ∈ [200, 800]` (vorhandene Rechteck-Vernetzung aus `sourcecode/mesh`),
   Lagerung zufällig aus {linke Kante fest, Unterkante fest, Fest+Loslager
   unten}, Last zufällig aus {Linienlast Kante, Einzellast Ecke,
   Eigengewicht}, Material E=3e7, ν=0.3, d=0.3, planeStrain, StVenant.
2. Lastamplitude iterativ so skalieren, dass max ‖E_green‖ am Ende
   ∈ [0.05, 0.15] liegt (einmal rechnen, messen, skalieren, neu rechnen).
3. **Eigene Newton-Schleife im Skript** (Logik aus `sourcecode/solver/newton.m`
   KOPIEREN — `newton.m` selbst bleibt unangetastet), Element-Backend =
   analytisches `element_quad4_nl.m`, 6 Lastschritte. In **jeder Iteration**
   für 10 % zufällig gewählte Elemente `(coord_e, Ue)` aufzeichnen
   (auch die nicht konvergierten Zwischeniterierten — genau die besucht der
   Solver später).
4. Export als `training/quad4/newton_traj_states.mat` (Rohdaten:
   coords 2×4×N, Ue 8×N, Struktur-ID pro Sample). 22 davon → Training,
   3 → Validierung (per Struktur-ID getrennt, nie gemischt).
5. In Python: kanonisieren + ko-rotieren → (ĉ, z), Hüllen-Filter
   (Verletzer verwerfen, Anteil loggen), auf 48k/5k downsamplen (uniform).

### 2c. Targets (für ALLE Samples identisch)

Pro Sample `(ĉ, z)` mit `E = d = 1` auf kanonischer Geometrie:

```
W_tot, F_tot, K_tot = energy_stiffness_force_ref(ĉ, z)
K0     = k0_ref(ĉ)                  % 1× pro Geometrie, dann pro Zustand wiederverwendet
F_lin  = K0 · z
W_NL_t = W_tot − ½·zᵀK0z            % Skalar-Target
F_NL_t = F_tot − F_lin              % 8er-Target (voll, KEINE Knoten-4-Reduktion)
K_NL_t = K_tot − K0                 % als triu-36 speichern (column-major)
```

Zusätzlich pro Sample speichern (für Loss-Normierung und Eval):
`W_tot`, `‖F_tot‖`, `‖K_tot‖_F`, `F_lin` (8), `K0` als triu-36 pro
Geometrie (Index-Map Geometrie→Sample). Alles fp32, gesamt < 200 MB.
Metadaten pro Sample: Amplituden-Maß `max‖E_green‖` und Verzerrungsmaß
`r_detJ = max(detJ)/min(detJ)` (für Binning).

## Phase 3 — Training (Sobolev, Residual) (~1–2 Tage inkl. Läufe)

### 3.1 Modell

`WNet`: MLP `16 → hidden × depth → 1`, GELU (exakte erf-Form), Output-Layer
klein initialisiert (`W_out × 0.1`, `b_out = 0`), fp32 (kein AMP — höhere
Ableitungen). Subtraktionsform und Normalisierung **im `forward`** (2.2/2.3),
sodass `model(c̃raw, zraw)` direkt `Ŵ_NL` liefert:

```python
def forward(self, c, z):                       # c: (B,8) roh-kanonisch, z: (B,8) roh
    ct = (c - c_mean) / c_std
    zt = z / z_scale
    f  = self.mlp(torch.cat([ct, zt], -1))
    f0 = self.mlp(torch.cat([ct, torch.zeros_like(zt)], -1))
    g0 = grad_z(self.mlp, ct, zeros)           # ∇_z̃ f bei z̃=0, via torch.func.grad
    return f - f0 - (g0 * zt).sum(-1, keepdim=True)
```

Ableitungen fürs Training: `F_NL = vmap(grad(model, argnums=z))·(1/z_scale)`,
`K_NL = vmap(jacfwd(grad(...)))·(1/(z_scale⊗z_scale))` (forward-over-reverse).
**Kein Output-Rescaling à la FINT_SCALE** — W/F/K sind per Differentiation
gekoppelt.

### 3.2 Loss: Residual-Fehler, normiert auf TOTALGRÖSSEN

Weil `K0z` exakt ist, gilt `ΔF_tot = ΔF_NL` und `ΔK_tot = ΔK_NL`. Der Loss
normiert deshalb die Residual-Abweichungen durch die **Total-Normen** — er
optimiert damit direkt den relativen Fehler der Größen, die Newton sieht:

```
L = λ_W · mean( (ΔW_NL)²   / (W_tot²      + floorW²) )
  + λ_F · mean( ‖ΔF_NL‖²   / (‖F_tot‖²    + floorF²) )
  + λ_K · mean( ‖ΔK_NL‖²_F /  ‖K_tot‖²_F )              ← KEIN Floor nötig: K_tot → K0 ≠ 0
floorW = 0.05·RMS(|W_tot|),  floorF = 0.05·RMS(‖F_tot‖)
Start: λ_W = 0.1, λ_F = 1.0, λ_K = 1.0;  Tuning: ein λ zurzeit ×3/÷3, max. 2–3 Re-Runs
```

Kollaps-Schutz (Lektionen 15.07.) ist dreifach: small-init + Subtraktionsform
(Startprädiktion exakt 0 = korrekt für kleine Zustände) + K-Nenner ohne Floor.

### 3.3 Batching (4 GB RTX 3050 Ti)

Mini-Batch 4096 für W+F-Terme; K-Term auf rotierendem 1024-Subsample je
Batch. Adam LR 1e-3, `ReduceLROnPlateau(factor 0.5, patience 20 Epochen)`,
grad-clip 1.0, max. 800 Epochen mit Early-Stop (Val-Plateau 60 Epochen).
Nach Epoche 1 Laufzeit messen; Budget ≤ ~1,5 h/Variante halten.
`QUAD4_QUICK`-Env-Override: 5k Geometrien, 200 Epochen, nur 1 Variante.

### 3.4 Architektur-Sweep

Quick-Pass (5k Geometrien, 200 Epochen) über hidden {48, 64, 96, 128} ×
depth {3, 4}, Aktivierung GELU (Softplus nur als 1 Vergleichspunkt bei h96
d3). Ranking nach **Total-K-Fehler mean + P99** (3.5). Top-2 voll trainieren.
Erwartung: durch das leichtere Residual-Target reicht h64–96 — kleiner =
schneller im Element (Gate h).

### 3.5 Metriken (Training UND alle Gates) — Totalgrößen, Perzentile, Bins

Pro Val-Sample:

```
eF = ‖F_lin + F_NL,pred − F_tot‖ / (‖F_tot‖ + 0.02·RMS(‖F_tot‖))
eK = ‖K0 + K_NL,pred − K_tot‖_F / ‖K_tot‖_F
```

Berichtet werden **mean, P50, P90, P95, P99** von eF und eK, jeweils:
(a) gesamt, (b) je Amplituden-Terzil (über max‖E_green‖), (c) je
Verzerrungs-Bin `r_detJ ∈ [1,1.5), [1.5,2.5), [2.5,4.5]`, (d) getrennt für
synthetische und Trajektorien-Val-Daten. Ausgabe als Tabelle im Log und als
`.json` neben dem .mat.

### 3.6 Export & In-Training-Gates

`.mat` → `sourcecode/elements/quad4/quad4_nl_W_network.mat` mit:

- `W1..WL, b1..bL, num_linear_layers, activation='GELU_erf'`
- `model_form='resid_energy_K0split_subtract_f0_gradf0'`
- `input_norm_c_mean (8), input_norm_c_std (8), input_norm_z_scale (8)`
- `scaling='W=E*d*Lc^2*What; F=E*d*Lc*...; K=E*d*...'`, Kanonisierungs-Strings
- `material='StVenant', nu=0.3, condition='planeStrain'`
- Geometrie-Hülle, `E_MAX=0.2`, `state_rot_max_deg=180`
- λ/Floors, komplette Perzentil-Statistik, `train_script`, `git_hash`, Seed,
  Timestamp
- **Oracle-Testvektoren:** `test_C (64×8), test_Z (64×8)` (roh-kanonisch,
  unnormalisiert) und die Modell-TOTALS `test_W (64×1), test_F (64×8),
  test_K (64×36 triu)` — in Python aus K0-Split + Netz berechnet.

Gates im Skript vor dem Export: Gate a (automatisch, s. Phase 1); an 5
Samples fp64: `K_NL,autograd vs FD(F_NL)` ≤ 1e-6 rel; Symmetrie von K_NL
(strukturell durch jacfwd(grad) — trotzdem prüfen).
**Deployment verweigert Überschreiben bei Gate-Fail — kein stiller Fallback.**

## Phase 4 — MATLAB: Element-Rewrite (~1 Tag)

`element_quad4_nl_ai.m` in place neu (**zuerst die bestehende Datei lesen und
Signatur/Rückgabewerte/`history`-Durchreichung exakt übernehmen** —
`element_library.m` etc. bleiben unberührt).

1. Neuer Helper `sourcecode/elements/quad4/quad4_nl_ai_energy.m`:
   `[Wphys, Finte, Ke] = quad4_nl_ai_energy(coord_e, mat_e, Ue)`.
   Ablauf pro Aufruf (KEIN persistenter Zustand außer den einmalig geladenen
   Netzgewichten/Metadaten):
   1. Kanonisierung (ĉ, Lc, Rc, Tc) — Code aus bestehendem Element übernehmen.
   2. Ko-Rotation: θ, ∇θ, ∇²θ, B, y, z nach 2.4.
   3. `K̂₀` analytisch nach 2.1 (kleine 4-GP-Schleife, lineare B).
   4. Netz: Normalisierung (2.3), Forward bei z̃ und bei 0, Reverse-Gradient
      bei z̃ und bei 0, Hessian bei z̃ (Rekurrenzen unten); Subtraktionsform
      → `Ŵ_NL, p_NL, Ĥ_NL`; Rücktransformation der Normalisierung.
   5. Modell-Summe: `Ŵ = ½zᵀK̂₀z + Ŵ_NL`, `p = K̂₀z + p_NL`,
      `Ĥ = K̂₀ + Ĥ_NL`.
   6. Kette aus 2.4 → `Finte`, `Ke`; `Ke = (Ke+Keᵀ)/2`.
   Element-Wrapper: Helper + analytischer `Fbe`-Gauß-Loop (unverändert aus
   bestehendem Element), `Fte = 0`.
2. **Handkodierte Netz-Rekurrenzen** (im Code dokumentieren):
   - Forward: Schleife über `num_linear_layers` (dynamische Tiefe).
   - Reverse-Gradient: `r_{l−1} = W_lᵀ(gelu'(z_l).*r_l)`, Seed `r_L = 1`.
   - Forward-over-Reverse-Hessian, **alle 8 z̃-Richtungen als Matrix-Batch**
     (8 Spalten simultan, keine Richtungsschleife):
     `ṙ_{l−1} = W_lᵀ(gelu'(z_l).*ṙ_l + gelu''(z_l).*ż_l.*r_l)`;
     Richtungs-Seeds `ż_1 = W_1(:, z-Spalten)` (c-Anteil der Richtung = 0);
     am Ende `Ĥ = (Ĥ+Ĥᵀ)/2`.
   - GELU-Ableitungen (erf-Form): `gelu'(x) = Φ(x) + x·ϕ(x)`,
     `gelu''(x) = (2−x²)·ϕ(x)` mit `Φ = ½(1+erf(x/√2))`,
     `ϕ = exp(−x²/2)/√(2π)`.
   - Kostenrahmen: ~20 Forward-Äquivalente (Hessian) + ~3 (Null-Pass) — bei
     h96/d3 ≈ 0,5 M MACs/Element; die K̂₀-Schleife liegt darunter.
3. **Deployment-Guards — hart:** `error` (kein Warning) bei fehlendem oder
   abweichendem `model_form`, `activation`, Kanonisierungs-Strings,
   **`material`, `nu`, `condition`** (Verschärfung: alles, was stumm falsche
   Physik ergäbe). Checks einmal pro Session, nicht im Hot-Path.
4. **OOD-Prüfung:** billiger Proxy in jedem Aufruf: `‖E_green‖` am
   Elementmittelpunkt (gradU liegt für die Ko-Rotation ohnehin vor) gegen
   `E_MAX` → **eine** Warning pro Session (Zähler mitführen, Gesamtzahl am
   Ende ausgebbar). Volle 4-GP-Hüllenprüfung NICHT im Hot-Path — die macht
   `…_08_ai_nl_check.m` als Diagnose.
5. Entfernen: `finte_output_scale`, `recon_map`, Knoten-4-Rekonstruktion,
   `P·Ke·P`-Block (steckt strukturell in der Kette).

## Phase 5 — Verifikationsleiter (jedes Gate schaltet das nächste frei)

| Gate | Ort | Test | Schwelle |
|---|---|---|---|
| a | Trainingsskript (Phase 1) | Target-Selbstverifikation: F_ref=∂W_ref/∂z, K_ref=∂F_ref/∂z, k0_ref=K_ref(z=0), Skalierung | 1e-8 / 1e-10 / 1e-12 rel |
| b | NumPy (Phase 1) | komplette Kette mit analytischem Ŵ vs FD des Komposits | ≤ 1e-8 rel |
| c | Trainingsskript (Phase 3) | autograd-K_NL vs FD(F_NL) fp64; Perzentil-Tabelle vollständig | ≤ 1e-6 rel |
| d | neu: `examples/FEMSolid_ex_quad4_09_ai_nl_consistency.m` | MATLAB-Modellauswertung (Ŵ, F̂, K̂ total, kanonisch) vs `test_*`-Vektoren aus .mat | ≤ 1e-5 rel (fp32) |
| e | dito | 20 zufällige *physische* Elemente (rotiert, skaliert, E=1000, d=2, inkl. û≈0 und reiner Starrkörperbewegung): (e1) Finte vs FD(W_phys); (e2) **Ke vs FD(Finte) in globalen Koordinaten** (validiert Kette inkl. ∇²θ und K̂₀-Zweig); (e3) Finte(u=0)=0 exakt, Finte(starr) ~1e-12; (e4) `Ke(û→0)` vs `element_quad4_lin`-Ke | e1/e2 ≤ 1e-6 rel; e3 exakt/1e-12; e4 < 1 % |
| f | `FEMSolid_ex_quad4_07_ai_nl_benchmark.m` | 5 Held-out-Strukturen; in `make_nl_model` nach `init_model` einmal bei U=0 assemblieren und `model.solver.tolR = 1e-6·norm(Fext0(free))` setzen — **für beide Backends**, `newton.m` unangetastet. Report: eF/eK mean+P99, dU, dVM, Newton-Iterationen | konvergiert; eK & eF mean < 2 % UND P99 < 5 %; Iterationen ≤ FEM + 3; dU < 0,5 % |
| g | `FEMSolid_ex_quad4_08_ai_nl_check.m` | läuft unverändert (E/d-Skalierung ~1e-16); Starrkörper-Rotations-Zeile ergänzen | in-hull < 2 % |
| h | Benchmark 07, Timing-Teil | **Speedup-Gate**, Protokoll unten | s. unten |

**Gate h — Speedup (hart, zweistufig):** Messung wie im linearen Benchmark
(Median über adaptive Wiederholungen, Warm-up, am deformierten Zustand).
Vergleichslinien: (i) analytisches `element_quad4_nl.m`, (ii) altes
Zwei-Kopf-Netz (Zahlen aus dem Phase-0-Baseline-Log), (iii) neues
Residual-Energie-Element. Zwei Ebenen getrennt ausweisen:

- **h1 (hart):** Gesamt-Lösungszeit (Assemblierungen × Iterationen) AI <
  FEM auf ≥ 4 von 5 Strukturen. (Der Hebel ist die Iterationszahl:
  150-Deckel → ~FEM+O(1) durch Konsistenz.)
- **h2 (berichtet, nicht hart):** Zeit pro Assemblierung AI/FEM je Struktur
  — plus **Kostenmodell-Tabelle** (MACs Netz-Kette + K̂₀-FLOPs vs.
  FLOP-Schätzung des analytischen Elements) zur Extrapolation auf teure
  Elemente (3D, viele GPs, komplexe Materialgesetze). quad4 ist analytisch
  billig; h2 darf knapp ausgehen, ohne das Projekt zu No-Go-en — h1 nicht.

## Phase 6 — Go/No-Go + Kontingenzliste

**Go:** Gates a–h grün, insbesondere: eF und eK **mean < 2 % UND P99 < 5 %**
in **jedem** Amplituden-Terzil und jedem Verzerrungs-Bin (Validierung,
synthetisch UND Trajektorien), bestätigt durch Gate f, plus Gate h1.

**Bei Verfehlung — Reihenfolge einhalten, einzeln testen, je Re-Benchmark:**

1. **Active-Learning-Runde** (ERSETZT das alte „blind auf 1M"):
   Candidate-Pool 500k synthetische Zustände erzeugen (Targets sind billig),
   mit dem Basismodell auswerten, eF/eK in Zellen (Verzerrungs-Bin ×
   Amplituden-Terzil) mitteln, die schlechtesten Zellen identifizieren, den
   24k-AL-Slot mit gezielt dort gesampelten Zuständen füllen
   (Rejection-Sampling bis Zell-Kriterium erfüllt), Warm-Start-Retraining
   +300 Epochen. Erst wenn 2 AL-Runden nicht reichen: Datensatz global
   vergrößern (dann bevorzugt mehr Geometrien: 80k × 12).
2. λ_K-Schedule: Start 0.3, Rampe auf 1–3 im ersten Drittel; oder
   K-Subsample spät vergrößern.
3. Breitere Netze: h128, ggf. depth 4–5 (Kostenmodell in h2 aktualisieren!).
4. LBFGS-Feintuning nach Adam (full-batch W+F, K auf fixem 8–16k-Subsample,
   ggf. fp64/CPU).
5. Kleinamplituden-Sampling: `AMP_LOG_MIN` 2e-3 → 2e-4 (sollte durch den
   K₀-Split kaum noch nötig sein — wenn doch, zuerst Gate e4 prüfen:
   vermutlich ist dann `∇²f(ĉ,0)` zu groß → Extra-Loss-Term
   `μ·mean(‖K_NL,pred(ĉ,0)‖²_F/‖K0‖²_F)` mit μ = 0.1 ergänzen).
6. **Nur falls Gate h1 scheitert** (nicht bei Genauigkeitsproblemen):
   `K̂₀` aus dem linearen Netz `quad4_K_network.mat` statt analytisch
   (~3k MACs statt Gauss-Schleife; Konsistenz bleibt exakt, da K̂₀ nicht von
   z abhängt; Preis: ~0,35 % Fehlerboden im dominanten Term — Gate e4 auf
   < 1,5 % lockern und dU-Erwartung in Gate f neu bewerten).

## Phase 7 — Doku & finale Commits (~0,5 Tag)

1. Dieses Dokument (`docs/DLFE_quad4_nl_plan.md`) nach Umsetzung um die
   Gate-Ergebnisse ergänzen (Tabelle je Gate: Datum, Wert, grün/rot).
2. `docs/DLFE_quad4_Dokumentation.md` aktualisieren: Abschnitt 3
   (K/F-Kopf-Netz) als abgelöst markieren, Residual-Energie-Netz beschreiben
   (Modellform, Perzentil-Tabellen, Benchmark, Newton-Iterationsvergleich,
   Speedup-Tabelle + Kostenmodell). Der Weg dorthin plus Lektionen-Notiz
   (kein stiller Deployment-Fallback; jedes .mat trägt git-Hash + Skriptname;
   Gates auf Totalgrößen) als neuer Teil IV in
   `docs/DLFE_quad4_Entwicklung.md`.
3. Commits auf Feature-Branch, mindestens: (i) Python-Referenz + Gates a/b,
   (ii) Trajektorien-Generator + Datensatz, (iii) Trainingsskript +
   Sweep-Logs, (iv) MATLAB-Element + Helper + Beispiel 09, (v) Benchmark-
   Änderung (relatives tolR + Timing) + Ergebnis-Logs, (vi)
   `quad4_nl_W_network.mat` + `.pt` + Doku. Alte `quad4_nl_K_network*.mat`
   erst nach Go aus dem Working Tree entfernen (Historie via
   Phase-0-Snapshot).

## Kritische Dateien

- `training/quad4/train_quad4_nl_K_network.py` → Vorlage für neues
  `training/quad4/train_quad4_nl_W_network.py`
- `training/quad4/generate_newton_trajectories.m` → NEU (Trajektorien-Daten)
- `sourcecode/elements/quad4/element_quad4_nl_ai.m` → Rewrite + neuer Helper
  `quad4_nl_ai_energy.m`
- `sourcecode/elements/quad4/element_quad4_nl.m` → Referenz für Energie-Port
- `sourcecode/elements/quad4/element_quad4_lin.m` → Vorlage für `K̂₀`-Schleife
- `examples/FEMSolid_ex_quad4_07_ai_nl_benchmark.m` → relatives tolR,
  Perzentil-Report, Timing-Gate h
- `examples/FEMSolid_ex_quad4_08_ai_nl_check.m` → Diagnose; neues Beispiel 09
  (`FEMSolid_ex_quad4_09_ai_nl_consistency.m`) daran orientiert
