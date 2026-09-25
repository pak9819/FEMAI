# DLFE brick8: Übertragung des Energienetzes auf das 8-Knoten-Hexaeder

Stand: 25. September 2026

Dieses Dokument beschreibt Plan und Umsetzungsstand der Übertragung des
nichtlinearen quad4-Energienetzes (Voll-Energie, Sobolev-Training) auf das
brick8-Element. Es enthält die gemessenen Gate- und Benchmark-Ergebnisse.

---

## 1. Ausgangslage und Kernproblem

Das quad4-Element erreicht Objektivität über eine **Ko-Rotation**. Der mittlere
Starrkörperwinkel θ = atan2(b, a) wird herausgedreht, und seine erste und zweite
Ableitung sind in 2D geschlossen bekannt. In 3D gibt es dafür keine
geschlossene Form. Nötig wären die Polarzerlegung von F und deren Ableitungen
erster und zweiter Ordnung.

Die Lösung ersetzt die Ko-Rotation durch einen **Metrik-Eingang**. Für jedes
hyperelastische Material hängt die Elementenergie nur von C = FᵀF ab. C ist an
jedem Punkt eine lineare Funktion der Gram-Matrix der deformierten
Knotenpositionen. Ein Netz auf dieser Größe ist deshalb exakt objektiv, ohne
Polarzerlegung und für 2D wie 3D gleich.

## 2. Phase 0: Gram-Kette am quad4 validiert

Die Kette wurde zuerst in der bestehenden 2D-Pipeline getestet. Datensatz,
Loss, Gates und Benchmarks sind identisch zum Ko-Rotationsnetz. Getestet wurden
zwei Metrik-Formen:

- **Knoten-Gram:** D = triu(YYᵀ − XXᵀ) der zentrierten Knoten, 10 Einträge.
- **Modal-Gram:** D = triu(Lᵀ(YYᵀ − XXᵀ)L) mit L = [dh0 | γ] aus der
  kanonischen Geometrie. dh0 sind die Formfunktionsgradienten im Zentrum,
  γ die Hourglass-Vektoren nach Flanagan-Belytschko. Damit ist
  D = triu([F0ᵀF0 − I, F0ᵀq ; qᵀF0, qᵀq]): Der Netzeingang enthält direkt die
  Green-Lagrange-Dehnung im Zentrum, die Kopplung und die Hourglass-Metrik.
  Die Form ist verlustfrei (L hat vollen Rang auf dem Komplement der
  Translationen) und bleibt exakt invariant.

Validierungsfehler, jeweils relativ, Mittel / P99:

| quad4, gleicher Datensatz | Parameter | eF | eK |
|---|---|---|---|
| Ko-Rotation h32 (deployt) | 2 689 | 1,03 / 6,53 % | 1,49 / 8,41 % |
| K0-Split h48 (alt) | 5 569 | 0,39 / 2,77 % | 0,60 / 4,29 % |
| Knoten-Gram h32 | 2 753 | 2,54 / 17,7 % | 3,15 / 16,8 % |
| Knoten-Gram h48 | 5 665 | 1,82 / 13,7 % | 2,25 / 13,9 % |
| **Modal-Gram h32** | 2 625 | **0,31 / 2,05 %** | **0,49 / 2,73 %** |
| Modal-Gram h48 | 5 473 | 0,28 / 2,06 % | 0,46 / 2,54 % |

Die Knoten-Gram-Form verfehlt das Go-Kriterium deutlich. Die modale Form ist
bei gleicher Größe etwa 3× genauer als die Ko-Rotation und schlägt sogar den
K0-Split. Der Grund: Für affine Zustände sieht das Netz direkt die Dehnung,
und die geometrieabhängige Mischung übernimmt die exakte Transformation L.

MATLAB-Gates mit dem modalen Netz, Beispiel 09:

- **Gate d:** 5e-13 gegen das Python-Oracle.
- **e1, e2:** 1e-9.
- **e3:** Starrkörperbewegung liefert Finte = 0 auf 3e-15.
- **e4:** max 1,76 % (h32) bzw. 1,36 % (h48), Mittel 0,49 % bzw. 0,46 %. Die
  Schwelle 1 % wird verfehlt, das deployte Ko-Rotationsnetz verfehlt sie mit
  max 2,84 % / Mittel 1,20 % ebenfalls. Die Ausreißer sind stark verjüngte
  Testelemente.

Benchmark 07 mit Modal-Gram h32, 5 Strukturen, 20 % Netzverzerrung:

- **Newton:** 87 Iterationen, identisch zum analytischen Element.
- **Genauigkeit:** dU 0,03–0,09 %, Finte-P99 ≤ 1,5 %, Ke-P99 ≤ 1,3 %.
- **Gates:** K und G grün.
- **Zeit:** Auf freier CPU 0,95× gegenüber dem analytischen Element, also in
  MATLAB kein Speedup (Ko-Rotation: 1,20×). Details in Abschnitt 5.

Das Ko-Rotationsnetz bleibt Standard. Das Gram-Netz wird mit
`QUAD4_STATE_FORM=gram` gewählt (`quad4_nl_W_network_gram.mat` = Modal h32).

## 3. Phasen 1–3: brick8 analytisch, Referenz, Daten

**Elemente (neu geschrieben).** FEM-Solid Edu lag lokal nicht vor, deshalb
wurden die Elemente neu erstellt: `shape_brick8.m`, `element_brick8_lin.m` und
`element_brick8_nl.m` (Total Lagrange, B-Matrix-Form, 3D-Materialien aus
`material_elasticity`). Beispiel `FEMSolid_ex_brick8_01_element_check`:

| Prüfung | Ergebnis |
|---|---|
| Ke vs. FD(Finte), StVenant / NeoHooke | 5e-10 / 6e-10 |
| Ke_nl(u=0) = Ke_lin | 0 |
| Finte bei Starrkörperbewegung | 8e-17 |
| Patch-Test, verzerrtes 3×3×3-Netz | 1e-16 |
| Schicht mit uz = 0 gegen quad4 planeStrain, linear / nichtlinear | 2e-13 / 4e-16 |
| Kragbalken gegen Balkentheorie, 20×2×2 / 40×4×4 | 0,87 / 0,96 (Schubversteifung) |

**Python-Referenz** `training/brick8/brick8_nl_ref.py`. W, F und K entstehen
per torch.func aus der StVenant-Energie (fp64).

- **Gate a:** F und K gegen FD auf 5e-10 / 8e-11. K(0) gegen explizites B'CB
  auf 2e-16. Skalierung s³ auf 1e-15.
- **Gate b:** Energie als Funktion von D auf 6e-14. D_modal ist
  rotationsinvariant (4e-13) und bei Starrkörperbewegung 0 (9e-16). Der
  lineare Block entspricht 2E im Zentrum (9e-14).
- **Gate a':** Python gegen MATLAB `element_brick8_nl` auf 40 Zuständen:
  Finte 5e-14, Ke 1e-15.

**Newton-Trajektorien** `generate_newton_trajectories_brick8.m`:

- **Strukturen:** 60 zufällige Quader mit 80–400 Elementen und
  Innenknoten-Verzerrung bis 0,25.
- **Lagerungen:** Kragarm, Fußeinspannung, beidseitige Einspannung.
- **Lasten:** Querlast, Schub, Eigengewicht, Torsion, Zug.
- **Ergebnis:** 105 635 Zustände, davon 13 056 aus Validierungsstrukturen.
  Laufzeit wenige Minuten.

**Synthetische Daten.**

- **Geometrien:** Seitenverhältnisse, Taper, Scherung und Jitter in der Hülle
  detJ > 0, detJ-Verhältnis ≤ 4,5, Flächenwinkel 20–160°.
- **Zustände:** affin plus nicht-affin. Ab v2 enthält die Hälfte der Zustände
  explizite Hourglass-Moden (Biegung und Verwindung) in der Größenordnung des
  affinen Anteils.

## 4. Phase 4: Training brick8

- **Netz:** Eingang 24 (ĉ) + 28 (D_modal) = 52, GELU (erf), Subtraktionsform.
- **Verlust:** Sobolev-Loss wie quad4 (0,1·W + F + K).
- **Datensatz:** 240 000 Trainings- und 29 000 Validierungs-Samples.

Zwei Datensatz-Versionen:
- **v1:** 40 Trajektorien-Strukturen (davon eine mit Torsion), nicht-affiner
  Anteil als schwaches Knotenrauschen.
- **v2:** 60 Strukturen, Torsion mit 30 % gezogen, die Hälfte der
  synthetischen Zustände mit ausgeprägten Hourglass-Moden. Die v2-Validierung
  ist dadurch schwerer; v1- und v2-Zahlen sind nicht direkt vergleichbar.

Validierungsfehler (Mittel / P99, relativ), MACs pro Elementaufruf mit
der projizierten Hessian (Abschnitt 5):

| Netz | Daten | Parameter | MACs | eF gesamt | eK gesamt | eF Trajektorien | eK Trajektorien | Go |
|---|---|---|---|---|---|---|---|---|
| h32 d3 | v2 | 3 841 | ~155 000 | 6,19 / 26,0 % | 6,79 / 15,9 % | 3,24 / 10,6 % | 5,07 / 14,6 % | nein |
| **h64 d3 (deployt)** | v2 | 11 777 | ~414 000 | 3,42 / 15,2 % | 3,73 / 10,0 % | 1,89 / 6,4 % | 2,89 / 10,0 % | nein |
| h96 d3 | v1 | 23 809 | ~788 000 | 1,78 / 7,3 % | 2,79 / 7,7 % | 0,99 / 5,4 % | 1,30 / 4,0 % | nein |

Kein Netz erfüllt das Go-Kriterium (Mittel < 2 %, P99 < 5 %). Trainings-
und Validierungsfehler sind jeweils gleich groß: Die Netze sind
unteranpassend, nicht überangepasst. Läufe h96 d3 (v2) und h128 d4 (v2)
wurden abgebrochen; h64 wurde als ausreichend festgelegt und deployt
(`brick8_nl_W_network.mat`). h32, h64 und h96 v1 liegen unter eigenem
Namen daneben und sind mit `BRICK8_NET_FILE=<name>` wählbar.

## 5. Phase 5–6: MATLAB-KI-Element und Benchmarks

`sourcecode/elements/brick8/`: `element_brick8_nl_ai.m`, `brick8_nl_ai_energy.m`
(Kette) und `brick8_nl_ai_network_file.m`. Die gemeinsamen Bausteine liegen in
`sourcecode/elements/dlfe/`: Kanonisierung, MLP mit
Forward-over-Reverse-Hessian, Metrik-Kette und Loader. Der Loader prüft Material,
nu, Zustandsform und Eingangsgröße hart.

**Gates (h64, `FEMSolid_ex_brick8_09_ai_nl_consistency`):**

- **Gate d:** 2,5e-12 gegen das Python-Oracle.
- **e1, e2:** 7,8e-9 und 6,1e-10.
- **e3:** Starrkörperbewegung liefert Finte = 0 auf 2,4e-15.
- **e4:** Ke(u = 0) gegen das lineare Element max 1,41 %, Mittel 1,07 %
  (Schwelle 1 % verfehlt).

**Einzelelement (h64, `FEMSolid_ex_brick8_08_ai_nl_check`):** In der Hülle
liegt der Ke-Fehler bei 2,2–2,9 % und der Finte-Fehler bei ≤ 1,1 %. Außerhalb
der Hülle steigt der Fehler (Ke 5,5 % bei ‖E‖ = 0,20, 42 % bei 0,69), die
OOD-Warnung greift. Die Skalierung in E und Elementgröße ist exakt (1e-16).

**Benchmark 07 (h64, freie CPU):**

| Struktur | NEL | Iterationen FEM / KI | dU | dVM | Finte P99 | Ke P99 | Speedup Gesamt |
|---|---|---|---|---|---|---|---|
| Kragbalken Endquerlast | 48 | 24 / 24 | 0,15 % | 0,23 % | 3,8 % | 3,5 % | 0,90× |
| Kragbalken Eigengewicht | 40 | 19 / 19 | 0,25 % | 0,29 % | 4,0 % | 5,7 % | 1,32× |
| Block Schub | 64 | 15 / 15 | 0,15 % | 0,17 % | 5,6 % | 6,0 % | 1,68× |
| Kragbalken Axialzug | 40 | 15 / 15 | 0,22 % | 0,15 % | 2,9 % | 5,6 % | 1,64× |
| Torsionsstab | 108 | 27 / 27 | 1,18 % | 0,60 % | 35 % | 5,3 % | 1,70× |
| Platte | 120 | 15 / 15 | 0,14 % | 0,26 % | 7,0 % | 5,1 % | 1,68× |

Gate K grün (115 / 115 Iterationen), Gate Z grün (Median 1,66× in MATLAB),
Gate G rot. Gegenüber h96 v1 ist die Platte deutlich genauer (dU 0,50 →
0,14 %), der Torsionsstab schlechter (0,69 → 1,18 %). Grafiken in
`docs/figs/brick8/`.

**Projizierte Hessian ohne Rückwärts-Tangenten (2026-09-25).** Beim MLP ist
nur die Aktivierung nichtlinear, daher gilt exakt
`TᵀHT = Σ_l Zd_lᵀ · diag(gelu''(z_l) ⊙ r_l) · Zd_l` mit den
Vorwärts-Tangenten `Zd_l` und dem Rückwärtsgradienten `r_l`. Der
Rückwärts-Tangentenpass entfällt, Lage 1 multipliziert nur die 28 D-Spalten.
Gate-Werte bleiben bitgleich, das MATLAB-Element (h96) wird von ~100 auf
~79 µs schneller, die MACs sinken für h96 von ~1,28 Mio. auf ~0,79 Mio.

**Laufzeit pro Elementaufruf** (`benchmarks/brick8_c/run_bench.sh`, Apple
M5 Pro, freie CPU). Das C-Programm implementiert beide Elemente im gleichen
Stil (clang -O3, Ke als oberes Dreieck), das MLP zusätzlich mit
Accelerate-BLAS. Ke und Finte stimmen mit MATLAB auf 1e-14 überein.

| Netz | MACs KI | MATLAB analytisch | MATLAB KI | C analytisch | C KI Schleifen | C KI BLAS | C: KI / analytisch |
|---|---|---|---|---|---|---|---|
| h32 | ~155 000 | 89 µs | 41 µs | 1,85 µs | 11,7 µs | 9,9 µs | 5,4× |
| h64 | ~414 000 | 89 µs | 45 µs | 1,85 µs | 29,6 µs | 14,9 µs | 8,0× |
| h96 | ~788 000 | 89 µs | 51 µs | 1,87 µs | 56,6 µs | 20,5 µs | 11,0× |

Analytisches Element: etwa 40 000–55 000 MACs, Neo-Hooke analytisch in
MATLAB ~320 µs.

Interpretation: In MATLAB ist das KI-Element 1,8–2,2× schneller, weil das
analytische Element vom Interpreter-Overhead der Gauss-Schleife dominiert
wird (48× langsamer als in C). Kompiliert kehrt sich das um: Das analytische
St.-Venant-brick8 ist 5–11× schneller als jedes Netz mit Hessian. Ein echter
Laufzeitvorteil des DLFE-Ansatzes ist erst bei deutlich teureren Elementen
oder Materialgesetzen zu erwarten.

**quad4 zum Vergleich (Benchmark 07, freie CPU):** Ko-Rotation h32 1,20×
Speedup, 87 / 89 Iterationen (Struktur 5: 12 / 14), Gate G rot (Ke-P99
5,47 %); identisch auf einer unveränderten HEAD-Kopie gemessen, also kein
Effekt der neuen Verzweigung. Metrik-Kette (modal h32) 0,95×, 87 / 87
Iterationen, dU 0,03–0,09 %, Gate G grün.

## 6. Offene Punkte

- **Go-Kriterium brick8:** Kein Netz erfüllt es; die Netze sind
  unteranpassend. Hebel: größere Netze (Läufe h96/h128 v2 abgebrochen) oder
  ein quadratischer Kopf `W = ½ Dᵀ A(ĉ) D`, der für StVenant exakt ist.
- **Torsion:** Finte-P99 35 % beim Torsionsstab, dU 1,18 % mit h64.
- **e4 (gelernte lineare Tangente) < 1 %:** Bei quad4 modal (1,4–1,8 %) und
  brick8 (1,2–1,4 %) verfehlt. Der quadratische Kopf würde K0 exakt machen.
- **Laufzeit:** Kompiliert ist das analytische StVenant-Element schneller
  (siehe Abschnitt 5). Der Break-even liegt bei teureren Elementen/Materialien.
- **quad4 Ko-Rotation, Benchmark 07:** 87 / 89 Iterationen und Gate G rot
  (Ke-P99 5,47 %) mit der aktuellen Benchmark-Version (20 % Verzerrung); die
  Zahlen im Paper (87 / 87) stammen aus der früheren Version.
- **Active-Learning-Slot:** Der 10-%-Slot wird bisher synthetisch gefüllt.
- **Neo-Hooke-Netz für brick8:** Infrastruktur vorhanden, zurückgestellt.
