# DLFE quad4 — Entwicklungsgeschichte

Dieses Dokument sammelt **den Weg**: welche Probleme bei der Entwicklung der
KI-quad4-Elemente auftraten, welche Irrwege es gab und wie die Lösungen
gefunden wurden. Der **aktuelle Stand** (Methode, Architektur, Ergebnisse,
Gültigkeitsbereich, offene Punkte) steht in `DLFE_quad4_Dokumentation.md` —
wer nur wissen will, *was* die Elemente heute können, liest dort.

Chronologie:

| Zeitraum | Thema | Abschnitt |
|---|---|---|
| bis Anfang Juli 2026 | Lineares Element: drei Hürden | Teil I |
| 2026-07-08 | Nichtlinear: Plan und Design-Entscheidung (Option A vs. B) | Teil II |
| 2026-07-15 | Nichtlinear: Debug-Protokoll (Verteilungs-Lücke, Kollaps, Ko-Rotation) | Teil III |
| 2026-08-18 | Nichtlinear: Neuentwurf als Residual-Energie-Netz (K₀-Split) | Teil IV |

---

# Teil I — Lineares Element: drei Probleme und ihre Lösung

Der Ansatz (Netz lernt die ganze `Ke` aus der Geometrie) war nicht von Anfang
an brauchbar. Drei Hürden mussten überwunden werden; jede war ein qualitativer
Sprung, kein Feintuning.

## I.1 Kleine Matrixfehler → 44 % Verschiebungsfehler (Starrkörper-Projektion)

**Problem.** Ein Netz mit ~1–2 % Frobenius-Fehler auf `Ke` führte zu **44 %**
Fehler in den berechneten Verschiebungen. Ursache: Die analytische `Ke` hat einen
exakten Nullraum aus 3 Starrkörpermoden (2 Translationen + 1 Rotation). Das
gelernte `Ke` erfüllt das nur näherungsweise → die kleinen Fehler erzeugen
spurious Verspannung, die sich beim Lösen des Gleichungssystems massiv verstärkt.

**Lösung.** Die drei Moden sind aus der Geometrie **exakt** bekannt. Eine
symmetrische Projektion zwingt `Ke·Mode = 0` exakt:

```matlab
Ke ← P · Ke · P ,   P = I − Q·Q'
```

`Q` = orthonormale Basis der drei Moden. Da die Moden bereits paarweise
orthogonal sind (Schwerpunkt = Mittelwert), genügt **Spaltennormierung** statt
einer SVD.

**Ergebnis:** Verschiebungsfehler 44 % → **~1,3 % Mittel** (Median 0,6 %).
**Ohne diese Projektion ist der Ansatz unbrauchbar.**

## I.2 Zu ungenau bei Verzerrung (Kanonisierung + Datensatz)

**Problem.** Der erste Datensatz bestand nur aus leicht gestörten Rechtecken. Ein
Verzerrungs-Patch-Test (`…_06_ai_patch_distortion.m`) deckte auf: Bei geometrisch
verzerrten Elementen — dem Normalfall in realen Netzen — stieg der Fehler stark
an. Das zur Ecke gestauchte Element eskalierte (>30 %), lange bevor es numerisch
entartete. Die Analyse zeigte: **kein Kapazitätsproblem**, sondern reine Distanz
zur Trainingsverteilung. Hebel sind also Datensatz und Features, nicht Netzgröße.

**Lösung (Architektur bleibt unverändert):**

- **Rotations-Kanonisierung** als Feature — nimmt dem Netz die
  Rotationsdimension komplett ab, statt sie aus Augmentation „erraten" zu lassen.
- **Verzerrungs-getriebener Datensatz** (`generate_distorted_quad`): gezielt
  Trapez-Taper, Scherung, Seitenverhältnis, Jitter — mit einer harten,
  **wohlgestellten Hülle** (detJ-Verhältnis ≤ 4.5, Innenwinkel [20°, 160°]).
  Deckt Verzerrung breit ab statt nur near-rectangular.
- **Per-Sample-relativer Loss** (`rel_frob_loss`): jedes Element durch seine
  eigene `Ke`-Norm normiert → optimiert direkt die berichtete relative
  Frobenius-Metrik.

**Ergebnis:** Validierungsfehler ~1,9 % → **0,35 %**; Patch-Test-Grundfehler
1,45 % → **0,10 %** (≈14×). Innerhalb der Hülle steigt der Fehler mit der
Verzerrung nun deutlich flacher.

## I.3 Kaum schneller als das analytische Element (Netz verkleinern + Hot-Path)

**Problem.** Gegenüber dem klassischen Element brachte die KI-Variante zunächst
kaum Zeitersparnis: Das Netz war groß (Hidden=128, ~39 000 Parameter) und der
Element-Code machte bei *jedem* der tausenden Aufrufe vermeidbare Arbeit (u.a.
eine SVD pro Element).

> Einordnung: Ein 4-Knoten-Element ist analytisch ohnehin billig. Diese
> Optimierung holt das Maximum aus quad4 heraus; der DLFE-Ansatz spielt seinen
> Vorteil grundsätzlich erst bei *teuren* Elementen aus (komplexe Materialgesetze,
> viele Gausspunkte, nichtlinear, 3D).

**Lösung — Netzgröße (Sweep):** `train_quad4_K_network.py` trainiert in EINEM
Lauf ein Gitter aus Breite × Tiefe × Aktivierung und deployt automatisch die
**billigste** GELU-Variante unter dem Fehler-Schwellwert (Ergebnis: 3×32 statt
128er-Netz, ~4 700 statt ~39 000 Parameter). Das Element liest die Hidden-Größe
dynamisch — ein Größenwechsel erfordert keinen Code-Eingriff.

**Lösung — Hot-Path im Element** (alle verhaltensneutral):

| Maßnahme | Wirkung |
|----------|---------|
| `orth()`-SVD → Spaltennormierung | keine SVD mehr pro Element (größter Hebel) |
| Gewichte `persistent`, einmalig aufbereitet | keine Reshapes/Struct-Zugriffe pro Aufruf |
| Symmetrie-Rekonstruktion als ein Gather | Matrixaufbau ohne Allokation |
| Konsistenzcheck nur 1× pro Session | keine String-Operationen im Hot-Path |
| Last-Gauss-Schleife überspringen wenn keine Last | spart 4 `shape_quad4`-Auswertungen |

**Lösung — Trainings-Speed:** Voll-Batch statt Mini-Batch (Netz winzig, Training
war overhead-gebunden) → **~450 s → ~22 s** (~20×). Kostet etwas Genauigkeit
(Voll-Batch = weniger Optimierer-Updates), bewusst akzeptiert.

---

# Teil II — Nichtlinear: Plan und Design-Entscheidung (2026-07-08)

**Entscheidung (Paul, 2026-07-08):** Das nichtlineare KI-Element soll — wie das
lineare — direkt die **K-Matrix lernen**, nicht die B-Matrix. Gewählt wurde
**Option B (Finte mitgelernt)**; die damals existierende B-Matrix-Variante
(`train_quad4_nl_B_network.py`) wurde ersetzt und gelöscht.

Der ursprüngliche Plan ist hier auf die Teile mit bleibendem Wert gekürzt —
die umgesetzten technischen Details (Kanonisierung, Ko-Rotation, Datensatz,
Training, Element) sind im Stand-Dokument beschrieben.

## II.1 Die zentrale Design-Entscheidung: Woher kommt `Finte`?

Newton braucht neben `Ke` das Residuum `R = Finte − λ·Fext`. Zwei Optionen:

| | **Option A: `Finte` analytisch** (Plan-Empfehlung) | **Option B: `Finte` mitlernen** (gewählt) |
|---|---|---|
| Netz-Ausgang | 36 (nur K̂) | 42 (36 K̂ + 6 F̂ mit Σf=0-Struktur) |
| Residuum | **exakt** (analytische Gauss-Schleife) | gelernt (mit Netzfehler) |
| Konvergierte Lösung | **exakt wie analytisches Element** (bis Newton-Toleranz) — Ke-Fehler kosten nur Iterationen | Gleichgewicht verschoben: dU direkt ∝ Finte-Fehler |
| Newton-Konvergenz | quasi-Newton (inkonsistente Tangente): +1–3 Iterationen erwartet | ebenfalls inkonsistent, zusätzlich falsches Ziel |
| Element-Kosten/Iteration | 1 Forward + kleine GP-Schleife für Finte | 1 Forward, sonst nichts |
| Analogie zum linearen Element | „Lastvektoren bleiben analytisch" konsequent weitergedacht | maximaler DLFE-Charakter |

Der Plan empfahl **Option A** (Lösung per Konstruktion richtig, Netzfehler
kosten nur Iterationen); gewählt wurde **Option B** als „maximaler
DLFE-Charakter". Die in Teil III analysierten Probleme (Finte-Fehler geht
direkt in die Lösung ein; Newton-Konvergenzrate) sind genau der Preis dieser
Entscheidung — Option A bleibt als Rückfallebene dokumentiert.

## II.2 Risiken aus dem Plan — und was daraus wurde

| Risiko (Plan) | Ausgang |
|---|---|
| 16-dim Eingaberaum → Genauigkeit pro Parameter schlechter als linear | Bestätigt; größeres Netz (4×128) nötig, Fehler ~2–15 % statt ~0,35 % |
| Objektivität nur datenbasiert, außerhalb `ROT_MAX = 45°` unkontrolliert | Trat als Kernproblem ein (Teil III); strukturell gelöst durch **Ko-Rotation** — `ROT_MAX` abgeschafft |
| „û = 0 und kleine Amplituden dicht sampeln" | War im ersten Sampler **nicht** umgesetzt → Verteilungs-Lücke (III.2); gelöst durch log-uniformes Amplituden-Sampling |
| Newton-Überschwingen verlässt die Zustands-Hülle temporär | Hüllen-Warnung im Element implementiert |
| Inkonsistente Tangente → Iterationsanstieg | Bestätigt und quantifiziert (III.6): Kontraktionsrate ~0,9, Newton erreicht 1e-8 nicht |

## II.3 Verworfene Alternativen (aus dem Plan)

- **B-Matrix-Variante** (war umgesetzt, wurde ersetzt): zustandsunabhängig,
  material-agnostisch, konsistente Tangente, cachebar — aber lernt eben nicht
  die K-Matrix (Nutzerentscheidung).
- **Energie-Netz** `W(ĉ,û)` mit `Finte = ∇W`, `Ke = ∂²W`: strukturell die
  sauberste Lösung (Konsistenz + Symmetrie + Konservativität garantiert),
  aber „lernt K nur indirekt" und braucht Gradient+Hessian-Propagation in
  MATLAB. Bleibt als spätere Ausbaustufe denkbar — der Datensatz
  (Geometrie × Zustand + Hüllen) wäre dafür 1:1 wiederverwendbar.
  *Stand August 2026: dies ist der aktuelle Ausgangspunkt für die
  Weiterentwicklung (siehe README / Stand-Dokument, offene Punkte).*

---

# Teil III — Nichtlinear: Debug-Protokoll (2026-07-15)

Dieser Teil dokumentiert chronologisch, wie aus „dU 114 %, Newton divergiert"
der aktuelle Stand (dU 0.4–7.8 %) wurde — **die konsolidierte Maßnahmenliste
steht in III.5**, der verbleibende offene Punkt (Newton-Konvergenzrate)
in III.6.

Ausgangsbefund des Benchmarks: **Finte-Fehler viel größer als Ke-Fehler,
Newton konvergiert nicht bzw. gegen eine falsche Lösung.** Die Einbindung des
Elements war dabei nachweislich korrekt (in Python nachgerechnet: auf
trainierten Zuständen ~1–2 % Fehler, E/d-Skalierung exakt). Zwei Ursachen:

## III.1 Erster Befund: das deployte Netz war eine veraltete Trainingsversion

Das zunächst deployte `quad4_nl_K_network.mat` (09.07., 09:15) stammte von
einer älteren Skriptversion, die den Finte-Loss nur **global** normierte (eine
Konstante für den ganzen Datensatz statt per-Sample). Der Umbau auf
per-Sample-relativen Loss + `FINT_SCALE` kam **nach** dem Deployment ins
Skript und war nie trainiert worden — erkennbar daran, dass im `.mat` das Feld
`finte_output_scale` fehlte. *(Behoben durch Neutraining am 15.07. — das
allein reichte aber nicht, siehe III.2: der Newton divergierte danach sogar
komplett, weil die eigentliche Ursache im Daten-Sampling liegt.)*

## III.2 Verteilungs-Lücke: das eigentliche Kernproblem

Warum trifft das ausgerechnet den Benchmark? Die Elementzustände an der
konvergierten Lösung (z. B. Kragträger, Struktur 1) sehen **fundamental anders
aus** als die Trainingszustände — in zwei Richtungen gleichzeitig:

| | Training (alter Sampler) | Benchmark (Struktur 1) |
|---|---|---|
| `‖E_green‖` median | 0.135 | **0.006** (~20× kleiner) |
| Rotation median | 2.3° | **9°** (~4× größer) |
| Anteil `‖E‖ < 0.02` | 0.6 % | Normalfall |
| Anteil „Rot > 5° **und** `‖E‖ < 0.02`" | **0.006 %** (1 : 17 000) | Normalfall |

Der alte Sampler skalierte mit `uniform(0.15, 1)` (keine kleinen Amplituden)
und bezog Rotation **und** Dehnung aus demselben `G ~ N(0, 0.15)` — beide
waren in der Größe gekoppelt. Ein Element in einem sich biegenden schlanken
Balken erlebt aber genau die gegenteilige Kombination: **viel
Starrkörperrotation bei winziger Dehnung.** Der Plan (Teil II) hatte
„superponierte Starrkörperrotation" und „kleine Amplituden dicht" explizit
gefordert — beides war im Sampler nicht umgesetzt.

In diesem untrainierten Regime dominiert der **absolute Rauschboden** des
Netzes:

* `Ke` bleibt gutartig (~2 % Fehler), denn `Ke → K_linear ≠ 0` für `u → 0` —
  der relative Fehler ist wohldefiniert und gelernt.
* `Finte → 0` für `u → 0`, der Netzfehler aber nicht: bei `û = 0` liefert das
  Netz spurioese Kräfte (`‖Finte‖ ≈ 0.3–0.5` bei `E = 1000` statt exakt 0).
  Gemessen an den kleinen Benchmark-Zuständen sind das **40–700 % relativer
  Fehler** pro Element.

Da `Finte` das Residuum definiert, verschiebt das den Gleichgewichtspunkt
massiv oder zerstört die Konvergenz ganz (Repro Netz vom 09.07.:
Tip-Durchbiegung −2.71 statt −1.38, dU ≈ 114 %; Netz vom 15.07. vormittags:
Newton divergiert komplett).

**Warum die Trainingsmetriken das nicht zeigten:** (a) die Validierungsdaten
kommen aus demselben Sampler — benchmark-artige Zustände kamen darin
schlicht nicht vor; (b) die Finte-Metrik hatte einen Floor bei `0.1·RMS`,
der Fehler an kleinen Zuständen systematisch dämpfte. Sweep-Werte von
„3 % Finte-Fehler" waren also korrekt — aber nur **innerhalb** der
gesampelten Verteilung gültig.

**Gegenprobe, dass es kein Benchmark-Bug ist:** dieselben Benchmark-Elemente,
dieselbe Pipeline, nur die Verschiebungsamplitude skaliert — bei ×2 (Dehnung
im Trainingsbereich) halbiert sich der Fehler, ab ×5 (Hülle verlassen)
explodiert er. Ein Einheiten-/Vorzeichen-/Reihenfolgefehler wäre
amplitudenunabhängig.

## III.3 Umgesetzte Fixes (15.07. vormittags)

Alle Fixes betreffen **Training und Metriken** — am MATLAB-Element war nichts
zu ändern.

1. **Amplituden log-uniform** (`sample_state`, `train_quad4_nl_K_network.py`):
   `u × exp(uniform(ln 2e-3, 0))` statt `u × uniform(0.15, 1)`. Jede
   Amplituden-Dekade von `‖E_green‖ ~ 1e-3` bis 0.2 bekommt gleich viele
   Samples — deckt Newton-Endphase und realistische Lastniveaus ab.

2. **Unabhängige Starrkörperrotation superponiert** (`sample_state`): auf 65 %
   der Zustände wird der deformierte Zustand um `θ` bis zum verbleibenden
   Hüllen-Spielraum gedreht (`u ← Rθ·(ĉ+u) − ĉ`). Entkoppelt Rotation von
   Dehnung; erzeugt auch reine Rotationszustände (`Finte = 0` exakt als
   Target). Exakt hüllenkonform, weil `E_green` unter `F → Rθ·F` invariant
   ist und sich der Polarwinkel exakt um `θ` verschiebt.
   *→ In Runde 2 **revidiert**: Rotation als Daten machte die Zielfunktion
   zu schwer (Kollaps); ersetzt durch die exakte Ko-Rotation (III.4).*

3. **`FINT_FLOOR` gesenkt**: `(0.2·RMS)² → (0.02·RMS)²`. Vorher wurden alle
   Zustände unter 20 % der RMS-Kraft absolut statt relativ gewichtet — genau
   das Benchmark-Regime. Ebenso Eval-Floor `FINTE_FLOOR_FRAC 0.1 → 0.02`,
   damit die berichtete Metrik den Kleinlast-Fehler nicht mehr verdeckt.
   *→ In Runde 2 auf `(0.05·RMS)²` korrigiert (III.4).*

4. **`STATES_PER_ELEM` 8 → 12**: der abgedeckte Zustandsraum ist um die
   Amplituden-Dekaden und die Rotationsachse gewachsen; Datenerzeugung und
   Voll-Batch-Training bleiben billig (240 000 Samples).

5. **Benchmark-Metrik mit Floor** (`elem_nl_error` in
   `…_07_ai_nl_benchmark.m`): relativer Finte-Fehler jetzt mit
   `Floor = 0.02·RMS(‖Finte‖)` wie im Training — fast unbelastete Elemente
   (freie Kragarm-Spitze) blähten den Mittelwert sonst beliebig auf
   (mean 92 % vs. median 41 % bei identischem Netz).

**Gemessene Wirkung auf die Datenabdeckung** (20 000 Sampler-Zustände,
kanonisches Quadrat; 0 Hüllen-Verletzungen):

| Kennzahl | vorher | nachher |
|---|---|---|
| Anteil `‖E‖ < 0.02` (Benchmark-Niveau) | 0.60 % | **63.9 %** |
| Anteil „Rot > 5° und `‖E‖ < 0.02`" (Biege-Regime) | 0.006 % | **37.3 %** |
| Rotation median | 2.3° | 10.7° |
| reine Rotationszustände | ~0 % | 6.5 % |

## III.4 Zweite Runde (15.07. nachmittags): Trainingskollaps → Ko-Rotation

Das Neutraining mit den Fixes aus III.3 lieferte **schlechtere** Zahlen
(Ke 12.7 %, „F(belastet)" 99.7 %). Die 99.7 % sind die Signatur eines
**Kollapses**: das Netz gibt für Finte konstant ≈ 0 aus (relativer Fehler
≈ 100 %). Der Trainingslog (Lauf 13:13) zeigt den Finte-Loss ab Epoche ~100
eingefroren bei 0.715 — 3900 Epochen ohne Bewegung — und die Lernrate wegen
des Plateaus schon bei Epoche ~1200 auf dem Minimum (dadurch fror auch das
Ke-Training ein).

**Zwei Ursachen, beide durch die Fixes aus III.3 selbst eingeführt:**

1. **Gradienten-Schock durch den kleinen Floor.** Beim zufälligen Init gibt
   das Netz O(1)-Werte aus; die skalierten Targets der (jetzt zahlreichen)
   kleinen Zustände sind winzig. Mit Floor `(0.02·RMS)²` explodieren die
   per-Sample-Nenner-Terme auf das ~2500-fache — der Finte-Loss startete bei
   ~6 statt ~1.2, und die brutalen Anfangsgradienten schlugen den Finte-Head
   dauerhaft auf „konstant 0".
2. **Rotation als Daten war der falsche Weg.** Die superponierten Rotationen
   (III.3, Fix 2) verlangten vom Netz, Objektivität über ±45° aus Daten zu
   lernen — die Zielfunktion wurde drastisch schwerer, während die
   Optimierung gleichzeitig instabiler wurde.

**Die Fixes der zweiten Runde:**

* **Ko-Rotation statt Rotations-Daten** (der strukturelle Fix; Methode im
  Stand-Dokument, Abschnitt 3.3): die Zustandsrotation wird exakt heraus- und
  wieder hineinrotiert — wie vorher schon Geometrie-Rotation, Größe und
  `E·d`. Das Netz lernt nur noch rotationsfreie Zustände, die
  Rotations-Augmentation im Sampler entfällt (rotierte Kopien ergäben
  identische Eingaben), und `ROT_MAX` ist als Laufzeitgrenze abgeschafft.
  Verifiziert: Identität gegen das analytische Element ~1e-15; komplette
  Element-Kette (Oracle-Test, Netz durch analytische Referenz ersetzt,
  Elemente frei rotiert/skaliert/verschoben, Zustände bis 60° rotiert) ~1e-13.
* **Klein-Init der Ausgabeschicht** (`W_out × 0.1`, `b_out = 0`): Start bei
  Vorhersage ≈ 0 ohne Gradienten-Schock.
* **Floor zurück auf `(0.05·RMS)²`**: Kompromiss — Relativ-Gewichtung bis
  hinab zu 5 % der RMS-Kraft, aber begrenzte Nenner-Spreizung
  (Verhältnis größter/kleinster Nenner ~400 statt ~2500).
* **Scheduler-Patience 40 → 100**, damit ein frühes Plateau die Lernrate
  nicht mehr tötet.

**Ergebnis (Neutraining + Benchmark, 15.07.):** Training läuft stabil durch,
und der Benchmark bestätigt den Fix — die Lösung ist jetzt richtig:

| Struktur | FintErr | KeErr | dU | dVM |
|---|---|---|---|---|
| Kragträger (Endquerlast) | 5.7 % | 2.2 % | **0.44 %** | 2.5 % |
| Kragträger (Eigengewicht) | 14.6 % | 2.1 % | 5.7 % | 5.9 % |
| Tiefer Kragträger (Querlast) | 4.0 % | 2.3 % | 7.8 % | 8.2 % |
| Scheibe (Scherung) | 5.6 % | 2.3 % | 2.8 % | 3.3 % |
| Kragträger (Axialzug) | 2.5 % | 2.6 % | 7.6 % | 2.7 % |

(Zum Vergleich der Ausgangslage: FintErr median 41–51 %, dU 114 % bzw.
Komplett-Divergenz.) Offen bleibt ein Punkt: itKI = 150 überall — Newton
erreicht das Konvergenzkriterium nicht (III.6).

## III.5 Wirksame Maßnahmen im Überblick

Die Verbesserung von „dU 114 % / Divergenz" auf „dU 0.4–7.8 %" kam aus
diesen Maßnahmen (chronologische Herleitung in III.1–III.4):

| # | Maßnahme | Wo | Wirkung |
|---|---|---|---|
| 1 | **Neutraining mit aktuellem Skriptstand** (per-Sample-relativer Finte-Loss, `finte_output_scale` korrekt deployt) | Training/Deployment | Voraussetzung — das alte `.mat` war eine veraltete Version mit global normiertem Loss (III.1) |
| 2 | **Log-uniformes Amplituden-Sampling** (`× exp(uniform(ln 2e-3, 0))` statt `uniform(0.15, 1)`) | `sample_state()` | Abdeckung des Benchmark-Regimes `‖E‖ < 0.02`: 0.6 % → 64 % der Trainingszustände (III.2/III.3) |
| 3 | **Ko-Rotation des Zustands** (exakt, in Training UND Element identisch) | `corot_state()` + `element_quad4_nl_ai.m` | Rotationsdimension komplett aus dem Lernproblem entfernt; Biege-Regime (viel Rotation, wenig Dehnung — vorher Häufigkeit 0.006 %) strukturell exakt abgedeckt; `ROT_MAX`-Laufzeitgrenze entfällt (III.4) |
| 4 | **Klein-Init der Ausgabeschicht** (`W_out×0.1`, `b_out=0`) | `KFNet` | verhindert den Gradienten-Schock am Trainingsstart, der den Finte-Head auf konstant 0 kollabieren ließ (III.4) |
| 5 | **Loss-Floor auf `(0.05·RMS)²`** (statt `(0.2)²` bzw. `(0.02)²`) | `FINT_FLOOR` | Relativ-Gewichtung bis 5 % der RMS-Kraft (Kleinlast-Genauigkeit), ohne die Nenner-Spreizung, die den Kollaps auslöste (III.3/III.4) |
| 6 | **Scheduler-Patience 40 → 100** | `train_one()` | Lernrate stirbt nicht mehr an einem frühen Plateau (vorher fror dadurch auch Ke ein) (III.4) |
| 7 | **12 statt 8 Zustände/Geometrie** | `STATES_PER_ELEM` | mehr Dichte im vergrößerten Amplitudenbereich (III.3) |
| 8 | **Benchmark-Metrik mit Floor** (`0.02·RMS`) | `elem_nl_error` | keine Modell-Verbesserung, aber ehrliche Messung — vorher blähten fast unbelastete Elemente den Mittelwert auf (mean 92 % vs. median 41 %) (III.3) |

Maßnahmen 2+3 tragen die Hauptlast (Verteilungs-Lücke geschlossen),
4–6 machen das Training auf dem neuen, härteren Datensatz überhaupt stabil,
1 und 8 sind Deployment- bzw. Mess-Hygiene.

## III.6 Offener Punkt: Newton erreicht tolR = 1e-8 nicht (Analyse 15.07.)

Der Benchmark zeigt itKI = 150 = 6 Lastschritte × MAX_ITER — Newton läuft
überall in den Deckel. Die Diagnose (Repro + FD-Jacobimatrix):

* Das Residuum **fällt monoton**, aber nur mit Rate ~0.9 pro Iteration
  (kein Rauschboden, keine Divergenz). Für 3e-2 → 1e-8 bräuchte es ~145
  Iterationen pro Schritt.
* Ursache ist die **Inkonsistenz der beiden Netz-Heads**: Newton sieht nicht
  den Ke-Fehler gegen das analytische Element (2.2 %), sondern die Abweichung
  zwischen Ke und der *tatsächlichen Jacobimatrix der gelernten
  Finte-Funktion* — gemessen **7.1 %** (FD am Benchmark-Endzustand).
* Der **weiche Biegemode** der schlanken Strukturen verstärkt das: der
  Spektralradius der Quasi-Newton-Iterationsmatrix `I − K⁻¹J` beträgt
  **0.902**, getragen von genau *einem* Eigenwert (der nächste: 0.56) —
  er stimmt exakt mit der beobachteten Residuen-Kontraktion überein.
* Die Lösung selbst ist davon **nicht** betroffen (dU 0.44 % bei Struktur 1):
  das Abbruch-Residuum ~1e-2 relativ zur Last liegt bereits in der
  Größenordnung des Finte-Modellfehlers — weiteres Iterieren unter die
  Modellgenauigkeit brächte physikalisch nichts.

Die daraus abgeleiteten (noch nicht umgesetzten) Abhilfen sind im
Stand-Dokument unter „offene Punkte" gelistet. Ein erster Versuch mit einem
Konsistenz-Loss (`‖∂F̂/∂û − K̂‖²` per Autodiff) verschlechterte die
Netzgenauigkeit, ohne die Konvergenzrate zu verbessern — das Grundproblem
bleibt offen.

### Einordnung: Warum ist Option B empfindlicher als der lineare Fall? (Stand Juli)

Linear bestimmte der `Ke`-Fehler die Lösung (44 %-Falle → Starrkörper-Projektion
als Fix, Teil I). Nichtlinear mit Option B bestimmt der **`Finte`-Fehler** die
Lösung — und `Finte` hat, anders als `Ke`, keinen festen „Sockel", an dem sich
ein relativer Fehler festhalten kann. Die Alternative **Option A** (`Finte`
analytisch per kleiner Gauss-Schleife, nur `Ke` aus dem Netz, Teil II) hätte
dieses Problem per Konstruktion nicht: die Lösung wäre exakt die des
analytischen Elements, Netzfehler kosten nur Newton-Iterationen. Option B
bleibt der „maximale DLFE-Charakter" — bezahlt mit genau der hier analysierten
Empfindlichkeit.

---

# Teil IV — Nichtlinear: Neuentwurf als Residual-Energie-Netz (2026-08-18)

Teil III endete mit einem funktionierenden, aber unbefriedigenden Zustand:
die Lösung war richtig (dU 0,4–7,8 %), doch Newton erreichte das
Konvergenzkriterium nie (Kontraktionsrate ~0,9, Iterationslimit in allen
5 Strukturen). Ursache war die **Inkonsistenz der beiden Netzköpfe**: `Ke`
war nur näherungsweise die Jacobi-Matrix der gelernten `Finte`-Funktion
(gemessen 7,1 %), und genau mit dieser Ableitung linearisiert Newton.

Ein Versuch, das über einen Konsistenz-Zusatzterm im Trainingsverlust zu
reparieren (`‖∂F̂/∂û − K̂‖²` per Autodiff), verschlechterte die Genauigkeit,
ohne die Konvergenzrate zu verbessern. Die Lehre daraus ist dieselbe wie
schon bei der Ko-Rotation in III.4: **eine exakt bekannte Struktur gehört
in die Konstruktion, nicht in den Verlust.**

## IV.1 Die Konstruktion: ein Potential statt zweier Köpfe

Statt `Ke` und `Finte` unabhängig zu lernen, lernt das Netz jetzt ein
**Skalarfeld** — die Formänderungsenergie — und beide Größen entstehen durch
Differentiation:

```
What(ĉ, z) = ½·zᵀ K̂₀(ĉ) z + Ŵ_NL(ĉ, z)
F̂ = ∂What/∂z = K̂₀z + ∇Ŵ_NL        K̂ = ∂²What/∂z² = K̂₀ + ∇²Ŵ_NL
```

Damit ist `Ke = ∂Finte/∂Ue` **per Konstruktion exakt** — die 7-%-Inkonsistenz
kann nicht mehr existieren. Symmetrie von `Ke` folgt gratis mit (Hessematrix
eines Skalars).

## IV.2 Der K₀-Split: das Netz lernt nur, was es lernen muss

Der entscheidende Zusatz gegenüber einem reinen Energie-Netz: der
**quadratische Anteil wird analytisch abgespalten**. `K̂₀` ist die exakte
lineare Steifigkeit der kanonischen Geometrie — für St.-Venant-Kirchhoff ist
das exakt die Tangente bei `u = 0` (`S(0) = 0` löscht den geometrischen Term).
Sie wird pro Elementaufruf mit einer billigen 4-Gausspunkt-Schleife berechnet;
ein **Cache wurde bewusst verworfen**, damit `assemble.m` zustandslos bleibt
und die Solver-Signaturen unangetastet.

Drei Gründe, warum der Split der Kern der Verbesserung ist:

1. **Er trifft genau den historischen Schmerzpunkt.** Das Kleinamplituden-
   Regime (Newton-Endphase), an dem die Zwei-Kopf-Variante mit 40–700 %
   relativem Finte-Fehler scheiterte (III.2), wird jetzt exakt von `K̂₀`
   dominiert. Der Netzanteil geht dort gegen null.
2. **Die Zielfunktion wird qualitativ leichter.** Für StVenant ist `What`
   exakt ein **Polynom 4. Grades** in `z`; der Split entfernt den
   quadratischen Term, das Netz lernt nur den kubisch/quartischen Rest.
   Leichteres Ziel = kleineres Netz = schnelleres Element — der Speed-Hebel.
3. **`Finte = 0` bei Starrkörperbewegung wird exakt.** Zusammen mit der
   Subtraktionsform `Ŵ_NL = f(ĉ,z) − f(ĉ,0) − ∇f(ĉ,0)ᵀz` gilt
   `F̂(ĉ,0) = 0` unabhängig von den Gewichten (gemessen: 1,4e-15).

Bewusst **nicht** übernommen wurde die härtere Variante, zusätzlich
`½zᵀ∇²f(ĉ,0)z` abzuziehen (was `K_T(ĉ,0) = K̂₀` exakt machen würde): ohne
Cache kostet das einen zweiten vollen Hessian pro Aufruf und damit genau die
Laufzeit, die der Ansatz gewinnen soll. `K_NL(ĉ,0) ≈ 0` bleibt deshalb die
einzige **weiche** Nebenbedingung — supervidiert über Targets, hart geprüft
in Gate e4.

## IV.3 Was beim Bauen wichtig war

**Die Ko-Rotation musste differenziert werden, nicht eingefroren.** `θ` hängt
von `Ue` ab. Das naheliegende Argument „Objektivität ⇒ θ darf eingefroren
werden" ist falsch, sobald ein *Netz* im Spiel ist: es erfüllt die Invarianz
nur bis O(Netzfehler), und die FD-Gates hätten das sofort aufgedeckt. Da `a`
und `b` in `atan2(b,a)` linear in `v` sind, gibt es geschlossene Formeln für
`∇θ` und `∇²θ` — die Kette wird exakt mitgeführt.

**Gates vor dem Netz.** Die komplette Kette wurde zuerst mit der
*analytischen* Energie als „Modell" gegen zentrale Differenzen und gegen das
analytische Element geprüft — bevor überhaupt ein Netz existierte. Das trennt
Kettenfehler von Lernfehlern; beide Gates waren im ersten Lauf grün
(Oracle-Vergleich 2,4e-13).

**Trainiert wird auf Residuen, gemessen auf Totalgrößen.** Weil `K̂₀z` exakt
ist, gilt `ΔF_tot = ΔF_NL` — der Loss normiert die Residual-Abweichungen
deshalb durch die *Total*-Normen und optimiert damit direkt den Fehler, den
Newton sieht. Ein Loss auf Residual-*Normen* hätte die Floor-Problematik
vom 15. Juli reproduziert.

**Leakage-Sperre bei den Trajektorien.** Newton-Zustände aus echten
FEM-Rechnungen sind als Trainingsdaten wertvoll (sie schließen genau die
Verteilungs-Lücke aus III.2), dürfen aber nicht aus den 5 Benchmark-Strukturen
stammen — sonst misst der Benchmark Memorierung statt Generalisierung. Der
Generator verwendet ausschließlich zufällige Strukturen; drei davon sind
zusätzlich als reine Validierungs-Strukturen reserviert. Erzeugt wurden
30 681 Zustände aus 25 Strukturen (Lastamplitude je Struktur so eingeregelt,
dass `max‖E_green‖` am Endzustand in [0,05; 0,15] liegt); nur 5 Zustände
fielen durch den Hüllen-Filter.

**Der Oracle-Test muss in fp64 rechnen.** Gate d vergleicht die
MATLAB-Netzauswertung gegen in Python berechnete Referenzvektoren. Mit
fp32-Export blieb ein Restfehler von ~1e-5 stehen — deutlich mehr als
fp32-Präzision (~6e-8) erwarten ließe. Ursache ist keine fehlerhafte
Implementierung, sondern **Auslöschung in `K̂₀·z`**: `K̂₀` hat einen
dreidimensionalen Nullraum, das Produkt ist eine Differenz betragsmäßig
größerer Terme, und der fp32-Rundungsfehler wird entsprechend verstärkt.
Werden die Oracle-Vektoren in fp64 erzeugt, fällt Gate d auf **4e-14** — der
Test misst dann tatsächlich die Übereinstimmung der Rekurrenzen und nicht die
Kondition eines Zwischenprodukts. Lehre: bei Oracle-Tests muss die Referenz
mindestens so genau sein wie das, was geprüft wird.

Zwei praktische Stolpersteine am Rande, beide dasselbe Muster: der erste
Trajektorien-Generator ließ die Sammel-Arrays per `cat`/`[...]` in der
Iterationsschleife wachsen — das kopiert quadratisch und dominierte die
Laufzeit vollständig; mit Präallokation lief derselbe Datensatz um ein
Vielfaches schneller. Und im Trainingsskript berechnete der W/F-Anteil des
Verlusts unnötig den (teuren) Hessian mit — nach dem Abschalten war ein
Trainingslauf gut 5× schneller bei identischem Verlustverlauf.

## IV.4 Ergebnis der Konstruktion

Die Verifikationsleiter (Details und Zahlen: `DLFE_quad4_nl_plan.md`,
Abschnitt „Umsetzungsstand") bestätigt in MATLAB, was der Ansatz strukturell
verspricht:

| Prüfung | Ergebnis |
|---|---|
| `Finte` vs. FD der Energie (physisch, frei rotiert/skaliert) | 9,2e-10 |
| `Ke` vs. FD von `Finte` (globale Koordinaten) | 8,8e-10 |
| `Finte` bei reiner Starrkörperbewegung | 1,4e-15 |
| Kette mit analytischer Energie vs. analytisches Element | 2,4e-13 |

Der praktische Effekt zeigte sich sofort im Solver-Integrationstest
(Kragträger, 24 Elemente, relatives Konvergenzkriterium): **Newton braucht
mit dem KI-Element exakt so viele Iterationen wie mit dem analytischen
Element** (9 vs. 9) — gegenüber „Iterationslimit in allen Strukturen" beim
Zwei-Kopf-Netz. Das war das Ziel des gesamten Neuentwurfs.

## IV.5 Ergebnis des ersten vollständigen Laufs

Der volle Benchmark bestätigte es auf allen fünf Strukturen: **identische
Iterationszahlen** (23/23, 22/22, 18/18, 12/12, 19/19), Element-Fehler von
Finte und Ke unter 0,31 % (auch im P99), Verschiebungsfehler 0,04–0,14 %.
Zum Vergleich der Ausgangslage aus Teil III: dU 0,44–7,8 % und
Iterationslimit überall.

Zwei Beobachtungen, die über das Erwartete hinausgehen:

**Das kleinste Netz genügt.** Deployt ist `h48 d3` mit 5 569 Parametern —
gegenüber 4×128 und ~57 000 Parametern der Zwei-Kopf-Variante. Der K₀-Split
vereinfacht die Zielfunktion so stark, dass ein Zehntel der Kapazität reicht;
das schlägt direkt auf die Assemblierungszeit durch. Die kostenbewusste
Auswahl im Sweep (billigste Variante, die das Go-Kriterium erfüllt) war
deshalb keine Kosmetik, sondern hat das Ergebnis bestimmt.

**Newton-Trajektorien sind die leichtere Teilmenge.** Auf den aus echten
FEM-Rechnungen gesammelten Zuständen liegt der Fehler bei eF 0,14 % / eK
0,17 %, auf den breit gestreuten synthetischen Daten bei 0,43 % / 0,66 %.
Der Zustandsraum, den der Löser tatsächlich besucht, ist also enger als der
trainierte — was erklärt, warum die Benchmark-Zahlen besser ausfallen als
die Validierungsmetrik.

**Und ein Befund, der Vorsicht verlangt:** Das Kostenmodell weist das
Energie-Element als arithmetisch **teurer** aus (~120 000 gegenüber ~3 600
MACs je Aufruf), gemessen ist es aber 1,41× schneller. Beides stimmt — der
Vorteil entsteht, weil die interpretierte Doppelknotenschleife des
analytischen Elements durch wenige dichte Matrixprodukte ersetzt wird, nicht
weil weniger gerechnet würde. Gegen eine vektorisierte oder kompilierte
Referenzimplementierung könnte das analytische quad4 gewinnen. Der gemessene
Speedup auf quad4 ist damit ein schwächeres Argument als er zunächst aussieht;
das tragfähige bleibt die Extrapolation auf teure Elemente, bei denen nur die
FEM-Seite mit Gausspunktzahl und Materialkomplexität wächst.
