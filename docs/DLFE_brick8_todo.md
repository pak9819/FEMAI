# brick8: Offene Schleifen schließen

Stand: 25. September 2026. Neo-Hooke ist bewusst ausgeklammert.

## Stand der Abarbeitung (25.09.2026)

| Abschnitt | Status | Abweichung vom Plan |
|---|---|---|
| 1 Hessian ohne Rückwärts-Tangenten | erledigt | Symmetrisches Produkt und 18 Richtungen (optional) nicht umgesetzt |
| 2 Training v2 | erledigt mit h32 und h64 | h96/h128 (v2) auf Wunsch abgebrochen, h64 deployt; kein Netz erfüllt das Go-Kriterium |
| 3 Verifikation, Benchmark | erledigt | 08 erstmals gelaufen; Grafiken in `docs/figs/brick8/` |
| 4 Kompilierte Laufzeit | erledigt | MATLAB Coder nicht lizenziert, kein mex-Compiler: stattdessen eigenständiges C-Programm (`benchmarks/brick8_c/`) mit clang -O3 und Accelerate-BLAS |
| 5 Doku, Hygiene | erledigt | quad4-Abweichung 87/89 auf HEAD-Kopie reproduziert, keine Regression |
| 6 Einchecken | erledigt | auf Branch `brick8-dlfe` statt direkt auf master |

Ergebnisse stehen in `docs/DLFE_brick8_plan.md`, Abschnitte 4 bis 6.

Reihenfolge ist so gewählt, dass jede Stufe die nächste absichert: erst die
Kette billiger machen (ändert Zahlen nicht), dann trainieren, dann messen,
dann dokumentieren und einchecken. Jede Aufgabe hat ein Abnahmekriterium.

Geschätzter Gesamtaufwand: 3 bis 4 Arbeitstage plus etwa 3 Stunden Rechenzeit.

---

## 1. Billigere Hessian in der MATLAB-Kette (½ Tag)

Ziel: MACs des KI-Elements etwa halbieren, Ergebnis bitgenau gleich.

**Was ändern**

- `sourcecode/elements/dlfe/dlfe_mlp.m`: Rückwärts-Tangentendurchlauf streichen.
  Beim MLP gilt für Tangentenmatrix `T` exakt

  ```
  Tᵀ·H·T = Σ_l  Zd_lᵀ · diag( gelu''(z_l) .* r_l ) · Zd_l
  ```

  `Zd_l` sind die Vorwärts-Tangenten, `r_l = df/da_l` der Rückwärtsgradient
  an den Aktivierungen. Beides liegt im jetzigen Code schon vor
  (`Zd{l}`, `D2{l}`, `r` vor der Multiplikation mit `D1{l}`). Rückgabe ist dann
  direkt die `ndir x ndir`-Matrix statt `H*T`.
- Symmetrisches Produkt nutzen: `A' * (w .* A)` mit `w >= 0` und `w < 0`
  getrennt als `B'*B` schreiben oder schlicht `A' * (w .* A)` lassen und
  danach symmetrisieren. Der BLAS-Vorteil ist zweitrangig, der Wegfall des
  Rückwärtspasses ist der große Posten.
- Lage 1 ohne Nullzeilen: `Zd_1 = W1(:, ndof+1:end) * Jfs` statt `W1 * [0; Jfs]`.
- `sourcecode/elements/dlfe/dlfe_gram_energy.m` anpassen: `Ku = Kmat + kron(...)`
  mit `Kmat` direkt aus `dlfe_mlp`.
- Optional, nur wenn Zeit: 18 statt 24 Richtungen. Basis `Q` (24x18) des
  Komplements der 6 Starrkörpermoden um die deformierte Lage per QR, dann
  `JᵀHJ = Q (QᵀJᵀHJ Q) Qᵀ`. Bringt weitere ~25 %, kostet eine QR pro Element.

**Abnahme**

- `FEMSolid_ex_quad4_09_ai_nl_consistency` mit `QUAD4_STATE_FORM=gram`:
  Gate d, e1, e2, e3 unverändert grün.
- `FEMSolid_ex_brick8_09_ai_nl_consistency`: Gate d ≤ 1e-10, e1/e2 ≤ 1e-6.
- Mikro-Timing (Skript aus der Session, `prof_b8.m`, liegt im Scratch-Ordner,
  sonst neu: 2000 Aufrufe `element_brick8_nl_ai` an einem Element):
  Zeit `dlfe_mlp` mit Hessian sinkt von ~32 µs deutlich, Ziel < 20 µs.
- MAC-Zählung in `docs/DLFE_brick8_plan.md` aktualisieren.

---

## 2. Training v2 zu Ende führen (1 Tag, davon ~1,5 h Rechenzeit)

Datensatz v2 (Hourglass-Sampler, 60 Trajektorienstrukturen, mehr Torsion)
liegt bereits im Cache `training/brick8/dataset_cache_brick8_*.npz`.
Abgebrochen wurden die Läufe h128d4 (v1 und v2).

**Läufe** (nacheinander, nicht parallel, sonst verfälschen sich die Zeiten
in den Logs und die CPU ist für MATLAB-Messungen blockiert):

```bash
cd training/brick8
BRICK8_HIDDEN=128 BRICK8_DEPTH=4 BRICK8_EPOCHS=800 \
  BRICK8_OUT=brick8_nl_W_network_h128d4.mat \
  ../../.venv/bin/python train_brick8_nl_W_network.py

BRICK8_HIDDEN=64  BRICK8_DEPTH=3 BRICK8_EPOCHS=600 \
  BRICK8_OUT=brick8_nl_W_network_h64d3.mat \
  ../../.venv/bin/python train_brick8_nl_W_network.py

BRICK8_HIDDEN=32  BRICK8_DEPTH=3 BRICK8_EPOCHS=600 \
  BRICK8_OUT=brick8_nl_W_network_h32d3.mat \
  ../../.venv/bin/python train_brick8_nl_W_network.py
```

h96d3 mit v2-Daten bei Bedarf ebenfalls, damit v1/v2 bei gleicher
Architektur vergleichbar sind (v1-Ergebnis h96: eF 1,78/7,27 %, eK 2,79/7,73 %).

**Auswahl**

- Deployen als `brick8_nl_W_network.mat` wird das kleinste Netz, das das
  Go-Kriterium erfüllt (eF, eK Mittel < 2 %, P99 < 5 %). Erfüllt keines das
  Kriterium: das genaueste, und im Plan-Dokument klar so benennen.
- Alle Kandidaten bleiben unter eigenem Namen in `sourcecode/elements/brick8/`
  und werden mit `BRICK8_NET_FILE=<name>` gewählt.

**Falls das Go-Kriterium weiter verfehlt wird**

- Fehler nach Bins prüfen (Ausgabe des Trainers): Ist der Fehler bei
  „Verzerrung >= 2.5“ konzentriert, Geometrie-Hülle im Sampler verengen
  (`ENV_RATIO_MAX` 4.5 → 3.5) und die Hülle im Loader mitziehen.
- Ist er über alle Bins gleich hoch: Kapazität ist das Problem, dann h160d4
  oder der quadratische Kopf (siehe Abschnitt 6, ist aber Forschung).

**Abnahme**

- Trainingslog mit Gate c grün und Metriken in
  `training/brick8/arch_sweep_nl_W/*_metrics.json`.
- Tabelle Architektur × (Parameter, MACs, eF, eK, Go) für das Plan-Dokument.

---

## 3. Verifikation und Benchmark mit dem neuen Netz (½ Tag)

Reihenfolge einhalten, jeweils mit `clear all` bzw. frischer MATLAB-Sitzung,
weil die Elemente das Netz persistent halten.

```matlab
startup
FEMSolid_ex_brick8_09_ai_nl_consistency   % Gates d, e1-e4
FEMSolid_ex_brick8_08_ai_nl_check         % Amplitudenrampe, Hülle, E/Lc-Skalierung (bisher NIE gelaufen)
FEMSolid_ex_brick8_07_ai_nl_benchmark     % K/G/Z, Grafiken
```

Für den Benchmark vorher `BRICK8_FIG_DIR` setzen, PNGs nach `docs/figs/brick8/`.
Die CPU muss dabei frei sein (kein Training parallel), sonst ist Gate Z wertlos.

**Abnahme**

- 09: d, e1, e2, e3 grün. e4 (gelernte Tangente bei u = 0) dokumentieren,
  auch wenn > 1 %; das ist ein bekannter, offener Punkt (Abschnitt 6).
- 08: Fehler bei amp = 0 nahe null, innerhalb der Hülle klein, außerhalb
  ansteigend. Skalierungschecks ~1e-15.
- 07: Gate K grün (Iterationen identisch), Gate G mit dem deployten Netz
  protokollieren, Gate Z auf freier CPU.

---

## 4. Faire Laufzeitmessung in kompiliertem Code (1 Tag)

Bisher misst der Benchmark MATLAB-Interpreter-Overhead: Das Netz braucht
~24× mehr MACs als das analytische Element und ist trotzdem 1,3× schneller.
Das ist als Speed-Aussage nicht haltbar.

**Vorgehen mit MATLAB Coder** (das Ursprungsprojekt hatte ein `mex`-Backend,
`material_elasticity.m` enthält bereits `coder.varsize`-Hinweise):

1. Analytisch: `codegen element_brick8_nl -args {zeros(8,3), zeros(1,4), zeros(3,1), 0, zeros(24,1), [], zeros(8,3), zeros(8,1), 'StVenant', '3D', struct()}`.
   Falls `material_elasticity` wegen Strings zickt: eine StVenant-only-Kopie
   `element_brick8_nl_sv.m` für die Messung, ohne `switch` über Materialnamen.
2. KI: eine codegen-fähige Variante `brick8_nl_ai_core(coord_e, Ue, Emod, W1, b1, W2, b2, W3, b3, W4, b4, c_mean, c_std, D_scale)`
   ohne `persistent`, ohne `load`, ohne Zellarrays (Tiefe 3 = 4 Gewichtslagen
   fest). `dlfe_mode_matrix`, Index-Tabellen aus `dlfe_load_network` als
   Konstanten inline.
3. Beide als MEX bauen, je 1e5 Aufrufe an einem Element messen, dazu eine
   Assemblierung der Benchmark-Struktur 5 (Torsion) mit beiden MEX.
4. Ergebnis als Tabelle: µs/Aufruf MATLAB vs. MEX, für analytisch und KI,
   h32 / h64 / h96 / h128. Dazu die MAC-Zahlen aus Abschnitt 1.

Erwartung: Kompiliert ist das analytische StVenant-Element schneller als
jedes Netz mit Hessian. Das ist kein Scheitern, sondern das ehrliche
Ergebnis, das den Break-even auf teurere Elemente und Materialien verschiebt.
Genau so ins Plan-Dokument schreiben.

**Abnahme**

- Beide MEX liefern Ke/Finte identisch zur MATLAB-Version (≤ 1e-12).
- Tabelle und zwei Sätze Interpretation im Plan-Dokument.

---

## 5. Dokumentation und Repo-Hygiene (½ Tag)

- `docs/DLFE_brick8_plan.md`: Platzhalter `ERGEBNISSE_BRICK8` und
  `BENCHMARK_BRICK8` durch die Tabellen aus Abschnitt 2 bis 4 ersetzen.
  Laufzeittabelle (Abschnitt 5 dort) mit den MEX-Zahlen ergänzen.
  Abschnitt „Offene Punkte“ auf den dann aktuellen Stand bringen.
- `README.md`, Abschnitt „Bekannter Stand“: ist veraltet (beschreibt noch das
  K0-Split-h48-Netz). Neu: quad4 Ko-Rotation h32 deployt, quad4 Metrik-Kette
  als Option mit `QUAD4_STATE_FORM=gram`, brick8 mit Ergebnis aus Abschnitt 3.
- quad4-Regressionslauf ohne Env-Variable: `FEMSolid_ex_quad4_09_ai_nl_consistency`
  und `FEMSolid_ex_quad4_07_ai_nl_benchmark` mit dem Standard-Netz einmal
  durchlaufen lassen. `quad4_nl_ai_energy.m` wurde um die Verzweigung auf
  die Gram-Kette erweitert, das darf den Standardpfad nicht verändert haben
  (Erwartung: 87 Iterationen, gleiche Zahlen wie im Paper).
- quad4-Benchmark 07 mit `QUAD4_STATE_FORM=gram` auf freier CPU wiederholen,
  damit Gate Z dort eine belastbare Zahl bekommt (bisher unter Last gemessen).
- Alte Sweep-Netze aufräumen: `sourcecode/elements/quad4/quad4_nl_W_network_gram_h32.mat`,
  `..._gram_h48.mat` (Knoten-Gram, unbrauchbar) löschen;
  `quad4_nl_W_network_gram_modal_h32.mat` ist identisch zu `quad4_nl_W_network_gram.mat`
  und kann weg, `..._modal_h48.mat` behalten oder nach `training/quad4/arch_sweep_nl_W_gram/` verschieben.
- `64_quad4_nl_W_network.mat` und `.DS_Store` sind versioniert, gehören aber
  nicht ins Repo. `.DS_Store` in `.gitignore`, beides mit `git rm --cached`.

---

## 6. Einchecken (¼ Tag)

Bisher ist **nichts** von der brick8-Arbeit committet. Sinnvoll sind drei
Commits, damit die Historie lesbar bleibt:

1. `dlfe: gemeinsame Metrik-Kette (2D/3D) + quad4 Gram-Variante`
   `sourcecode/elements/dlfe/*`, `sourcecode/elements/quad4/quad4_nl_ai_energy_gram.m`,
   Änderungen in `quad4_nl_ai_energy.m`, `quad4_nl_ai_network_file.m`,
   `examples/FEMSolid_ex_quad4_09_ai_nl_consistency.m`,
   `training/common/dlfe_gram.py`, `training/quad4/train_quad4_nl_W_network_gram.py`,
   `sourcecode/elements/quad4/quad4_nl_W_network_gram.mat`.
2. `brick8: analytische Elemente, Quader-Vernetzer, Referenz, Trajektorien`
   `sourcecode/elements/brick8/shape_brick8.m`, `element_brick8_lin.m`, `element_brick8_nl.m`,
   `sourcecode/mesh/create_model_data_box.m`, `box_face_load.m`,
   `training/brick8/brick8_nl_ref.py`, `export_brick8_oracle.m`,
   `generate_newton_trajectories_brick8.m`, `examples/FEMSolid_ex_brick8_01_element_check.m`.
3. `brick8: KI-Element, Training, Gates, Benchmark, Doku`
   `element_brick8_nl_ai.m`, `brick8_nl_ai_energy.m`, `brick8_nl_ai_network_file.m`,
   `brick8_nl_W_network.mat`, `training/brick8/train_brick8_nl_W_network.py`,
   `examples/FEMSolid_ex_brick8_0{7,8,9}_*.m`, `docs/DLFE_brick8_plan.md`,
   `README.md`, `startup.m` (Light-Theme), `.gitignore`.

Nicht mit einchecken, weil regenerierbar und in `.gitignore`: Datensatz-Caches,
Trajektorien-`.mat`, Oracle-Dateien, Sweep-Ordner. Vorher `git status` prüfen:
`paper.tex`, `docs/DLFE_quad4_Paper.md`, `docs/DLFE_quad4_neohooke_plan.md` und die
Änderung an `docs/DLFE_quad4_Dokumentation.md` waren schon vor dieser Arbeit
unversioniert und sind eine eigene Entscheidung.

---

## Bewusst offen gelassen (keine Schleife, sondern Forschung)

- **e4 < 1 %** (gelernte lineare Tangente): wird bei quad4-modal (1,4–1,8 %)
  und brick8 (1,2 %) verfehlt. Sauberer Fix ist der quadratische Kopf
  `W = ½ Dᵀ A(ĉ) D`, der K0 exakt macht und den Netz-Hessian eliminiert.
  Nur protokollieren.
- **Active-Learning-Slot**: seit dem ersten Plan vorgesehen, nie benutzt.
- **Neo-Hooke brick8**: Infrastruktur vorhanden, auf Wunsch zurückgestellt.
