# FEMAI

Eigenständiger Ausschnitt aus **FEM-Solid Edu** (Daniel Materna, TH OWL) mit
Fokus auf die **Deep Learned Finite Elements (DLFE)** für das quad4-Element
und das 8-Knoten-Hexaeder brick8: klassische analytische Elementroutine und
KI-Element (Backend `ai`), plus Trainingsskripte und Benchmarks.

Andere Elementtypen (bar2, tria3, tetra4) und Backends (`mex`, `vectorized`)
aus dem Ursprungsprojekt sind bewusst weggelassen. Die brick8-Elemente wurden
für dieses Repo neu geschrieben (siehe `docs/DLFE_brick8_plan.md`).

## Start in MATLAB

```matlab
startup
FEMSolid_ex_quad4_01_two_elements
```

## Struktur

```text
startup.m
sourcecode/
  elements/        Elementbibliothek, Dispatch (element_library.m, element_routine.m)
    quad4/          shape_quad4.m, element_quad4_lin[.m|_ai.m], element_quad4_nl[.m|_ai.m],
                    quad4_nl_ai_energy.m (Kette), quad4_nl_ai_model.m (Energienetz),
                    quad4_nl_ai_network_file.m (Netzdatei je Material),
                    quad4_K_network.mat, quad4_nl_W_network.mat (StVenant),
                    quad4_nl_W_network_NeoHookean1.mat (Neo-Hooke),
                    quad4_nl_ai_energy_gram.m + quad4_nl_W_network_gram.mat
                    (Metrik-Kette, Wahl per QUAD4_STATE_FORM=gram)
    brick8/         shape_brick8.m, element_brick8_lin.m, element_brick8_nl.m,
                    element_brick8_nl_ai.m, brick8_nl_ai_energy.m (Kette),
                    brick8_nl_ai_network_file.m, brick8_nl_W_network.mat
    dlfe/           gemeinsame DLFE-Bausteine 2D/3D: dlfe_canonical_frame,
                    dlfe_gram_energy (Metrik-Kette), dlfe_mlp (Wert/Gradient/
                    Hessian-Richtungen), dlfe_load_network, dlfe_mode_matrix
  solver/           Assemblierung, linearer Löser, Newton-Verfahren
  model/            Modellaufbau (init_model, init_setup, ...)
  material/         Materialgesetze (Hooke, StVenant, NeoHooke)
  mesh/             Rechteck- und Quader-Vernetzung (create_model_data_box,
                    box_face_load)
  postprocessing/   Ergebnisauswertung, Plots
  tools/            Hilfswerkzeuge
training/
  quad4/            train_quad4_K_network.py       (linear)
                    train_quad4_nl_W_network.py    (nichtlinear, StVenant, Voll-Energie)
                    quad4_nl_ref.py                Referenzmathematik + Gates a/b
                    generate_newton_trajectories.m Trainingsdaten aus echten Newton-Läufen
                    train_quad4_nl_W_network_neohooke.py  (nichtlinear, Neo-Hooke)
                    quad4_nh_ref.py                Neo-Hooke-Referenz + Gates a/b/a'
                    generate_newton_trajectories_neohooke.m  Neo-Hooke-Trajektorien
                    export_nh_oracle.m             MATLAB-Oracle fuer Gate a'
                    train_quad4_nl_W_network_gram.py  Metrik-/Gram-Netz (Phase 0 brick8)
                    -> schreiben ihre .mat direkt nach sourcecode/elements/quad4/
  brick8/           brick8_nl_ref.py (Referenz + Gates a/b/a'), export_brick8_oracle.m,
                    generate_newton_trajectories_brick8.m, train_brick8_nl_W_network.py
  common/           dlfe_gram.py (Metrik-Kette, Netz, Sobolev-Training, Export)
examples/
  FEMSolid_ex_quad4_01_two_elements.m       Basisbeispiel (Referenz-Backend)
  FEMSolid_ex_quad4_02_beam_nel.m           Balken, vernetzbar
  FEMSolid_ex_quad4_03_ai.m                 Klassisch vs. KI, linear
  FEMSolid_ex_quad4_06_ai_patch_distortion.m Patch-Test / Verzerrungsgrenzen
  FEMSolid_ex_quad4_07_ai_nl_benchmark.m    5 Baustrukturen, nichtlinear (Newton)
  FEMSolid_ex_quad4_08_ai_nl_check.m        Einzelelement-Check, nichtlinear
  FEMSolid_ex_quad4_09_ai_nl_consistency.m  Konsistenz-/Ketten-Verifikation (FD-Gates)
  FEMSolid_ex_quad4_10_ai_nl_large_deformation_benchmark.m 6 Strukturen, grosse Verformungen
  FEMSolid_ex_quad4_11_ai_nl_neohooke_benchmark.m Neo-Hooke: Rechteck stark gezogen/gedrueckt
  FEMSolid_ex_brick8_01_element_check.m     brick8 analytisch: FD, Patch-Test, == quad4 planeStrain
  FEMSolid_ex_brick8_07_ai_nl_benchmark.m   brick8 FEM vs. KI, 6 Strukturen (inkl. Torsion)
  FEMSolid_ex_brick8_08_ai_nl_check.m       brick8 Einzelelement-Check
  FEMSolid_ex_brick8_09_ai_nl_consistency.m brick8 Gates d/e
benchmarks/
  brick8_c/         Laufzeitvergleich in C (analytisch vs. KI), run_bench.sh
docs/
  DLFE_quad4_Dokumentation.md    AKTUELLER STAND: Methode, Architektur, Ergebnisse,
                                 offene Punkte — linear und nichtlinear
  DLFE_quad4_Entwicklung.md      Entwicklungsgeschichte: Hürden linear, Plan
                                 (Option A vs. B), Debug-Protokoll nichtlinear
  DLFE_quad4_nl_plan.md          Plan + Umsetzungsstand: nichtlineares Residual-
                                 Energie-Netz (K0-Split, Sobolev), Gate-Ergebnisse
  DLFE_brick8_plan.md            brick8: Metrik-Kette statt Ko-Rotation, Phase 0
                                 (quad4), Elemente, Training, Gates, Benchmarks
  DLFE_brick8_todo.md            Abarbeitungsliste (offene Schleifen, erledigt)
  figs/brick8/                   Grafiken aus FEMSolid_ex_brick8_07
```

## Training

Python-Abhängigkeiten: `torch`, `numpy`, `scipy`. Netz wird direkt in
`sourcecode/elements/quad4/*.mat` deployt (Pfad ist im Skript relativ zu
`training/quad4/` gesetzt):

```bash
cd training/quad4
python quad4_nl_ref.py                 # Gates a/b (Referenz + Kette, ohne Netz)
python train_quad4_K_network.py        # linear
python train_quad4_nl_W_network.py     # nichtlinear, StVenant
```

Neo-Hooke (separate Routine, Reihenfolge wichtig):

```bash
python quad4_nh_ref.py --oracle-states          # Zustaende fuer Gate a'
# MATLAB: export_nh_oracle                      # MATLAB-Element als Oracle
# MATLAB: generate_newton_trajectories_neohooke # Trajektorien (~30 min)
python train_quad4_nl_W_network_neohooke.py     # Gates a/b/a' + Training
```

Das Neo-Hooke-Netz wird nur deployt, wenn Gate c gruen UND das
Go-Kriterium erfuellt ist (`QUAD4_DEPLOY_FORCE=1` erzwingt es).

brick8 (StVenant, Metrik-Kette; Reihenfolge wichtig):

```bash
cd training/brick8
python brick8_nl_ref.py --oracle-states         # Zustaende fuer Gate a'
# MATLAB: export_brick8_oracle                  # MATLAB-Element als Oracle
# MATLAB: generate_newton_trajectories_brick8   # Trajektorien (~3 min)
python train_brick8_nl_W_network.py             # Gates a/b/a'/c + Training
# BRICK8_HIDDEN / BRICK8_DEPTH / BRICK8_EPOCHS / BRICK8_OUT / BRICK8_QUICK
```

quad4 mit Metrik-Kette (Alternative zur Ko-Rotation):

```bash
cd training/quad4
python train_quad4_nl_W_network_gram.py         # modal (Standard); QUAD4_GRAM_MODAL=0: Knoten-Gram
# MATLAB: setenv('QUAD4_STATE_FORM','gram'); clear all; FEMSolid_ex_quad4_09_...
```

Für den vollen Datenmix vorher in MATLAB `generate_newton_trajectories`
laufen lassen (20 % der Trainingsdaten stammen aus echten Newton-Läufen).
Schnelllauf zum Pipeline-Test: Umgebungsvariable `QUAD4_QUICK` setzen.

## Bekannter Stand (siehe docs/)

- **Linear (quad4):** Ke-Fehler ~0.3–1.3 % (nach Starrkörper-Projektion), gut benutzbar.
- **Nichtlinear quad4, Standard (Ko-Rotation, Voll-Energie-Netz GELU h32):**
  `Ke = ∂Finte/∂Ue` per Konstruktion. Benchmark 07 (20 % Netzverzerrung):
  87 / 89 Newton-Iterationen FEM / KI, dU ≤ 0,27 %, Ke-P99 bis 5,5 %,
  Gesamtlösung 1,2× schneller (MATLAB).
- **Nichtlinear quad4, Metrik-Kette (`QUAD4_STATE_FORM=gram`, modal h32):**
  etwa 3× genauer als die Ko-Rotation (Validierung eF 0,31 / 2,05 %),
  Benchmark 07 87 / 87 Iterationen, dU 0,03–0,09 %, in MATLAB ohne Speedup.
- **Nichtlinear brick8 (Metrik-Kette, h64 deployt):** Newton-Iterationen
  identisch zum analytischen Element (115 / 115 auf 6 Strukturen), dU
  0,14–0,25 %, Torsion 1,18 %. Go-Kriterium verfehlt (Validierung eF 3,4 %,
  eK 3,7 % im Mittel). In MATLAB 1,7× schneller, kompiliert (C) ist das
  analytische Element 5–11× schneller (`benchmarks/brick8_c/`).
- Details, Gates und offene Punkte: `docs/DLFE_brick8_plan.md`.

## License

MIT, siehe `LICENSE`. Ursprungsprojekt: Daniel Materna, Fachgebiet Mathematik
und Computersimulation, TH OWL.
