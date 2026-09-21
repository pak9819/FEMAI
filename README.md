# FEMAI

Eigenständiger Ausschnitt aus **FEM-Solid Edu** (Daniel Materna, TH OWL) mit
Fokus auf die **Deep Learned Finite Elements (DLFE)** für das quad4-Element:
klassische analytische Elementroutine und KI-Element (Backend `ai`) jeweils
linear und nichtlinear, plus Trainingsskripte und Benchmarks.

Enthält nur, was zum Trainieren, Ausführen und Vergleichen der KI-quad4-Elemente
nötig ist. Andere Elementtypen (bar2, brick8) und Backends (`mex`,
`vectorized`) aus dem Ursprungsprojekt sind bewusst weggelassen.

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
                    quad4_nl_W_network_NeoHookean1.mat (Neo-Hooke)
  solver/           Assemblierung, linearer Löser, Newton-Verfahren
  model/            Modellaufbau (init_model, init_setup, ...)
  material/         Materialgesetze (Hooke, StVenant, NeoHooke)
  mesh/             Rechteck-Vernetzung
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
                    -> schreiben ihre .mat direkt nach sourcecode/elements/quad4/
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
docs/
  DLFE_quad4_Dokumentation.md    AKTUELLER STAND: Methode, Architektur, Ergebnisse,
                                 offene Punkte — linear und nichtlinear
  DLFE_quad4_Entwicklung.md      Entwicklungsgeschichte: Hürden linear, Plan
                                 (Option A vs. B), Debug-Protokoll nichtlinear
  DLFE_quad4_nl_plan.md          Plan + Umsetzungsstand: nichtlineares Residual-
                                 Energie-Netz (K0-Split, Sobolev), Gate-Ergebnisse
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

Für den vollen Datenmix vorher in MATLAB `generate_newton_trajectories`
laufen lassen (20 % der Trainingsdaten stammen aus echten Newton-Läufen).
Schnelllauf zum Pipeline-Test: Umgebungsvariable `QUAD4_QUICK` setzen.

## Bekannter Stand (siehe docs/)

- Linear: Ke-Fehler ~0.3–1.3 % (nach Starrkörper-Projektion), gut benutzbar.
- Nichtlinear: **Residual-Energie-Netz** (GELU h48/d3, 5 569 Parameter) — das
  Netz lernt ein Energiepotential, `Finte` und `Ke` entstehen durch
  Differentiation. Konsistenz `Ke = ∂Finte/∂Ue` gilt per Konstruktion (per FD
  auf ~1e-9 verifiziert). Benchmark über 5 Strukturen: Newton braucht
  **exakt so viele Iterationen wie das analytische Element**, dU 0,04–0,14 %,
  Element-Fehler ≤ 0,31 %, Gesamtlösung 1,10–1,64× schneller. Die frühere
  Zwei-Kopf-Variante (`quad4_nl_K_network.mat`) ist abgelöst.

## License

MIT, siehe `LICENSE`. Ursprungsprojekt: Daniel Materna, Fachgebiet Mathematik
und Computersimulation, TH OWL.
