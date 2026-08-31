# Umbau: Voll-Energie-Netz statt K₀-Split (nichtlineares KI-quad4)

**Stand: 2026-08-31 — Code umgebaut, Retraining steht aus.**

## Was geändert wurde

Das nichtlineare KI-Element lernte bisher nur die *nichtlineare
Energieabweichung* (Residual-Energie): der quadratische Anteil `½·zᵀK̂₀z`
wurde in jedem Elementaufruf analytisch berechnet (4-Gausspunkt-Schleife in
`quad4_nl_ai_model.m`). Neu sagt das Netz die **volle Energie** voraus —
quadratischer und nichtlinearer Anteil zusammen. Im Element findet damit
**keine numerische Energie-/Steifigkeitsberechnung mehr statt**; `Finte` und
`Ke` entstehen ausschließlich durch Differentiation des Netzes:

```
vorher:  What(ĉ,z) = ½·zᵀK̂₀(ĉ)z + Ŵ_NL(ĉ,z)     K̂₀ analytisch pro Aufruf
nachher: What(ĉ,z) = Ŵ_net(ĉ,z)                   alles aus dem Netz

beide:   F̂ = ∂What/∂z        K̂ = ∂²What/∂z²      (Konsistenz per Konstruktion)
```

## Geänderte Dateien

| Datei | Änderung |
|---|---|
| `training/quad4/train_quad4_nl_W_network.py` | Targets sind jetzt die Totalgrößen `W`, `F`, `K` statt der Residuen; Sobolev-Loss und Floors unverändert (der K-Nenner braucht weiterhin keinen Floor, da `‖K_tot‖ → ‖K_lin‖ ≠ 0`). Oracle-Export ohne K₀-Addition. `model_form = 'total_energy_subtract_f0_gradf0'`. Default-Architektur h64/d3 statt h48/d3. Datensatz-Cache bleibt kompatibel (`K0triu` wird weiter gespeichert, nur nicht mehr benutzt). |
| `sourcecode/elements/quad4/quad4_nl_ai_model.m` | `k0_canonical`-Schleife, `hooke_C` und die Modell-Summe entfernt — Wert, Gradient und Hessian kommen direkt aus dem Netz. Harter Guard auf die neue `model_form`. |
| `sourcecode/elements/quad4/element_quad4_nl_ai.m`, `quad4_nl_ai_energy.m` | Nur Kommentare — Kette, Signaturen und Verhalten unverändert. |
| `training/quad4/quad4_nl_ref.py` | Nur Modul-Doku. `k0_ref` bleibt als Referenz erhalten (Gate a prüft `k0_ref = K(z=0)`). |
| `examples/FEMSolid_ex_quad4_09_ai_nl_consistency.m` | Nur Kommentare — die Gates d/e prüfen unverändert. |

## Was gleich bleibt

* Kanonisierung, Ko-Rotation und die exakte Rücktransformations-Kette.
* Die **Subtraktionsform** `Ŵ = f(c̃,z̃) − f(c̃,0) − ∇f(c̃,0)ᵀz̃` — sie ist
  für die volle Energie genauso physikalisch korrekt und garantiert weiterhin
  strukturell `Ŵ(ĉ,0) = 0` und `F̂(ĉ,0) = 0` (Starrkörperbewegung →
  `Finte = 0` in Maschinengenauigkeit).
* Konsistenz `Ke = ∂Finte/∂Ue`, Symmetrie von `Ke`, Kräftegleichgewicht,
  Translationsnullraum — alles weiterhin strukturell exakt.
* Die komplette Gate-Leiter a–h.

## Was sich inhaltlich verschiebt

* **Die Tangente bei `z = 0` ist jetzt gelernt, nicht analytisch.** Beim
  K₀-Split war das Kleinamplituden-Regime (Newton-Endphase) exakt von `K̂₀`
  dominiert; jetzt muss das Netz den quadratischen Term dort mit hoher
  *relativer* Genauigkeit treffen. Gate e4 (`Ke(u→0)` vs. `element_quad4_lin`,
  Schwelle 1 %) ist damit vom Nebenkriterium zum zentralen Qualitätsmaß
  geworden.
* **Schwerere Zielfunktion → mehr Kapazität.** Default von h48 auf h64
  angehoben; falls das Go-Kriterium (mean < 2 %, P99 < 5 %) verfehlt wird,
  `QUAD4_SWEEP=1` laufen lassen.
* **Laufzeit-Trade-off:** pro Elementaufruf entfällt die K̂₀-Schleife, dafür
  ist das Netz größer (h64: ~9 900 MACs statt 5 424 plus K̂₀-Schleife). Der
  Netto-Effekt gehört im Benchmark (Gate h) nachgemessen.

## Nächste Schritte

1. Retraining: `python train_quad4_nl_W_network.py` (vorher optional
   `QUAD4_QUICK=1` als Pipeline-Test; für den vollen Datenmix zuerst
   `generate_newton_trajectories` in MATLAB). **Bis dahin wirft das Element
   beim Laden des alten Netzes absichtlich einen harten Fehler**
   (Metadaten-Mismatch `model_form`) — kein stiller Fallback.
2. Gates d/e in MATLAB: `FEMSolid_ex_quad4_09_ai_nl_consistency` — besonders
   e4 beobachten.
3. Benchmark `FEMSolid_ex_quad4_07_ai_nl_benchmark` (Iterationszahlen müssen
   identisch zum analytischen Element bleiben; dU/Speedup neu erheben).
4. Nach grünen Gates: Zahlen in `DLFE_quad4_Dokumentation.md` (Abschnitte
   3.0, 3.5, 3.8–3.9) und `README.md` nachziehen.
