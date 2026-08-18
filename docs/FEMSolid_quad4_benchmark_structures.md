# Benchmark-Strukturen – quad4 (FEM vs. Deep Learned Ke)

Die 10 Beispiele in `FEMSolid_ex_quad4_05_ai_benchmark.m` sind ebene
**Stahlbeton-Scheiben** aus der Tragwerksplanung. Sie decken unterschiedliche
Geometrien (Seitenverhältnisse), Lagerungen und Lastfälle ab und bleiben dabei
im **Gültigkeitsbereich des Netzes**.

## Feste Annahmen (Netz-Gültigkeit)

| Größe | Wert | Bemerkung |
|-------|------|-----------|
| Materialgesetz | Hooke (linear elastisch) | fest |
| Spannungszustand | `planeStrain` | fest ins Netz eintrainiert |
| Querkontraktion ν | 0.3 | fest ins Netz eintrainiert |
| E-Modul E | 3.0·10⁷ kN/m² (≈ C30/37) | frei (Ke = E·d·K̂) |
| Scheibendicke d | 0.30 m | frei (Ke = E·d·K̂) |
| Wichte γ | 25 kN/m³ | nur Eigengewichts-Fall |

**Einheiten:** m, kN, kN/m, kN/m³.
Die Vernetzung ist jeweils so gewählt, dass die Elemente näherungsweise
**quadratisch** sind (Seitenverhältnis ≈ 1) – also im Trainingsbereich des Netzes,
in dem es am genauesten ist.

## Mesh-Skalierung (Elementzahl)

Die Netze werden **nicht** mit fester, kleiner Elementzahl gebaut, sondern über
eine **Ziel-Elementzahl je Struktur** (`NEL_TARGETS` im Benchmark-Skript)
hochskaliert – gestaffelt von wenigen Tausend bis ~100 000 Elementen. `nx` und
`ny` werden dabei aus dem Seitenverhältnis `Lx/Ly` berechnet
(`nx ≈ √(NEL·Lx/Ly)`, `ny ≈ √(NEL·Ly/Lx)`), damit die Elemente ~quadratisch
bleiben.

**Warum:** Bei ~400 Elementen liegen die Assemblierungszeiten im Millisekunden-
Bereich und werden vom Messrauschen bzw. der momentanen PC-Last dominiert. Große
Systeme heben die Zeiten deutlich über das Rauschen (reproduzierbar), die
Staffelung deckt zusätzlich einen breiten Größenbereich für die
Skalierungsanalyse ab. Die Elementzahl ist damit ein Stellhebel und keine feste
Eigenschaft der Struktur mehr – die Tabelle unten nennt daher **Seitenverhältnis
und Beispiel-Netz** (bei kleinem Ziel), nicht mehr eine feste Auflösung.

Da Lagerung und Lasten **koordinatenbasiert** gesetzt werden (Kantenknoten,
nächster Knoten) und Linienlasten über die Trapezregel in Knotenlasten umgerechnet
werden (Gesamtlast = q · Kantenlänge), bleibt die Physik bei jeder Auflösung
identisch – die Verfeinerung ist last- und lagerungs-neutral.

Die Übersichtsgrafik der 10 Strukturen wird bewusst mit einem **groben** Netz
(`PLOT_NEL`) gezeichnet; die feinen Benchmark-Netze wären zum Plotten ungeeignet.

> Hinweis: Physikalisch sind dünne Scheiben eigentlich ein `planeStress`-Problem.
> Da das aktuelle Netz auf `planeStrain` trainiert wurde, wird hier konsistent
> `planeStrain` verwendet – der Vergleich FEM vs. KI bleibt damit fair
> (identischer Zustand auf beiden Seiten).

---

## Die 10 Strukturen

Die Spalte **Ziel-NEL** zeigt die Default-Elementzahl aus `NEL_TARGETS`; die
tatsächliche Zahl weicht durch die ganzzahlige `nx,ny`-Rundung leicht ab.

| Nr | Bezeichnung | L × H [m] | Ziel-NEL (default) | Lagerung | Last | Tragverhalten |
|----|-------------|-----------|--------------------|----------|------|---------------|
| 1 | Kragscheibe (Endlast) | 4.0 × 1.0 | 2 000 | linke Kante voll eingespannt | Endlast 100 kN vertikal (rechte Kante) | Biegung + Schub, Kragarm |
| 2 | Einfeldträger (Gleichlast) | 4.0 × 2.0 | 3 500 | Festlager links unten, Loslager rechts unten | Gleichlast q = 60 kN/m (Oberkante) | wandartiger Träger, Biegung |
| 3 | Schubwand (Horizontallast) | 3.0 × 6.0 | 6 000 | Unterkante voll eingespannt | H = 80 kN horizontal (Oberkante) | aussteifende Wand, Schub + Biegung |
| 4 | Konsole (Einzellast) | 1.5 × 1.0 | 10 000 | linke Kante voll eingespannt | Einzellast 150 kN vertikal (freie obere Ecke) | kurze Konsole, schubdominiert |
| 5 | Wandscheibe (Eigengewicht) | 3.0 × 4.0 | 16 000 | Unterkante voll eingespannt | Eigengewicht γ = 25 kN/m³ | Volumenlast, Normalkraft |
| 6 | Stütze (zentr. Druck) | 1.0 × 5.0 | 25 000 | Unterkante voll eingespannt | zentrische Drucklast 200 kN/m (Oberkante) | schlanke Stütze, Normalkraft |
| 7 | Exzentr. Stütze (Druck + Biegung) | 1.5 × 4.0 | 40 000 | Unterkante voll eingespannt | Drucklast 200 kN/m nur rechte Hälfte | Normalkraft + Biegung |
| 8 | Zweifeldträger (Gleichlast) | 8.0 × 2.0 | 60 000 | 3 Auflager unten (Mitte + Enden) | Gleichlast q = 50 kN/m (Oberkante) | Durchlaufträger, Stützmoment |
| 9 | Teilflächenlast (Lasteinleitung) | 4.0 × 2.0 | 80 000 | Festlager + Loslager unten | Teillast 300 kN/m im mittleren Viertel (oben) | lokale Lasteinleitung, Spaltzug |
| 10 | Wandscheibe (Auflast + Wind) | 3.0 × 5.0 | 100 000 | Unterkante voll eingespannt | Auflast 80 kN/m vertikal + Wind 60 kN horizontal (obere Ecke) | kombinierte Beanspruchung |

---

## Lagerungs- und Lastkonventionen

- **Lagerung** (`bcond = [Knoten, Richtung, Wert]`, Richtung 1 = x, 2 = y):
  „voll eingespannt“ = beide Richtungen gesperrt; „Festlager“ = x + y;
  „Loslager“ = nur y. Im Plot rote Marker (`<` = x gesperrt, `v` = y gesperrt).
- **Linienlasten** werden über die Trapezregel in äquivalente Knotenlasten
  umgerechnet (Summe = q · Kantenlänge).
- **Lasten** im Plot als blaue Pfeile (Knotenlasten), Eigengewicht als grünes
  Pfeilraster.

---

## Auswertung im Benchmark

Pro Struktur werden gemessen und geplottet:

1. **Element-Assemblierungszeit** (nur Ke, ohne linearen Löser) – isoliert die
   Element-/Netzkosten, also genau das, was optimiert wurde.
2. **Gesamt-Lösungszeit** (`solve_FE`, end-to-end) – zur Einordnung; enthält den
   identischen linearen Löser und verwässert daher den Element-Speedup.
3. **Genauigkeit**: rel. L2-Fehler der Verschiebung, der von-Mises-Spannung und
   mittlerer Element-Ke-Fehler (Frobenius). Der Ke-Fehler wird auf einer
   **Stichprobe** von bis zu `KE_SAMPLE` Elementen gemessen – bei ~100 000
   Elementen ist der Einzelvergleich aller Elemente zu teuer und unnötig.

Alle Zeiten als **Median** über mehrere Wiederholungen (mit Warm-up, damit das
einmalige Laden des Netzes nicht mitgemessen wird). Die Wiederholungszahl ist
**adaptiv**: große, ohnehin teure Systeme werden seltener wiederholt als kleine,
rauschanfällige (`reps ≈ WORK/NEL`, geklemmt) – so bleibt der Messaufwand je
Struktur etwa konstant und die Gesamtlaufzeit beschränkt.
