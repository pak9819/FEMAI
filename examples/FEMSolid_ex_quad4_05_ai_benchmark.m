%FEMSOLID_EX_QUAD4_05_AI_BENCHMARK Zeit- und Genauigkeitsvergleich FEM vs. KI.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Vergleicht das klassische quad4-Element (Gauss-Integration) mit dem
%   Deep-Learned-Element (ganze Ke aus dem Netz) an 10 realistischen
%   Tragwerksplanungs-Beispielen (ebene Stahlbeton-Scheiben).
%
%   MESH-SKALIERUNG (wichtig fuer belastbare Zeiten):
%     Die Netze werden NICHT mit fester, kleiner Elementzahl gebaut, sondern
%     ueber eine ZIEL-Elementzahl je Struktur (NEL_TARGETS) hochskaliert --
%     gestaffelt von wenigen Tausend bis ~100k Elementen. Bei zu kleinen
%     Netzen (~400 Elemente) liegen die Assemblierungszeiten im ms-Bereich
%     und werden vom Messrauschen / der momentanen PC-Last dominiert. Grosse
%     Systeme heben die Zeiten deutlich ueber das Rauschen und machen den
%     Vergleich reproduzierbar. Die Staffelung (verschiedene Groessen) macht
%     ausserdem die "Kosten pro Element"-Skalierungskurve aussagekraeftig.
%     nx,ny werden je Struktur aus dem Seitenverhaeltnis berechnet, damit die
%     Elemente ~quadratisch bleiben (Trainingsbereich des Netzes).
%
%   Gemessen wird:
%     (A) ZEIT  -- der eigentliche Fokus:
%           - Assemblierungszeit (assemble.m) -> realer Pfad je Backend:
%             FEM = Element-Schleife mit analytischer Ke, KI = batched
%             Forward-Pass + vektorisierte Sparse-Matrix
%           - Gesamt-Loesungszeit (solve_FE) -> Einordnung end-to-end,
%             inkl. identischem linearen Loeser
%         Wiederholungen sind ADAPTIV: grosse Systeme (schon stabile Zeit)
%         werden seltener wiederholt als kleine -> beschraenkt die Laufzeit.
%     (B) GENAUIGKEIT:
%           - rel. L2-Fehler der Verschiebungen
%           - rel. L2-Fehler der von-Mises-Spannung
%           - mittlerer Element-Ke-Fehler (Frobenius) -- netznah,
%             unabhaengig vom Loeser; auf einer Stichprobe der Elemente
%             (bei grossen Netzen zu teuer, alle einzeln zu vergleichen)
%
%   Die 10 Strukturen werden zusaetzlich grafisch dargestellt (Geometrie,
%   Lagerung, Lasten) -- dafuer wird ein GROBES Netz verwendet (die feinen
%   Benchmark-Netze sind zum Plotten ungeeignet). Beschreibung siehe
%   FEMSolid_quad4_benchmark_structures.md.
%
%   FEST (Gueltigkeitsbereich des Netzes): Hooke, planeStrain, nu = 0.3.
%   E und Dicke d sind frei (Ke = E*d*Khat).
%
% PREREQUISITE
%   Run train_quad4_K_network.py first to generate quad4_K_network.mat.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-07-08
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

close all; clear; clc;

fprintf('=== quad4 Benchmark: Klassische FEM vs. Deep Learned Ke ===\n');
fprintf('    10 Tragwerksplanungs-Beispiele (Hooke, planeStrain, nu = 0.3)\n\n');

% ------------------------------------------------------------------------
% Globale (feste) Materialannahmen -- Stahlbeton-Scheibe
% ------------------------------------------------------------------------
E_MOD   = 3.0e7;     % E-Modul [kN/m^2]  (~ C30/37)
NU      = 0.3;       % Querkontraktion (fest ins Netz eintrainiert)
D_THICK = 0.30;      % Scheibendicke [m]
GAMMA   = 25.0;      % Wichte Stahlbeton [kN/m^3] (nur Eigengewichts-Fall)

MATNAME = 'Hooke';
MATCOND = 'planeStrain';

% ------------------------------------------------------------------------
% MESH-SKALIERUNG -- der zentrale Stellhebel dieses Benchmarks
%
%   NEL_TARGETS(k) = ungefaehre Ziel-Elementzahl der k-ten Struktur. Die
%   tatsaechliche Zahl weicht leicht ab (nx,ny werden ganzzahlig aus dem
%   Seitenverhaeltnis gerundet). Gestaffelt von wenigen Tausend bis ~100k,
%   damit (a) jede Messung deutlich ueber dem Rauschen liegt und (b) ein
%   breiter Groessenbereich fuer die Skalierungsanalyse abgedeckt wird.
%
%   Zum Verkleinern/Vergroessern einfach hier skalieren, z.B.
%     NEL_TARGETS = 0.25 * NEL_TARGETS;   % viertel so gross (schneller)
% ------------------------------------------------------------------------
NEL_TARGETS = [ 25000,  25000,   25000,  25000,  25000, ...
               25000, 25000,  25000,  25000, 25000 ];

% Grobes Netz nur fuer die Uebersichtsgrafik (feine Netze sind unplotbar).
PLOT_NEL = 200;

% ------------------------------------------------------------------------
% Adaptive Wiederholungen fuer stabile Zeitmessung (Median)
%
%   Grosse Systeme liefern schon aus einer Messung stabile Zeiten und sind
%   pro Messung teuer -> wenige Wiederholungen. Kleine Systeme sind billig
%   und rauschanfaellig -> mehr Wiederholungen. reps ~ WORK / NEL, geklemmt
%   auf [MIN, MAX]. So bleibt der Messaufwand je Struktur etwa konstant.
% ------------------------------------------------------------------------
WORK_ELEM      = 40000;  REPS_ELEM_MIN  = 4;  REPS_ELEM_MAX  = 20;  % Assemblierung
WORK_SOLVE     = 16000;  REPS_SOLVE_MIN = 2;  REPS_SOLVE_MAX = 8;   % Gesamt-Loesung

% Element-Ke-Fehler nur auf einer Stichprobe (alle 100k einzeln = zu teuer).
KE_SAMPLE = 2000;

% ------------------------------------------------------------------------
% Strukturen definieren (fein = Benchmark, grob = Plot)
% ------------------------------------------------------------------------
S     = define_benchmark_structures(E_MOD, NU, D_THICK, GAMMA, NEL_TARGETS);
Splot = define_benchmark_structures(E_MOD, NU, D_THICK, GAMMA, PLOT_NEL * ones(1, numel(S)));
nS    = numel(S);

% ------------------------------------------------------------------------
% (1) Strukturen plotten (Geometrie + Lagerung + Lasten) -- GROBES Netz
% ------------------------------------------------------------------------
figure('Name', 'Benchmark: 10 Tragwerksstrukturen', 'NumberTitle', 'off', ...
       'Position', [60 60 1500 620]);
for k = 1:nS
    subplot(2, 5, k);
    model_k = make_model(Splot(k), 'matlab');
    plot_structure(model_k, sprintf('%d) %s', k, Splot(k).name));
end
sgtitle('Benchmark-Strukturen (grob dargestellt): Geometrie, Lagerung (rot), Lasten (blau)');
drawnow;

% ------------------------------------------------------------------------
% (2) Benchmark-Schleife
% ------------------------------------------------------------------------
relU      = zeros(nS,1);   % rel. L2-Fehler Verschiebung [%]
relVM     = zeros(nS,1);   % rel. L2-Fehler von-Mises    [%]
keErr     = zeros(nS,1);   % mittl. Element-Ke-Fehler    [%]
tElemFEM  = zeros(nS,1);   % Element-Assemblierung FEM   [s]
tElemAI   = zeros(nS,1);   % Element-Assemblierung KI    [s]
tSolveFEM = zeros(nS,1);   % Gesamt-Loesung FEM          [s]
tSolveAI  = zeros(nS,1);   % Gesamt-Loesung KI           [s]
nelem     = zeros(nS,1);
ndof      = zeros(nS,1);

fprintf('Mesh-Skalierung: %d ... %d Elemente je Struktur (adaptive Wiederholungen)\n\n', ...
    min(NEL_TARGETS), max(NEL_TARGETS));
fprintf([' Nr | Struktur                         |    NEL  | t_el FEM | t_el KI  | Speedup | KeErr | dU   | dVM\n']);
fprintf([' ---|----------------------------------|---------|----------|----------|---------|-------|------|------\n']);

for k = 1:nS

    model_fem = make_model(S(k), 'matlab');
    model_ai  = make_model(S(k), 'ai');

    nelem(k) = model_fem.info.NEL;
    ndof(k)  = model_fem.info.NDOF;

    % --- Loesungen (Ausgabe unterdrueckt) ---
    [U_fem, res_fem] = solve_quiet(model_fem);
    [U_ai,  res_ai ] = solve_quiet(model_ai);

    % --- Genauigkeit ---
    relU(k)  = norm(U_ai - U_fem) / max(norm(U_fem), eps) * 100;
    vmF = res_fem.vonMises.node;  vmA = res_ai.vonMises.node;
    relVM(k) = norm(vmA - vmF) / max(norm(vmF), eps) * 100;
    keErr(k) = ke_frob_error(model_fem, model_ai, KE_SAMPLE);

    % --- Adaptive Wiederholungszahlen (je groesser NEL, desto weniger) ---
    Rel  = clamp_reps(WORK_ELEM  / nelem(k), REPS_ELEM_MIN,  REPS_ELEM_MAX);
    Rsol = clamp_reps(WORK_SOLVE / nelem(k), REPS_SOLVE_MIN, REPS_SOLVE_MAX);

    % --- Zeit: reine Element-Assemblierung ---
    tElemFEM(k) = time_assembly(model_fem, Rel);
    tElemAI(k)  = time_assembly(model_ai,  Rel);

    % --- Zeit: Gesamt-Loesung ---
    tSolveFEM(k) = time_solve(model_fem, Rsol);
    tSolveAI(k)  = time_solve(model_ai,  Rsol);

    sp = tElemFEM(k) / max(tElemAI(k), eps);
    fprintf(' %2d | %-32s | %7d | %6.2f ms | %6.2f ms | %6.2fx | %4.2f%% | %4.2f%%| %4.2f%%\n', ...
        k, S(k).name, nelem(k), 1e3*tElemFEM(k), 1e3*tElemAI(k), sp, ...
        keErr(k), relU(k), relVM(k));
end

% ------------------------------------------------------------------------
% Zusammenfassung
% ------------------------------------------------------------------------
spElem  = tElemFEM  ./ max(tElemAI,  eps);
spSolve = tSolveFEM ./ max(tSolveAI, eps);

fprintf('\n--- Zusammenfassung (%d Strukturen, %d ... %d Elemente) ---\n', ...
    nS, min(nelem), max(nelem));
fprintf('  Assemblierung (global): FEM %.1f ms | KI %.1f ms  -> Speedup Median %.2fx (Bereich %.2f..%.2f)\n', ...
    1e3*sum(tElemFEM), 1e3*sum(tElemAI), median(spElem), min(spElem), max(spElem));
fprintf('  Gesamt-Loesung        : FEM %.1f ms | KI %.1f ms  -> Speedup Median %.2fx\n', ...
    1e3*sum(tSolveFEM), 1e3*sum(tSolveAI), median(spSolve));
fprintf('  Genauigkeit           : dU  Median %.3f %% (max %.3f %%)\n', median(relU), max(relU));
fprintf('                          dVM Median %.3f %% (max %.3f %%)\n', median(relVM), max(relVM));
fprintf('                          Ke  Median %.3f %% (max %.3f %%)\n', median(keErr), max(keErr));
if median(spElem) > 1
    fprintf('  => KI-Element ist im Mittel SCHNELLER bei der Assemblierung (batched Pfad).\n');
else
    fprintf('  => KI-Element ist im Mittel NICHT schneller -- analytische Ke ist sehr billig.\n');
end

% ------------------------------------------------------------------------
% (3) Ergebnis-Grafiken
% ------------------------------------------------------------------------
figure('Name', 'Benchmark: Zeit & Genauigkeit', 'NumberTitle', 'off', ...
       'Position', [80 80 1200 760]);

% (a) Assemblierungszeit FEM vs KI
subplot(2,2,1);
bar(1e3*[tElemFEM, tElemAI]);
xlabel('Struktur Nr.'); ylabel('Zeit [ms]');
title('Assemblierungszeit (global, assemble.m)');
legend({'FEM (klassisch)', 'KI-Element'}, 'Location', 'northwest');
grid on;

% (b) Speedup je Struktur (Element-Ebene und Gesamt)
subplot(2,2,2);
bar([spElem, spSolve]);
hold on; yline_compat(1.0, 'gleich schnell');
xlabel('Struktur Nr.'); ylabel('Speedup  FEM/KI  [-]');
title('Speedup (>1 = KI schneller)');
legend({'Element-Assemblierung', 'Gesamt-Loesung'}, 'Location', 'best');
grid on;

% (c) Genauigkeit je Struktur
subplot(2,2,3);
bar([relU, relVM, keErr]);
xlabel('Struktur Nr.'); ylabel('rel. Fehler [%]');
title('Genauigkeit KI vs. FEM');
legend({'Verschiebung', 'von-Mises', 'Ke (Frobenius)'}, 'Location', 'northwest');
grid on;

% (d) Zeit pro Elementaufruf ueber Elementanzahl (Skalierung, log-x)
subplot(2,2,4);
scatter(nelem, 1e6*tElemFEM./nelem, 55, 'filled', 'MarkerFaceColor', [0.20 0.45 0.70]); hold on;
scatter(nelem, 1e6*tElemAI ./nelem, 55, 'd', 'MarkerEdgeColor', [0.85 0.40 0.20], 'LineWidth', 1.2);
set(gca, 'XScale', 'log');
xlabel('Anzahl Elemente (log)'); ylabel('Zeit pro Elementaufruf [\mus]');
title('Kosten pro Element ueber Systemgroesse');
legend({'FEM (klassisch)', 'KI-Element'}, 'Location', 'best');
grid on;

sgtitle('Deep Learned Ke (quad4): Zeit- und Genauigkeitsvergleich');

fprintf('\nFertig. Grafiken erzeugt.\n');


% ========================================================================
% Strukturdefinition: 10 Tragwerksplanungs-Beispiele
% ========================================================================
function S = define_benchmark_structures(E, nu, d, gamma, nelTargets)
%DEFINE_BENCHMARK_STRUCTURES Liefert 10 ebene Scheiben (Stahlbeton).
%   Jede Struktur: Rechteckvernetzung + realistische Lagerung/Lasten.
%   nelTargets(k) = Ziel-Elementzahl der k-ten Struktur; nx,ny werden aus dem
%   Seitenverhaeltnis so gewaehlt, dass die Elemente ~quadratisch bleiben.
%   Einheiten: m, kN, kN/m, kN/m^3.

matcard = [E, nu, d, 0];
S = struct('name', {}, 'coord', {}, 'elem', {}, 'mat', {}, ...
           'bcond', {}, 'fnode', {}, 'fvol', {});
k = 0;

% --- 1) Kragscheibe (Kragarm) mit Endlast ---------------------------------
k = k + 1;
Lx = 4.0; Ly = 1.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), -100/Ly);   % Endlast 100 kN
S(k) = pack('Kragscheibe (Endlast)', coord, elem, mat, ...
            bcond, [right, 2*ones(numel(right),1), vals], [0 0]);

% --- 2) Wandartiger Traeger / Einfeldtraeger (Gleichlast oben) ------------
k = k + 1;
Lx = 4.0; Ly = 2.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bl = nearest_node(coord, 0,  0);  br = nearest_node(coord, Lx, 0);
bcond = [bl 1 0; bl 2 0; br 2 0];                          % Festlager + Loslager
top   = edge_nodes(coord, 'top', Lx, Ly);
vals  = line_load_nodes(top, coord(top,1), -60);          % q = 60 kN/m
S(k) = pack('Einfeldtraeger (Gleichlast)', coord, elem, mat, ...
            bcond, [top, 2*ones(numel(top),1), vals], [0 0]);

% --- 3) Schubwand / aussteifende Wandscheibe (Horizontallast) ------------
k = k + 1;
Lx = 3.0; Ly = 6.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
top    = edge_nodes(coord, 'top', Lx, Ly);
vals   = line_load_nodes(top, coord(top,1), 80/Lx);       % H = 80 kN (Wind)
S(k) = pack('Schubwand (Horizontallast)', coord, elem, mat, ...
            bcond, [top, ones(numel(top),1), vals], [0 0]);

% --- 4) Konsole / Kragkonsole (Einzellast nahe Rand) ---------------------
k = k + 1;
Lx = 1.5; Ly = 1.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
left  = edge_nodes(coord, 'left', Lx, Ly);
bcond = fix_xy(left);
P     = nearest_node(coord, Lx, Ly);                       % freier oberer Eckknoten
S(k) = pack('Konsole (Einzellast)', coord, elem, mat, ...
            bcond, [P 2 -150], [0 0]);                     % 150 kN

% --- 5) Wandscheibe unter Eigengewicht -----------------------------------
k = k + 1;
Lx = 3.0; Ly = 4.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
S(k) = pack('Wandscheibe (Eigengewicht)', coord, elem, mat, ...
            bcond, [], [0 -gamma]);

% --- 6) Stuetze / Wandpfeiler, zentrische Drucklast ----------------------
k = k + 1;
Lx = 1.0; Ly = 5.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
top    = edge_nodes(coord, 'top', Lx, Ly);
vals   = line_load_nodes(top, coord(top,1), -200);        % zentrischer Druck
S(k) = pack('Stuetze (zentr. Druck)', coord, elem, mat, ...
            bcond, [top, 2*ones(numel(top),1), vals], [0 0]);

% --- 7) Exzentrisch belastete Stuetze (Druck + Biegung) ------------------
k = k + 1;
Lx = 1.5; Ly = 4.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
top    = edge_nodes(coord, 'top', Lx, Ly);
sel    = top(coord(top,1) >= Lx/2 - 1e-6);                % nur rechte Haelfte
vals   = line_load_nodes(sel, coord(sel,1), -200);
S(k) = pack('Exzentr. Stuetze (Druck+Biegung)', coord, elem, mat, ...
            bcond, [sel, 2*ones(numel(sel),1), vals], [0 0]);

% --- 8) Zweifeldtraeger / durchlaufender Wandtraeger ---------------------
k = k + 1;
Lx = 8.0; Ly = 2.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bl = nearest_node(coord, 0,    0);
bm = nearest_node(coord, Lx/2, 0);
br = nearest_node(coord, Lx,   0);
bcond = [bl 1 0; bl 2 0; bm 2 0; br 2 0];                  % drei Auflager
top   = edge_nodes(coord, 'top', Lx, Ly);
vals  = line_load_nodes(top, coord(top,1), -50);
S(k) = pack('Zweifeldtraeger (Gleichlast)', coord, elem, mat, ...
            bcond, [top, 2*ones(numel(top),1), vals], [0 0]);

% --- 9) Teilflaechenlast / Lasteinleitung (Auflagerpressung) -------------
k = k + 1;
Lx = 4.0; Ly = 2.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bl = nearest_node(coord, 0,  0);  br = nearest_node(coord, Lx, 0);
bcond = [bl 1 0; bl 2 0; br 2 0];
top   = edge_nodes(coord, 'top', Lx, Ly);
sel   = top(abs(coord(top,1) - Lx/2) <= Lx/8 + 1e-6);     % mittleres Viertel
vals  = line_load_nodes(sel, coord(sel,1), -300);
S(k) = pack('Teilflaechenlast (Lasteinleitung)', coord, elem, mat, ...
            bcond, [sel, 2*ones(numel(sel),1), vals], [0 0]);

% --- 10) Wandscheibe kombiniert (Auflast + Wind) -------------------------
k = k + 1;
Lx = 3.0; Ly = 5.0; [nx, ny] = mesh_for_target(Lx, Ly, nelTargets(k));
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
top    = edge_nodes(coord, 'top', Lx, Ly);
valsV  = line_load_nodes(top, coord(top,1), -80);         % Auflast (vertikal)
corner = nearest_node(coord, Lx, Ly);
fnode  = [top, 2*ones(numel(top),1), valsV; corner 1 60]; % + Wind (horizontal)
S(k) = pack('Wandscheibe (Auflast + Wind)', coord, elem, mat, ...
            bcond, fnode, [0 0]);

end

% ========================================================================
% Hilfsfunktionen: Strukturaufbau
% ========================================================================
function [nx, ny] = mesh_for_target(Lx, Ly, nelTarget)
%MESH_FOR_TARGET nx,ny fuer ~nelTarget Elemente bei ~quadratischen Elementen.
%   Aus  nx*ny ~ nelTarget  und  nx/ny ~ Lx/Ly  folgt
%       nx = sqrt(nelTarget * Lx/Ly),  ny = sqrt(nelTarget * Ly/Lx).
%   Ganzzahlig gerundet, mindestens 1. Die tatsaechliche Elementzahl nx*ny
%   weicht dadurch leicht vom Ziel ab (unkritisch fuer die Zeitmessung).
    nx = max(1, round(sqrt(nelTarget * Lx / Ly)));
    ny = max(1, round(sqrt(nelTarget * Ly / Lx)));
end

function R = clamp_reps(val, Rmin, Rmax)
%CLAMP_REPS Ganzzahlige Wiederholungszahl in [Rmin, Rmax].
    R = min(Rmax, max(Rmin, round(val)));
end

function [coord, elem, bcond, mat] = mesh_rect(Lx, Ly, nx, ny, matcard)
%MESH_RECT Nur Netz + Material (Lager/Lasten werden separat gesetzt).
    [coord, elem, bcond, mat] = ...
        create_model_data_rectangle(Lx, Ly, nx, ny, [0 0 0 0], [0 0 0 0], matcard);
end

function bcond = fix_xy(nodes)
%FIX_XY Sperrt x- und y-Verschiebung der angegebenen Knoten.
    nodes = nodes(:);
    bcond = [nodes, ones(numel(nodes),1),  zeros(numel(nodes),1);
             nodes, 2*ones(numel(nodes),1), zeros(numel(nodes),1)];
end

function ids = edge_nodes(coord, side, Lx, Ly)
%EDGE_NODES Knoten auf einer Rechteckkante (per Koordinate, robust).
    tol = 1e-6 * max(Lx, Ly);
    switch lower(side)
        case 'bottom', ids = find(abs(coord(:,2))      < tol);
        case 'top',    ids = find(abs(coord(:,2) - Ly) < tol);
        case 'left',   ids = find(abs(coord(:,1))      < tol);
        case 'right',  ids = find(abs(coord(:,1) - Lx) < tol);
        otherwise, error('Unbekannte Kante: %s', side);
    end
end

function id = nearest_node(coord, x, y)
%NEAREST_NODE Knoten am naechsten zu (x,y).
    [~, id] = min((coord(:,1)-x).^2 + (coord(:,2)-y).^2);
end

function vals = line_load_nodes(nodeIds, pos, q)
%LINE_LOAD_NODES Linienlast q [Kraft/Laenge] -> aequivalente Knotenlasten.
%   pos: Position jedes Knotens entlang der Kante. Einflusslaenge je Knoten
%   (Trapezregel); Summe der Lasten = q * Kantenlaenge.
    [ps, order] = sort(pos(:));
    n = numel(ps);
    trib = zeros(n,1);
    if n == 1
        trib(1) = 0;
    else
        trib(1)   = (ps(2) - ps(1)) / 2;
        trib(end) = (ps(end) - ps(end-1)) / 2;
        for i = 2:n-1
            trib(i) = (ps(i+1) - ps(i-1)) / 2;
        end
    end
    v = q * trib;
    vals = zeros(n,1);
    vals(order) = v;        % zurueck in urspruengliche Knotenreihenfolge
end

function s = pack(name, coord, elem, mat, bcond, fnode, fvol)
%PACK Buendelt eine Struktur in ein struct.
    s.name  = name;
    s.coord = coord;  s.elem = elem;  s.mat = mat;
    s.bcond = bcond;  s.fnode = fnode;  s.fvol = fvol;
end

% ========================================================================
% Hilfsfunktionen: Modell, Zeitmessung, Genauigkeit
% ========================================================================
function model = make_model(s, backend)
%MAKE_MODEL Lineares quad4-Modell mit gewaehltem Backend.
    setup = init_setup;
    setup.analysis.nl        = false;
    setup.element.type       = 'quad4';
    setup.element.backend    = backend;
    setup.material.name      = 'Hooke';
    setup.material.condition = 'planeStrain';
    setup.solver.verbose     = false;
    model = init_model(s.coord, s.elem, s.mat, s.bcond, s.fnode, s.fvol, [], setup);
end

function [U, res] = solve_quiet(model)
%SOLVE_QUIET Loest das Modell und unterdrueckt die Konsolenausgabe.
    txt = evalc(['[U, sr] = solve_FE(model); ', ...
                 'res = compute_model_results(model, sr(end));']); %#ok<NASGU>
end

function t = time_assembly(model, R)
%TIME_ASSEMBLY Median-Zeit fuer die GLOBALE Assemblierung (assemble.m).
%   Misst den realen Assemblierungspfad je Backend:
%     - FEM ('matlab'): Element-Schleife mit analytischer Ke
%     - KI  ('ai')    : batched Forward-Pass + vektorisierte Sparse-Matrix
%   Warm-up loest das einmalige Laden des Netzes (persistent) im KI-Element.
    assemble(model);                                 % Warm-up (Netz laden, JIT)
    ts = zeros(R,1);
    for r = 1:R
        t0 = tic;
        K = assemble(model); %#ok<NASGU>            % verhindert Wegoptimieren
        ts(r) = toc(t0);
    end
    t = median(ts);
end

function t = time_solve(model, R)
%TIME_SOLVE Median-Zeit fuer die Gesamt-Loesung (solve_FE, end-to-end).
    evalc('solve_FE(model);');                       % Warm-up + Ausgabe unterdrueckt
    ts = zeros(R,1);
    for r = 1:R
        t0 = tic;
        evalc('solve_FE(model);');
        ts(r) = toc(t0);
    end
    t = median(ts);
end

function err = ke_frob_error(model_fem, model_ai, nSample)
%KE_FROB_ERROR Mittlerer rel. Element-Ke-Fehler (Frobenius) in Prozent.
%   Auf einer zufaelligen Stichprobe von bis zu nSample Elementen -- bei
%   grossen Netzen ist der Vergleich aller Elemente einzeln (MATLAB-Schleife
%   mit je zwei Elementroutine-Aufrufen) zu teuer und ohnehin unnoetig.
    NEL    = model_fem.info.NEL;
    ndofEl = model_fem.element.ndofElement;
    rf = model_fem.element.routine;  ra = model_ai.element.routine;
    gp = model_fem.element.gp;  w = model_fem.element.w;  opts = model_fem.element.opts;
    MN = model_fem.material.name;  MC = model_fem.material.condition;
    Ue = zeros(ndofEl, 1);

    if NEL > nSample
        idx = randperm(NEL, nSample);
    else
        idx = 1:NEL;
    end
    errs = zeros(numel(idx), 1);
    for i = 1:numel(idx)
        e  = idx(i);
        ce = model_fem.coord(model_fem.elem(e,:),:);
        me = model_fem.mat(e,:);
        be = model_fem.fvol(e,:)';
        Kf = rf(ce, me, be, 0, Ue, [], gp, w, MN, MC, opts);
        Ka = ra(ce, me, be, 0, Ue, [], gp, w, MN, MC, opts);
        errs(i) = norm(Ka - Kf, 'fro') / max(norm(Kf, 'fro'), eps);
    end
    err = mean(errs) * 100;
end

% ========================================================================
% Hilfsfunktionen: Plot
% ========================================================================
function plot_structure(model, ttl)
%PLOT_STRUCTURE Zeichnet Netz, Lagerung (rot) und Lasten (blau).
    plot_results(model, 'undeformed');
    plot_results(model, 'support', 1.0);
    draw_loads(model);
    title(ttl, 'FontSize', 9, 'Interpreter', 'none');
    axis equal; axis tight;
end

function draw_loads(model)
%DRAW_LOADS Lastpfeile aus fnode (Knotenlasten) und fvol (Eigengewicht).
    coord = model.coord;
    span  = max(max(coord,[],1) - min(coord,[],1));
    Larr  = 0.18 * span;

    % --- Knotenlasten ---
    if ~isempty(model.fnode)
        F = zeros(model.info.NNODE, 2);
        for i = 1:size(model.fnode,1)
            F(model.fnode(i,1), model.fnode(i,2)) = ...
                F(model.fnode(i,1), model.fnode(i,2)) + model.fnode(i,3);
        end
        mag = sqrt(sum(F.^2, 2));
        mmax = max(mag);
        if mmax > 0
            sc = Larr / mmax;
            nz = find(mag > 0);
            quiver(coord(nz,1) - sc*F(nz,1), coord(nz,2) - sc*F(nz,2), ...
                   sc*F(nz,1), sc*F(nz,2), 0, ...
                   'Color', [0 0.2 0.8], 'LineWidth', 1.0, 'MaxHeadSize', 0.4);
        end
    end

    % --- Eigengewicht (Volumenlast) als Pfeilraster ---
    if any(model.fvol(:) ~= 0)
        xr = linspace(min(coord(:,1)), max(coord(:,1)), 4);
        yr = linspace(min(coord(:,2)), max(coord(:,2)), 4);
        [Xg, Yg] = meshgrid(xr(2:end-1), yr(2:end-1));
        quiver(Xg(:), Yg(:), zeros(numel(Xg),1), -0.5*Larr*ones(numel(Xg),1), 0, ...
               'Color', [0 0.55 0.2], 'LineWidth', 0.8, 'MaxHeadSize', 0.5);
    end
end

function yline_compat(yval, label)
%YLINE_COMPAT Horizontale Referenzlinie (auch ohne yline-Funktion).
    xl = xlim; hold on;
    plot(xl, [yval yval], 'k--', 'LineWidth', 1.2);
    text(xl(1), yval, ['  ' label], 'VerticalAlignment', 'bottom');
end
