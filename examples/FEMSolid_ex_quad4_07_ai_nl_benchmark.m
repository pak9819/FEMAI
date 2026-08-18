%FEMSOLID_EX_QUAD4_07_AI_NL_BENCHMARK Nichtlinearer FEM- vs. KI-Vergleich.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Vergleicht das klassische nichtlineare quad4-Element (element_quad4_nl,
%   Total Lagrange) mit dem Deep-Learned-Element (element_quad4_nl_ai), das
%   ein ENERGIEPOTENTIAL lernt und Finte sowie Ke daraus durch Differentiation
%   gewinnt (Residual-Energie mit analytischem K0-Split).
%
%   Ausser der Volumenlast bleibt keine Mechanik analytisch. Vier
%   Eigenschaften stehen im Fokus:
%
%     (K) KONVERGENZ: Ke ist per Konstruktion exakt die Jacobimatrix von
%         Finte (gemeinsames Potential) -> Newton konvergiert wieder mit der
%         Rate des analytischen Elements. Erwartung: gleiche Iterationszahl
%         (+/- 0..3). Deutlich mehr Iterationen waeren ein Alarmsignal.
%
%     (G) GENAUIGKEIT der konvergierten Loesung -- der gelernte Finte-Fehler
%         geht DIREKT ins Gleichgewicht ein:
%         - rel. L2-Fehler der Verschiebungen (dU) am Endlastzustand
%         - rel. L2-Fehler der von-Mises-Spannung (dVM)
%         - Element-Fehler von Finte UND Ke an einer Stichprobe, ausgewertet
%           am DEFORMIERTEN Zustand -- mit MITTELWERT UND P99, denn Newton
%           wird vom schlechtesten Element limitiert, nicht vom mittleren.
%
%     (Z) ZEIT auf ZWEI Ebenen (Gate h): je Assemblierung UND Gesamtloesung
%         (Assemblierungen x Iterationen). Die Gesamtzeit ist die ehrliche
%         Metrik fuer das Forschungsziel; dazu ein Kostenmodell (MACs vs.
%         FLOPs) zur Extrapolation auf teure Elemente.
%
%     (T) TOLERANZ: modell-ehrliches, RELATIVES Konvergenzkriterium
%         (TOL_REL * ||Fext||) statt absolut 1e-8 -- identisch fuer beide
%         Backends. newton.m bleibt unangetastet.
%
%   Material: St.-Venant-Kirchhoff, planeStrain, nu = 0.3 (fest ins Netz
%   eintrainiert, wird beim Laden HART geprueft). E und Dicke d sind frei.
%   ACHTUNG Gueltigkeit: das Netz kennt nur die trainierte ZUSTANDS-Huelle
%   (||E_green|| <= E_MAX); die Zustandsrotation ist dank Ko-Rotation
%   unbeschraenkt. Die Benchmark-Lasten muessen in der Dehnungs-Huelle bleiben.
%
% PREREQUISITE
%   Run training/quad4/train_quad4_nl_W_network.py first to generate
%   quad4_nl_W_network.mat.
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

% "clear all" statt "clear": das KI-Element haelt das geladene Netz in
% PERSISTENT-Variablen. Ein einfaches "clear" loescht die nicht -- eine
% Sitzung, die frueher schon einmal assembliert hat, wuerde mit VERALTETEN
% Gewichten weiterrechnen, auch wenn das .mat inzwischen neu deployt wurde.
close all; clear all; clc; %#ok<CLALL>

fprintf('=== quad4 NL-Benchmark: Klassische FEM vs. Deep Learned Energie ===\n');
fprintf('    St.-Venant-Kirchhoff, planeStrain, nu = 0.3, Total Lagrange\n\n');

% ------------------------------------------------------------------------
% Feste Annahmen
% ------------------------------------------------------------------------
E_MOD   = 1.0e3;       % E-Modul (moderat -> gut konvergierende geom. Nichtlin.)
NU      = 0.3;
D_THICK = 1.0;

MATNAME = 'StVenant';
MATCOND = 'planeStrain';

NUM_STEPS = 6;         % Lastschritte (Lastinkrementierung)
MAX_ITER  = 150;        % max. Newton-Iterationen je Lastschritt

% Modell-ehrliches, RELATIVES Konvergenzkriterium (Plan Gate f):
%   tolR = TOL_REL * ||Fext(frei)||
% Ein gelerntes Residuum kann das absolute 1e-8 aus init_setup prinzipiell
% nicht erreichen -- unterhalb der Modellgenauigkeit zu iterieren bringt
% physikalisch nichts. Beide Backends bekommen denselben Wert.
TOL_REL   = 1e-6;

% ACHTUNG LASTHOEHE: Die Lasten sind bewusst KLEIN gewaehlt, damit die
% Verschiebungszustaende in der Trainings-Huelle bleiben (||E_green|| <= ~0.2,
% Rotation <= ~45 Grad). Zu grosse Lasten (z.B. Kragarm-Spitzenauslenkung in
% Groessenordnung der Laenge) fuehren aus der Huelle -> das Netz extrapoliert,
% Ke/Finte werden unbrauchbar und Newton divergiert. Erst mit
% FEMSolid_ex_quad4_08_ai_nl_check.m die Element-Genauigkeit in der Huelle
% pruefen, dann hier die Lasten so hoch wie moeglich (aber in der Huelle) setzen.

% Netzfeinheit: nl ist pro Loesung teuer (Newton x Lastschritte) -> moderat.
% MESH_SCALE skaliert alle Netze gemeinsam (1 = klein/schnell).
MESH_SCALE = 2;

% Zeitmessung Assemblierung (am deformierten Zustand)
R_ELEM    = 12;

% Element-Fehler (Finte/Ke) auf einer Stichprobe
KE_SAMPLE = 1000;

% ------------------------------------------------------------------------
% Strukturen definieren (geometrisch nichtlineare Szenarien)
% ------------------------------------------------------------------------
S  = define_nl_structures(E_MOD, NU, D_THICK, MESH_SCALE);
nS = numel(S);

% ------------------------------------------------------------------------
% (1) Strukturen plotten (unverformt + Lagerung/Last)
% ------------------------------------------------------------------------
figure('Name', 'NL-Benchmark: Strukturen', 'NumberTitle', 'off', ...
       'Position', [60 60 1400 500]);
for k = 1:nS
    subplot(1, nS, k);
    model_k = make_nl_model(S(k), 'matlab', NUM_STEPS, MAX_ITER);
    plot_structure(model_k, sprintf('%d) %s', k, S(k).name));
end
sgtitle('NL-Benchmark: Ausgangsgeometrie, Lagerung (rot), Last (blau)');
drawnow;

% ------------------------------------------------------------------------
% (2) Benchmark-Schleife
% ------------------------------------------------------------------------
relU     = zeros(nS,1);   % rel. L2-Fehler Verschiebung [%]
relVM    = zeros(nS,1);   % rel. L2-Fehler von-Mises    [%]
fintErr  = zeros(nS,1);   % mittl. Element-Finte-Fehler [%] (alle Element-DOFs)
fintFree = zeros(nS,1);   % glob. Finte-Fehler NUR freie DOFs [%]
keErrN   = zeros(nS,4);   % mittl. Element-Ke-Fehler [%] in [Frob, Spektral-2, 1-Norm, MaxAbs]
itFEM    = zeros(nS,1);   % Newton-Iterationen gesamt (FEM)
itAI     = zeros(nS,1);   % Newton-Iterationen gesamt (KI)
tElemFEM = zeros(nS,1);   % Assemblierung deformiert   [s]
tElemAI  = zeros(nS,1);   % Assemblierung deformiert   [s]
tSolFEM  = zeros(nS,1);   % Gesamt-Loesungszeit        [s]  (Gate h1)
tSolAI   = zeros(nS,1);   % Gesamt-Loesungszeit        [s]  (Gate h1)
fintP99  = zeros(nS,1);   % P99 des Element-Finte-Fehlers [%]
keP99    = zeros(nS,1);   % P99 des Element-Ke-Fehlers    [%]
nelem    = zeros(nS,1);
okFEM    = false(nS,1);
okAI     = false(nS,1);

fprintf('Loese (NUM_STEPS = %d, MAX_ITER = %d) ...\n\n', NUM_STEPS, MAX_ITER);
fprintf([' Nr | Struktur                     |  NEL  | itFEM | itKI | Fint(el) | Fint(frei) | Ke(Frob) | dU    | dVM\n']);
fprintf([' ---|------------------------------|-------|-------|------|----------|------------|----------|-------|------\n']);

for k = 1:nS

    % Modell-ehrliches, RELATIVES Konvergenzkriterium -- identisch fuer beide
    % Backends (Gate f). newton.m bleibt unangetastet.
    tolR_k = reference_tolR(S(k), NUM_STEPS, MAX_ITER, TOL_REL);

    model_fem = make_nl_model(S(k), 'matlab', NUM_STEPS, MAX_ITER, tolR_k);
    model_ai  = make_nl_model(S(k), 'ai',     NUM_STEPS, MAX_ITER, tolR_k);

    nelem(k) = model_fem.info.NEL;

    % --- Nichtlineare Loesungen (mit Gesamtzeit fuer Gate h1) ---
    t0 = tic;  [U_fem, res_fem, itFEM(k), okFEM(k)] = solve_nl_quiet(model_fem);
    tSolFEM(k) = toc(t0);
    t0 = tic;  [U_ai,  res_ai,  itAI(k),  okAI(k) ] = solve_nl_quiet(model_ai);
    tSolAI(k)  = toc(t0);

    % --- Genauigkeit (konvergierte Loesung) ---
    relU(k)  = norm(U_ai - U_fem) / max(norm(U_fem), eps) * 100;
    vmF = res_fem.vonMises.node;  vmA = res_ai.vonMises.node;
    relVM(k) = norm(vmA - vmF) / max(norm(vmF), eps) * 100;

    % --- Element-Fehler Finte/Ke am DEFORMIERTEN Zustand (FEM-Loesung) ---
    %     fintErr: Element-Finte-Fehler ueber ALLE Element-DOFs (netznah).
    %     keErrN : Ke-Fehler in mehreren Normen (Frob, Spektral, 1-Norm, MaxAbs).
    [fintErr(k), keErrN(k,:), fintP99(k), keP99(k)] = ...
        elem_nl_error(model_fem, model_ai, U_fem, KE_SAMPLE);

    % --- Globaler Finte-Fehler NUR auf den freien Freiheitsgraden ---------
    %     Newton loest das Gleichgewicht Fint = Fext ausschliesslich auf den
    %     freien DOFs (newton.m); an den gebundenen DOFs steht die (grosse)
    %     Auflagerreaktion, die die Norm sonst verfaelscht. Daher: nur freie.
    fintFree(k) = global_fint_error_free(model_fem, model_ai, U_fem);

    % --- Zeit: Element-Assemblierung am deformierten Zustand ---
    tElemFEM(k) = time_assembly(model_fem, U_fem, R_ELEM);
    tElemAI(k)  = time_assembly(model_ai,  U_fem, R_ELEM);

    fprintf(' %2d | %-28s | %5d | %5d | %4d | %7.2f%% | %9.2f%% | %7.2f%% | %5.2f%%| %4.2f%%\n', ...
        k, S(k).name, nelem(k), itFEM(k), itAI(k), fintErr(k), fintFree(k), keErrN(k,1), relU(k), relVM(k));
end

% ------------------------------------------------------------------------
% Zusatztabelle: Ke-Fehler in verschiedenen Matrixnormen
%   Hinweis: Ke ist symmetrisch -> 1-Norm == inf-Norm (identisch), daher nur
%   die 1-Norm gezeigt. Spektralnorm (groesster Singulaerwert) beschraenkt
%   den relativen Fehler in JEDER Verformungsmode und ist physikalisch am
%   aussagekraeftigsten; MaxAbs zeigt den groessten Einzeleintrags-Fehler.
% ------------------------------------------------------------------------
fprintf('\n--- Ke-Fehler je Struktur in verschiedenen Normen [%%] ---\n');
fprintf([' Nr | Struktur                     | Frobenius | Spektral-2 |  1-Norm  |  MaxAbs\n']);
fprintf([' ---|------------------------------|-----------|------------|----------|---------\n']);
for k = 1:nS
    fprintf(' %2d | %-28s | %8.2f  | %9.2f  | %7.2f  | %7.2f\n', ...
        k, S(k).name, keErrN(k,1), keErrN(k,2), keErrN(k,3), keErrN(k,4));
end

% ------------------------------------------------------------------------
% Zusammenfassung
% ------------------------------------------------------------------------
spElem = tElemFEM ./ max(tElemAI, eps);

fprintf('\n--- Zusammenfassung (%d NL-Strukturen) ---\n', nS);
if ~all(okFEM) || ~all(okAI)
    fprintf('  WARNUNG: nicht alle Loesungen konvergiert (FEM ok: %d/%d, KI ok: %d/%d)\n', ...
        sum(okFEM), nS, sum(okAI), nS);
end
fprintf('  Newton-Iterationen: FEM gesamt %d | KI gesamt %d  -> Differenz %d\n', ...
    sum(itFEM), sum(itAI), sum(itAI) - sum(itFEM));
fprintf('  Assemblierung (deformiert): FEM %.2f ms | KI %.2f ms  -> Speedup Median %.2fx\n', ...
    1e3*sum(tElemFEM), 1e3*sum(tElemAI), median(spElem));
fprintf('  Genauigkeit: dU           Median %.3f %% (max %.3f %%)\n', median(relU), max(relU));
fprintf('               dVM          Median %.3f %% (max %.3f %%)\n', median(relVM), max(relVM));
fprintf('               Finte (el)   Median %.3f %% (max %.3f %%)\n', median(fintErr), max(fintErr));
fprintf('               Finte (frei) Median %.3f %% (max %.3f %%)\n', median(fintFree), max(fintFree));
fprintf('               Ke Frobenius Median %.3f %% (max %.3f %%)\n', median(keErrN(:,1)), max(keErrN(:,1)));
fprintf('               Ke Spektral  Median %.3f %% (max %.3f %%)\n', median(keErrN(:,2)), max(keErrN(:,2)));
fprintf('               Ke 1-Norm    Median %.3f %% (max %.3f %%)\n', median(keErrN(:,3)), max(keErrN(:,3)));
fprintf('               Ke MaxAbs    Median %.3f %% (max %.3f %%)\n', median(keErrN(:,4)), max(keErrN(:,4)));
if all(okAI) && sum(itAI) <= sum(itFEM) + 3*nS
    fprintf('  => Newton konvergiert robust (Konsistenz greift: hoechstens wenige Zusatziterationen).\n');
elseif ~all(okAI)
    fprintf('  => WARNUNG: KI-Loesung nicht ueberall konvergiert -- Zustands-Huelle/Lasten pruefen.\n');
else
    fprintf('  => KI braucht deutlich mehr Newton-Iterationen -- Ke/Finte pruefen.\n');
end

% ------------------------------------------------------------------------
% Gate f: Genauigkeit inkl. Perzentilen
% ------------------------------------------------------------------------
fprintf('\n--- Gate f: Genauigkeit (mean UND P99 je Struktur) ---\n');
fprintf(' Nr | Struktur                     | Fint mean |  Fint P99 |  Ke mean |   Ke P99\n');
fprintf(' ---|------------------------------|-----------|-----------|----------|---------\n');
for k = 1:nS
    fprintf(' %2d | %-28s | %8.2f%% | %8.2f%% | %7.2f%% | %7.2f%%\n', ...
        k, S(k).name, fintErr(k), fintP99(k), keErrN(k,1), keP99(k));
end
gateF = all(okAI) && max(fintErr) < 2 && max(keErrN(:,1)) < 2 && ...
        max(fintP99) < 5 && max(keP99) < 5 && max(relU) < 0.5 && ...
        sum(itAI) <= sum(itFEM) + 3*nS;
fprintf('  Gate f (mean < 2 %%, P99 < 5 %%, dU < 0.5 %%, Iter <= FEM+3): %s\n', ...
    gate_verdict(gateF));

% ------------------------------------------------------------------------
% Gate h: SPEEDUP -- das eigentliche Forschungsziel
%   h1 (hart)     : Gesamt-Loesungszeit AI < FEM auf >= 4 von 5 Strukturen.
%                   Der Hebel ist die Iterationszahl (Konsistenz).
%   h2 (berichtet): Zeit je Assemblierung + Kostenmodell zur Extrapolation
%                   auf teure Elemente (3D, viele GP, komplexe Materialien).
%                   quad4 ist analytisch billig -- h2 darf knapp ausgehen.
% ------------------------------------------------------------------------
spSol = tSolFEM ./ max(tSolAI, eps);
nWin  = sum(tSolAI < tSolFEM);
fprintf('\n--- Gate h: Speedup (Forschungsziel) ---\n');
fprintf(' Nr | Struktur                     | t_asm FEM | t_asm KI | Speedup | t_ges FEM | t_ges KI | Speedup\n');
fprintf(' ---|------------------------------|-----------|----------|---------|-----------|----------|--------\n');
for k = 1:nS
    fprintf(' %2d | %-28s | %8.2fms | %7.2fms | %6.2fx | %8.2fs | %7.2fs | %6.2fx\n', ...
        k, S(k).name, 1e3*tElemFEM(k), 1e3*tElemAI(k), spElem(k), ...
        tSolFEM(k), tSolAI(k), spSol(k));
end
fprintf('  h1 (hart)     : Gesamtzeit KI schneller auf %d von %d Strukturen -> %s\n', ...
    nWin, nS, gate_verdict(nWin >= max(1, nS - 1)));
fprintf('  h2 (berichtet): Assemblierung Speedup Median %.2fx\n', median(spElem));
print_cost_model();
fprintf('  Hinweis: dU/dVM messen die Finte-Netzqualitaet; die Konsistenz\n');
fprintf('           (Ke = dFinte/dUe) ist strukturell garantiert und wird in\n');
fprintf('           FEMSolid_ex_quad4_09_ai_nl_consistency.m per FD geprueft.\n');

% ------------------------------------------------------------------------
% (3) Ergebnis-Grafiken
% ------------------------------------------------------------------------
figure('Name', 'NL-Benchmark: Konvergenz, Genauigkeit, Zeit', 'NumberTitle', 'off', ...
       'Position', [60 60 1500 760]);

% (a) Newton-Iterationen FEM vs KI
subplot(2,3,1);
bar([itFEM, itAI]);
xlabel('Struktur Nr.'); ylabel('Newton-Iter. (gesamt)');
title('Newton-Iterationen (Konsistenz-Check)');
legend({'FEM (analytisch)', 'KI-Element'}, 'Location', 'northwest');
grid on;

% (b) Genauigkeit dU / dVM
subplot(2,3,2);
bar([relU, relVM]);
xlabel('Struktur Nr.'); ylabel('rel. Fehler [%]');
title('Genauigkeit der konvergierten Loesung');
legend({'Verschiebung dU', 'von-Mises dVM'}, 'Location', 'northwest');
grid on;

% (c) Finte-Fehler: alle Element-DOFs vs. nur freie DOFs (global)
subplot(2,3,3);
bar([fintErr, fintFree]);
xlabel('Struktur Nr.'); ylabel('rel. Fehler [%]');
title('Finte-Fehler: alle DOFs vs. nur freie DOFs');
legend({'Element (alle DOFs)', 'global (freie DOFs)'}, 'Location', 'northwest');
grid on;

% (d) Ke-Fehler in verschiedenen Normen (deformiert)
subplot(2,3,4);
bar(keErrN);
xlabel('Struktur Nr.'); ylabel('rel. Fehler [%]');
title('Ke-Fehler in verschiedenen Normen');
legend({'Frobenius', 'Spektral-2', '1-Norm', 'MaxAbs'}, 'Location', 'northwest');
grid on;

% (e) Assemblierungszeit FEM vs KI
subplot(2,3,5);
bar(1e3*[tElemFEM, tElemAI]);
xlabel('Struktur Nr.'); ylabel('Zeit [ms]');
title('Assemblierungszeit (deformiert, assemble.m)');
legend({'FEM (klassisch)', 'KI-Element'}, 'Location', 'northwest');
grid on;

sgtitle('Deep Learned Energie (quad4, nichtlinear): FEM vs. KI');

fprintf('\nFertig. Grafiken erzeugt.\n');


% ========================================================================
% Strukturdefinition: geometrisch nichtlineare Szenarien
% ========================================================================
function S = define_nl_structures(E, nu, d, scale)
%DEFINE_NL_STRUCTURES Liefert geometrisch nichtlineare quad4-Beispiele.
%   Moderate Lasten mit Lastinkrementierung, damit Newton robust konvergiert.

matcard = [E, nu, d, 0];
S = struct('name', {}, 'coord', {}, 'elem', {}, 'mat', {}, ...
           'bcond', {}, 'fnode', {}, 'fvol', {});
k = 0;

% --- 1) Schlanker Kragtraeger, Endquerlast (grosse Verdrehung) ------------
k = k + 1;
Lx = 10.0; Ly = 1.0; nx = scale*20; ny = scale*2;
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), -0.4/Ly);    % Endquerlast (klein: in Huelle)
S(k) = pack('Kragtraeger (Endquerlast)', coord, elem, mat, ...
            bcond, [right, 2*ones(numel(right),1), vals], [0 0]);

% --- 2) Kragtraeger unter Eigengewicht (grosse Durchbiegung) --------------
k = k + 1;
Lx = 8.0; Ly = 1.0; nx = scale*18; ny = scale*2;
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
left  = edge_nodes(coord, 'left', Lx, Ly);
bcond = fix_xy(left);
S(k) = pack('Kragtraeger (Eigengewicht)', coord, elem, mat, ...
            bcond, [], [0 -0.2]);

% --- 3) Tiefer Kragtraeger, Endquerlast (Schub + Nichtlin.) ---------------
k = k + 1;
Lx = 4.0; Ly = 2.0; nx = scale*12; ny = scale*6;
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), -4/Ly);      % klein: in Huelle
S(k) = pack('Tiefer Kragtraeger (Querlast)', coord, elem, mat, ...
            bcond, [right, 2*ones(numel(right),1), vals], [0 0]);

% --- 4) Scheibe unter grosser In-Plane-Scherung ---------------------------
k = k + 1;
Lx = 3.0; Ly = 3.0; nx = scale*8; ny = scale*8;
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
top    = edge_nodes(coord, 'top', Lx, Ly);
vals   = line_load_nodes(top, coord(top,1), 6/Lx);          % Horizontalschub (klein: in Huelle)
S(k) = pack('Scheibe (Scherung)', coord, elem, mat, ...
            bcond, [top, ones(numel(top),1), vals], [0 0]);

% --- 5) Kragtraeger, Axialzug (grosse Dehnung) ----------------------------
k = k + 1;
Lx = 6.0; Ly = 1.0; nx = scale*16; ny = scale*2;
[coord, elem, ~, mat] = mesh_rect(Lx, Ly, nx, ny, matcard);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), 60/Ly);      % Axialzug (Dehnung ~0.06, in Huelle)
S(k) = pack('Kragtraeger (Axialzug)', coord, elem, mat, ...
            bcond, [right, ones(numel(right),1), vals], [0 0]);

end

% ========================================================================
% Hilfsfunktionen: Strukturaufbau (identisch zum linearen Benchmark)
% ========================================================================
function [coord, elem, bcond, mat] = mesh_rect(Lx, Ly, nx, ny, matcard)
    [coord, elem, bcond, mat] = ...
        create_model_data_rectangle(Lx, Ly, nx, ny, [0 0 0 0], [0 0 0 0], matcard);
end

function bcond = fix_xy(nodes)
    nodes = nodes(:);
    bcond = [nodes, ones(numel(nodes),1),  zeros(numel(nodes),1);
             nodes, 2*ones(numel(nodes),1), zeros(numel(nodes),1)];
end

function ids = edge_nodes(coord, side, Lx, Ly)
    tol = 1e-6 * max(Lx, Ly);
    switch lower(side)
        case 'bottom', ids = find(abs(coord(:,2))      < tol);
        case 'top',    ids = find(abs(coord(:,2) - Ly) < tol);
        case 'left',   ids = find(abs(coord(:,1))      < tol);
        case 'right',  ids = find(abs(coord(:,1) - Lx) < tol);
        otherwise, error('Unbekannte Kante: %s', side);
    end
end

function vals = line_load_nodes(nodeIds, pos, q)
%LINE_LOAD_NODES Linienlast q -> aequivalente Knotenlasten (Trapezregel).
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
    vals(order) = v;
end

function s = pack(name, coord, elem, mat, bcond, fnode, fvol)
    s.name  = name;
    s.coord = coord;  s.elem = elem;  s.mat = mat;
    s.bcond = bcond;  s.fnode = fnode;  s.fvol = fvol;
end

% ========================================================================
% Hilfsfunktionen: Modell, Loesung, Zeit, Fehler
% ========================================================================
function model = make_nl_model(s, backend, numSteps, maxIter, tolR)
%MAKE_NL_MODEL Nichtlineares quad4-Modell (StVenant) mit gewaehltem Backend.
%   tolR (optional): absolutes Residuen-Kriterium fuer newton.m. Wird es
%   weggelassen, bleibt der Default aus init_setup (1e-8).
%
%   MODELL-EHRLICHES KRITERIUM (Plan Gate f): ein GELERNTES Residuum kann ein
%   absolutes 1e-8 prinzipiell nicht erreichen -- der Modellfehler des Netzes
%   liegt darueber. Der Aufrufer setzt deshalb ein RELATIVES Kriterium
%   tolR = TOL_REL * ||lambda*Fext||, berechnet EINMAL aus dem analytischen
%   Modell und identisch an BEIDE Backends gegeben (faire Iterationszahlen).
%   newton.m selbst bleibt unangetastet.
    setup = init_setup;
    setup.analysis.nl        = true;
    setup.analysis.numSteps  = numSteps;
    setup.element.type       = 'quad4';
    setup.element.backend    = backend;
    setup.material.name      = 'StVenant';
    setup.material.condition = 'planeStrain';
    setup.solver.maxIter     = maxIter;
    setup.solver.verbose     = false;
    if nargin > 4 && ~isempty(tolR)
        setup.solver.tolR = tolR;
    end
    model = init_model(s.coord, s.elem, s.mat, s.bcond, s.fnode, s.fvol, [], setup);
end

function tolR = reference_tolR(s, numSteps, maxIter, tolRel)
%REFERENCE_TOLR Relatives Konvergenzkriterium aus der aeusseren Last.
%   Einmal am unverformten Zustand (U = 0) mit dem ANALYTISCHEN Backend
%   assemblieren -> ||Fext|| auf den freien DOFs. Das Ergebnis geht identisch
%   an beide Backends, damit die Iterationszahlen vergleichbar bleiben.
    m0 = make_nl_model(s, 'matlab', numSteps, maxIter);
    [~, Fext0, ~] = assemble(m0, zeros(m0.info.NDOF, 1));
    tolR = tolRel * max(norm(Fext0(m0.dofs.free)), eps);
end

function [U, res, iters, ok] = solve_nl_quiet(model)
%SOLVE_NL_QUIET Nichtlineare Loesung, Konsolenausgabe unterdrueckt.
%   iters = Summe der Newton-Iterationen ueber alle Lastschritte.
%   ok    = alle Lastschritte konvergiert.
    txt = evalc(['[U, sr] = solve_FE(model); ', ...
                 'res = compute_model_results(model, sr(end));']); %#ok<NASGU>
    iters = sum([sr.nIter]);
    ok    = all([sr.converged]);
end

function t = time_assembly(model, U, R)
%TIME_ASSEMBLY Median-Zeit der globalen Assemblierung am Zustand U (deformiert).
%   Warm-up laedt das Netz (persistent) im KI-Element.
    assemble(model, U);
    ts = zeros(R,1);
    for r = 1:R
        t0 = tic;
        K = assemble(model, U); %#ok<NASGU>
        ts(r) = toc(t0);
    end
    t = median(ts);
end

function [fintErr, keErrN, fintP99, keP99] = elem_nl_error(model_fem, model_ai, U, nSample)
%ELEM_NL_ERROR Mittlerer rel. Element-Fehler von Finte und Ke am Zustand U.
%   Auf einer Zufallsstichprobe von bis zu nSample Elementen; beide Backends
%   werden mit demselben (deformierten) Elementzustand Ue ausgewertet.
%
%   fintErr : skalarer rel. Finte-Fehler [%] (mit Floor, s.u.).
%   keErrN  : 1x4-Vektor der mittleren rel. Ke-Fehler [%] in den Normen
%             [Frobenius, Spektral-2, 1-Norm, MaxAbs]. Da Ke symmetrisch ist,
%             gilt 1-Norm == inf-Norm; die Spektralnorm (groesster Singulaer-
%             wert) beschraenkt den rel. Fehler in JEDER Verformungsmode.
    NEL    = model_fem.info.NEL;
    DOF    = model_fem.info.DOF;
    rf = model_fem.element.routine;  ra = model_ai.element.routine;
    gp = model_fem.element.gp;  w = model_fem.element.w;  opts = model_fem.element.opts;
    MN = model_fem.material.name;  MC = model_fem.material.condition;

    if NEL > nSample
        idx = randperm(NEL, nSample);
    else
        idx = 1:NEL;
    end
    dF = zeros(numel(idx),1);   % absoluter Finte-Fehler je Element
    nF = zeros(numel(idx),1);   % ||Finte|| je Element (analytisch)
    eK = zeros(numel(idx),4);   % rel. Ke-Fehler je Element in 4 Normen
    for i = 1:numel(idx)
        e     = idx(i);
        dofs  = get_element_dofs(e, model_fem.elem, DOF);
        ce    = model_fem.coord(model_fem.elem(e,:),:);
        me    = model_fem.mat(e,:);
        be    = model_fem.fvol(e,:)';
        Ue    = U(dofs);
        [Kf,~,~,Ff] = rf(ce, me, be, 0, Ue, [], gp, w, MN, MC, opts);
        [Ka,~,~,Fa] = ra(ce, me, be, 0, Ue, [], gp, w, MN, MC, opts);
        dF(i) = norm(Fa - Ff);
        nF(i) = norm(Ff);

        Dk = Ka - Kf;
        nrmK = [norm(Kf,'fro'), norm(Kf,2), norm(Kf,1), max(abs(Kf(:)))];
        nrmD = [norm(Dk,'fro'), norm(Dk,2), norm(Dk,1), max(abs(Dk(:)))];
        eK(i,:) = nrmD ./ max(nrmK, eps);
    end
    % Relativer Finte-Fehler mit Floor (wie die Trainings-Metrik, 2026-07-15):
    % fast unbelastete Elemente (||Finte|| -> 0, z.B. an der freien Spitze)
    % wuerden den Mittelwert sonst beliebig aufblaehen, obwohl ihr absoluter
    % Fehler winzig ist. Floor = 2 % der RMS-Kraft der Stichprobe.
    floorF  = 0.02 * sqrt(mean(nF.^2));
    relF    = dF ./ max(nF, floorF) * 100;
    fintErr = mean(relF);
    keErrN  = mean(eK, 1) * 100;

    % PERZENTILE (Plan): Newton wird vom SCHLECHTESTEN Element limitiert,
    % nicht vom mittleren -- die Konvergenzrate haengt am dominanten
    % Eigenwert der Iterationsmatrix. Ein Mittelwert von 1.5 % kann trotzdem
    % problematisch sein, wenn einzelne Elemente 10-20 % Fehler haben.
    fintP99 = percentile_local(relF, 99);
    keP99   = percentile_local(eK(:,1) * 100, 99);
end

function p = percentile_local(x, q)
%PERCENTILE_LOCAL Perzentil ohne Statistics Toolbox (lineare Interpolation).
    x = sort(x(:));
    n = numel(x);
    if n == 1, p = x; return; end
    pos = max(1, min(n, (q/100) * n + 0.5));
    lo  = floor(pos);  hi = ceil(pos);
    p   = x(lo) + (pos - lo) * (x(hi) - x(lo));
end

function fintFree = global_fint_error_free(model_fem, model_ai, U)
%GLOBAL_FINT_ERROR_FREE Rel. Fehler des globalen inneren Kraftvektors [%],
%   ausgewertet NUR auf den freien Freiheitsgraden.
%
%   Motivation: Das FE-Gleichgewicht Fint = Fext wird von Newton
%   ausschliesslich auf den freien DOFs geloest (newton.m: Ra = R(dofsfree)).
%   An den gebundenen DOFs (Auflager) steht dagegen die Auflagerreaktion
%   (Fint = Fext + Freact, siehe compute_reaction_forces.m) -- oft eine sehr
%   grosse Kraft, die die relative Norm dominieren und den fuer die Loesung
%   relevanten Fehler verschleiern wuerde. Daher: nur die freien DOFs.
%
%   Beide Backends werden am DEMSELBEN (deformierten) Zustand U assembliert.
    free = model_fem.dofs.free;
    [~, ~, Fint_fem] = assemble(model_fem, U);
    [~, ~, Fint_ai ] = assemble(model_ai,  U);
    fintFree = norm(Fint_ai(free) - Fint_fem(free)) ...
             / max(norm(Fint_fem(free)), eps) * 100;
end

% ========================================================================
% Hilfsfunktionen: Gates
% ========================================================================
function s = gate_verdict(ok)
if ok, s = 'GRUEN'; else, s = 'ROT'; end
end

function print_cost_model()
%PRINT_COST_MODEL Arithmetik-Kostenmodell Netz-Kette vs. Gauss-Schleife.
%   ACHTUNG BEI DER INTERPRETATION: Diese Zaehlung erfasst nur ARITHMETIK,
%   nicht den Interpreter-Overhead. Sie erklaert deshalb NICHT die gemessenen
%   Zeiten und darf nicht als deren Begruendung gelesen werden:
%
%     * Das Netz braucht arithmetisch MEHR Operationen als das analytische
%       Element -- gemessen ist es in MATLAB trotzdem schneller.
%     * Grund: das analytische Element verbringt seine Zeit in der
%       interpretierten Doppelknotenschleife (4 GP x 16 Knotenpaare mit
%       winzigen Matrizen), das Netz in wenigen dichten BLAS-Produkten.
%       Der gemessene Vorteil kommt also aus der Ausfuehrungsform, nicht aus
%       weniger Rechenarbeit. In einer vektorisierten oder kompilierten
%       Referenzimplementierung koennte das analytische quad4 gewinnen.
%
%   Wofuer die Zahlen dann taugen: fuer die Extrapolation. Die FEM-Seite
%   waechst mit Gausspunktzahl, Knotenzahl und Materialkomplexitaet, die
%   Netz-Seite bleibt konstant -- dort liegt der eigentliche DLFE-Vorteil.
    netFile = fullfile(fileparts(which('element_quad4_nl_ai')), 'quad4_nl_W_network.mat');
    if ~exist(netFile, 'file'), return; end
    N = load(netFile, 'hidden', 'depth', 'num_linear_layers');
    h = double(N.hidden);  dep = double(N.depth);

    macsFwd = 16*h + (dep-1)*h*h + h;        % ein Forward des Skalar-MLP
    % Kosten je Elementaufruf: 1 Forward + Reverse (~2x) + Hessian mit 8
    % Richtungen (~2x8 Forward-Aequivalente) + Null-Pass (Forward+Reverse ~3x)
    macsElem = macsFwd * (1 + 2 + 16 + 3);
    macsK0   = 4 * (3*8*8 + 8*8*3);          % 4 GP: B'CB (grob)
    macsFEM  = 4 * (8*8*(3*3 + 4) + 3*8*3);  % 4 GP: Doppelknotenschleife (grob)

    fprintf('  Kostenmodell -- NUR ARITHMETIK, ohne Interpreter-Overhead:\n');
    fprintf('    Netz-Kette (h%d d%d): %8d   K0-Schleife: %6d   Summe KI: %8d\n', ...
        h, dep, macsElem, macsK0, macsElem + macsK0);
    fprintf('    analytisches quad4 : %8d   -> Verhaeltnis KI/FEM ~ %.1fx\n', ...
        macsFEM, (macsElem + macsK0) / macsFEM);
    fprintf('    ACHTUNG: Das Netz rechnet arithmetisch MEHR und ist gemessen\n');
    fprintf('    trotzdem schneller. Der Vorteil kommt daher, dass die\n');
    fprintf('    interpretierte Doppelknotenschleife durch wenige dichte\n');
    fprintf('    Matrixprodukte ersetzt wird -- nicht aus weniger Rechenarbeit.\n');
    fprintf('    Fuer die Extrapolation zaehlt: bei teuren Elementen (3D, viele\n');
    fprintf('    GP, komplexe Materialgesetze) waechst NUR die FEM-Seite.\n');
end

% ========================================================================
% Hilfsfunktionen: Plot
% ========================================================================
function plot_structure(model, ttl)
    plot_results(model, 'undeformed');
    plot_results(model, 'support', 1.0);
    draw_loads(model);
    title(ttl, 'FontSize', 9, 'Interpreter', 'none');
    axis equal; axis tight;
end

function draw_loads(model)
    coord = model.coord;
    span  = max(max(coord,[],1) - min(coord,[],1));
    Larr  = 0.18 * span;
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
    if any(model.fvol(:) ~= 0)
        xr = linspace(min(coord(:,1)), max(coord(:,1)), 4);
        yr = linspace(min(coord(:,2)), max(coord(:,2)), 4);
        [Xg, Yg] = meshgrid(xr(2:end-1), yr(2:end-1));
        quiver(Xg(:), Yg(:), zeros(numel(Xg),1), -0.5*Larr*ones(numel(Xg),1), 0, ...
               'Color', [0 0.55 0.2], 'LineWidth', 0.8, 'MaxHeadSize', 0.5);
    end
end
