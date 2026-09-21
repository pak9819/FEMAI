%FEMSOLID_EX_QUAD4_07_AI_NL_BENCHMARK Nichtlinearer FEM- vs. KI-Vergleich.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Vergleicht das klassische nichtlineare quad4-Element (element_quad4_nl,
%   Total Lagrange) mit dem Deep-Learned-Element (element_quad4_nl_ai). Das
%   KI-Element lernt die volle Formaenderungsenergie; Finte und Ke entstehen
%   daraus durch Differentiation (Ke = dFinte/dUe per Konstruktion).
%
%   Fuenf geometrisch nichtlineare Strukturen mit bewusst MODERATEN Lasten
%   Die inneren Knoten werden um DISTORTION
%   (Anteil der Elementgroesse) zufaellig verschoben: regulaere Quadrat-Netze
%   sind im Training ueber die Newton-Trajektorien stark vertreten und
%   wuerden die Genauigkeit zu guenstig darstellen. DISTORTION = 0 liefert
%   die regulaeren Netze. Fuer grosse Verformungen
%   siehe FEMSolid_ex_quad4_10_ai_nl_large_deformation_benchmark.m.
%
%   Ausgewertet werden:
%     (K) KONVERGENZ  Newton-Iterationen FEM vs. KI. Gleiche Zahlen belegen
%                     die Konsistenz von Ke und Finte -- NICHT die
%                     Genauigkeit des Netzes.
%     (G) GENAUIGKEIT rel. Fehler der Verschiebungen (dU) und der
%                     von-Mises-Spannung (dVM, analytisch aus U berechnet),
%                     dazu Element-Fehler von Finte und Ke (Frobenius) am
%                     deformierten Zustand, jeweils Mittel und P99.
%     (Z) ZEIT        Assemblierung und Gesamtloesung.
%
%   Konvergenzkriterium: relativ, TOL_REL * ||Fext||, identisch fuer beide
%   Backends.
%
%   Material: St.-Venant-Kirchhoff, planeStrain, nu = 0.3 (fest ins Netz
%   eintrainiert, wird beim Laden HART geprueft). E und Dicke d sind frei.
%
% PREREQUISITE
%   Run training/quad4/train_quad4_nl_W_network.py first to generate
%   quad4_nl_W_network.mat.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-16
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
E_MOD   = 1.0e3;
NU      = 0.3;
D_THICK = 1.0;

NUM_STEPS  = 6;        % Lastschritte
MAX_ITER   = 150;      % max. Newton-Iterationen je Lastschritt
TOL_REL    = 1e-6;     % rel. Konvergenzkriterium (beide Backends)
MESH_SCALE = 2;        % skaliert alle Netze gemeinsam
DISTORTION = 0.2;      % innere Knoten +-20 % der Elementgroesse (0 = regulaer)
R_ELEM     = 12;       % Wiederholungen fuer die Assemblierungszeit
KE_SAMPLE  = 1000;     % Element-Fehler auf einer Stichprobe

% ------------------------------------------------------------------------
% Strukturen
% ------------------------------------------------------------------------
S  = define_nl_structures(E_MOD, NU, D_THICK, MESH_SCALE, DISTORTION);
nS = numel(S);

[ratioMax, angMin, angMax] = mesh_quality(S);
fprintf('Netzverzerrung %.0f %%: max. detJ-Verhaeltnis %.2f, Innenwinkel %.0f..%.0f Grad\n\n', ...
    100*DISTORTION, ratioMax, angMin, angMax);

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
% Benchmark-Schleife
% ------------------------------------------------------------------------
relU     = nan(nS,1);   relVM   = nan(nS,1);
fintErr  = zeros(nS,1); fintP99 = zeros(nS,1);
keErr    = zeros(nS,1); keP99   = zeros(nS,1);
itFEM    = zeros(nS,1); itAI    = zeros(nS,1);
okFEM    = false(nS,1); okAI    = false(nS,1);
tElemFEM = zeros(nS,1); tElemAI = zeros(nS,1);
tSolFEM  = zeros(nS,1); tSolAI  = zeros(nS,1);
nelem    = zeros(nS,1);

fprintf('Loese (NUM_STEPS = %d, MAX_ITER = %d) ...\n\n', NUM_STEPS, MAX_ITER);
fprintf(' Nr | Struktur                      |  NEL | itFEM | itKI |   dU    |   dVM\n');
fprintf(' ---|-------------------------------|------|-------|------|---------|--------\n');

for k = 1:nS
    tolR_k    = reference_tolR(S(k), NUM_STEPS, MAX_ITER, TOL_REL);
    model_fem = make_nl_model(S(k), 'matlab', NUM_STEPS, MAX_ITER, tolR_k);
    model_ai  = make_nl_model(S(k), 'ai',     NUM_STEPS, MAX_ITER, tolR_k);
    nelem(k)  = model_fem.info.NEL;

    t0 = tic;  [U_fem, res_fem, itFEM(k), okFEM(k)] = solve_nl_quiet(model_fem);
    tSolFEM(k) = toc(t0);
    t0 = tic;  [U_ai,  res_ai,  itAI(k),  okAI(k) ] = solve_nl_quiet(model_ai);
    tSolAI(k)  = toc(t0);

    relU(k)  = norm(U_ai - U_fem) / max(norm(U_fem), eps) * 100;
    vmF = res_fem.vonMises.node;  vmA = res_ai.vonMises.node;
    relVM(k) = norm(vmA - vmF) / max(norm(vmF), eps) * 100;

    [fintErr(k), keErr(k), fintP99(k), keP99(k)] = ...
        elem_nl_error(model_fem, model_ai, U_fem, KE_SAMPLE);

    tElemFEM(k) = time_assembly(model_fem, U_fem, R_ELEM);
    tElemAI(k)  = time_assembly(model_ai,  U_fem, R_ELEM);

    fprintf(' %2d | %-29s | %4d | %5d | %4d | %6.3f%% | %6.3f%%\n', ...
        k, S(k).name, nelem(k), itFEM(k), itAI(k), relU(k), relVM(k));
end

% ------------------------------------------------------------------------
% Element-Fehler und Zeiten
% ------------------------------------------------------------------------
spElem = tElemFEM ./ max(tElemAI, eps);
spSol  = tSolFEM  ./ max(tSolAI,  eps);

fprintf('\n--- Element-Fehler am deformierten Endzustand [%%] und Zeiten ---\n');
fprintf(' Nr | Struktur                      | Finte mean | Finte P99 | Ke mean | Ke P99 | Asm-Speedup | Ges-Speedup\n');
fprintf(' ---|-------------------------------|------------|-----------|---------|--------|-------------|------------\n');
for k = 1:nS
    fprintf(' %2d | %-29s | %9.2f  | %8.2f  | %6.2f  | %5.2f  | %10.2fx | %9.2fx\n', ...
        k, S(k).name, fintErr(k), fintP99(k), keErr(k), keP99(k), spElem(k), spSol(k));
end

% ------------------------------------------------------------------------
% Gates
% ------------------------------------------------------------------------
gateK = all(okFEM) && all(okAI) && all(itAI <= itFEM + 3);
gateG = gateK && max(relU) < 0.5 && max(fintP99) < 5 && max(keP99) < 5;
gateZ = sum(tSolAI < tSolFEM) >= max(1, nS - 1);

fprintf('\n--- Gates ---\n');
fprintf('  K (alle konvergiert, KI <= FEM + 3 Iterationen je Struktur) : %s\n', gate_verdict(gateK));
fprintf('  G (dU < 0.5 %%, Finte-P99 < 5 %%, Ke-P99 < 5 %%)               : %s\n', gate_verdict(gateG));
fprintf('  Z (Gesamtloesung KI schneller auf >= %d von %d Strukturen)    : %s\n', ...
    max(1, nS - 1), nS, gate_verdict(gateZ));
fprintf('  Newton-Iterationen gesamt: FEM %d | KI %d\n', sum(itFEM), sum(itAI));
fprintf('  Assemblierung Speedup Median %.2fx | Gesamtloesung Speedup Median %.2fx\n', ...
    median(spElem), median(spSol));

% ------------------------------------------------------------------------
% Grafiken
% ------------------------------------------------------------------------
figure('Name', 'NL-Benchmark: Kennzahlen', 'NumberTitle', 'off', ...
       'Position', [60 60 1500 420]);

subplot(1,4,1);
bar([itFEM, itAI]);
xlabel('Struktur Nr.'); ylabel('Newton-Iterationen (gesamt)');
title('Konvergenz'); legend({'FEM', 'KI'}, 'Location', 'northwest'); grid on;

subplot(1,4,2);
bar([relU, relVM]);
xlabel('Struktur Nr.'); ylabel('rel. Fehler [%]');
title('Loesung'); legend({'dU', 'dVM'}, 'Location', 'northwest'); grid on;

subplot(1,4,3);
bar([fintErr, fintP99, keErr, keP99]);
xlabel('Struktur Nr.'); ylabel('rel. Fehler [%]');
title('Element-Fehler (deformiert)');
legend({'Finte mean', 'Finte P99', 'Ke mean', 'Ke P99'}, 'Location', 'northwest'); grid on;

subplot(1,4,4);
bar(1e3*[tElemFEM, tElemAI]);
xlabel('Struktur Nr.'); ylabel('Zeit [ms]');
title('Assemblierung (deformiert)'); legend({'FEM', 'KI'}, 'Location', 'northwest'); grid on;

sgtitle('Deep Learned Energie (quad4, nichtlinear): FEM vs. KI');

fprintf('\nFertig. Grafiken erzeugt.\n');


% ========================================================================
% Strukturdefinition: geometrisch nichtlineare Szenarien
% ========================================================================
function S = define_nl_structures(E, nu, d, scale, distortion)
%DEFINE_NL_STRUCTURES Liefert geometrisch nichtlineare quad4-Beispiele.
%   Moderate Lasten mit Lastinkrementierung, damit Newton robust konvergiert.

matcard = [E, nu, d, 0];
S = struct('name', {}, 'coord', {}, 'elem', {}, 'mat', {}, ...
           'bcond', {}, 'fnode', {}, 'fvol', {});
k = 0;

% --- 1) Schlanker Kragtraeger, Endquerlast (grosse Verdrehung) ------------
k = k + 1;
Lx = 10.0; Ly = 1.0; nx = scale*20; ny = scale*2;
[coord, elem, mat] = mesh_rect(Lx, Ly, nx, ny, matcard, distortion);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), -0.4/Ly);    % Endquerlast (klein)
S(k) = pack('Kragtraeger (Endquerlast)', coord, elem, mat, ...
            bcond, [right, 2*ones(numel(right),1), vals], [0 0]);

% --- 2) Kragtraeger unter Eigengewicht (grosse Durchbiegung) --------------
k = k + 1;
Lx = 8.0; Ly = 1.0; nx = scale*18; ny = scale*2;
[coord, elem, mat] = mesh_rect(Lx, Ly, nx, ny, matcard, distortion);
left  = edge_nodes(coord, 'left', Lx, Ly);
bcond = fix_xy(left);
S(k) = pack('Kragtraeger (Eigengewicht)', coord, elem, mat, ...
            bcond, [], [0 -0.2]);

% --- 3) Tiefer Kragtraeger, Endquerlast (Schub + Nichtlin.) ---------------
k = k + 1;
Lx = 4.0; Ly = 2.0; nx = scale*12; ny = scale*6;
[coord, elem, mat] = mesh_rect(Lx, Ly, nx, ny, matcard, distortion);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), -4/Ly);      % klein
S(k) = pack('Tiefer Kragtraeger (Querlast)', coord, elem, mat, ...
            bcond, [right, 2*ones(numel(right),1), vals], [0 0]);

% --- 4) Scheibe unter grosser In-Plane-Scherung ---------------------------
k = k + 1;
Lx = 3.0; Ly = 3.0; nx = scale*8; ny = scale*8;
[coord, elem, mat] = mesh_rect(Lx, Ly, nx, ny, matcard, distortion);
bottom = edge_nodes(coord, 'bottom', Lx, Ly);
bcond  = fix_xy(bottom);
top    = edge_nodes(coord, 'top', Lx, Ly);
vals   = line_load_nodes(top, coord(top,1), 6/Lx);          % Horizontalschub (klein)
S(k) = pack('Scheibe (Scherung)', coord, elem, mat, ...
            bcond, [top, ones(numel(top),1), vals], [0 0]);

% --- 5) Kragtraeger, Axialzug (grosse Dehnung) ----------------------------
k = k + 1;
Lx = 6.0; Ly = 1.0; nx = scale*16; ny = scale*2;
[coord, elem, mat] = mesh_rect(Lx, Ly, nx, ny, matcard, distortion);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
bcond = fix_xy(left);
vals  = line_load_nodes(right, coord(right,2), 60/Ly);      % Axialzug (Dehnung ~0.06)
S(k) = pack('Kragtraeger (Axialzug)', coord, elem, mat, ...
            bcond, [right, ones(numel(right),1), vals], [0 0]);

end

% ========================================================================
% Hilfsfunktionen: Strukturaufbau
% ========================================================================
function [coord, elem, mat] = mesh_rect(Lx, Ly, nx, ny, matcard, distortion)
%MESH_RECT Rechtecknetz; innere Knoten um +-distortion*Elementgroesse verschoben.
%   Die Raender bleiben gerade (Lager und Lasten sitzen unveraendert).
%   Fester Seed -> reproduzierbare Netze.
    [coord, elem, ~, mat] = ...
        create_model_data_rectangle(Lx, Ly, nx, ny, [0 0 0 0], [0 0 0 0], matcard);
    if distortion > 0
        tol   = 1e-9 * max(Lx, Ly);
        inner = coord(:,1) > tol & coord(:,1) < Lx - tol & ...
                coord(:,2) > tol & coord(:,2) < Ly - tol;
        rs = RandStream('mt19937ar', 'Seed', 7);
        n  = nnz(inner);
        coord(inner,1) = coord(inner,1) + distortion * (Lx/nx) * (2*rand(rs, n, 1) - 1);
        coord(inner,2) = coord(inner,2) + distortion * (Ly/ny) * (2*rand(rs, n, 1) - 1);
    end
end

function [ratioMax, angMin, angMax] = mesh_quality(S)
%MESH_QUALITY Groesstes detJ-Verhaeltnis und Innenwinkelbereich aller Elemente.
    gp = [-1 -1; 1 -1; 1 1; -1 1] / sqrt(3);
    ratioMax = 1;  angMin = 180;  angMax = 0;
    for k = 1:numel(S)
        for e = 1:size(S(k).elem, 1)
            ce = S(k).coord(S(k).elem(e,:), :);
            dJ = zeros(4,1);
            for g = 1:4
                [~, ~, dJ(g)] = shape_quad4(ce, gp(g,:));
            end
            ratioMax = max(ratioMax, max(dJ) / min(dJ));
            for i = 1:4
                a = ce(mod(i-2,4)+1,:) - ce(i,:);
                b = ce(mod(i,4)+1,:)   - ce(i,:);
                ang = acosd(dot(a,b) / (norm(a)*norm(b)));
                angMin = min(angMin, ang);  angMax = max(angMax, ang);
            end
        end
    end
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

function [fintErr, keErr, fintP99, keP99] = elem_nl_error(model_fem, model_ai, U, nSample)
%ELEM_NL_ERROR Rel. Element-Fehler von Finte und Ke (Frobenius) am Zustand U.
%   Beide Backends werden auf einer Stichprobe von bis zu nSample Elementen
%   mit demselben (deformierten) Elementzustand Ue ausgewertet.
%   Finte mit Floor (2 % der RMS-Kraft): fast unbelastete Elemente (z.B. an
%   der freien Spitze) wuerden den Mittelwert sonst beliebig aufblaehen.
%   P99 zusaetzlich zum Mittel, weil Newton vom schlechtesten Element
%   limitiert wird, nicht vom mittleren.
    NEL = model_fem.info.NEL;  DOF = model_fem.info.DOF;
    rf = model_fem.element.routine;  ra = model_ai.element.routine;
    gp = model_fem.element.gp;  w = model_fem.element.w;  opts = model_fem.element.opts;
    MN = model_fem.material.name;  MC = model_fem.material.condition;

    if NEL > nSample, idx = randperm(NEL, nSample); else, idx = 1:NEL; end
    dF = zeros(numel(idx),1);  nF = zeros(numel(idx),1);  eK = zeros(numel(idx),1);
    for i = 1:numel(idx)
        e    = idx(i);
        dofs = get_element_dofs(e, model_fem.elem, DOF);
        ce   = model_fem.coord(model_fem.elem(e,:),:);
        me   = model_fem.mat(e,:);
        be   = model_fem.fvol(e,:)';
        Ue   = U(dofs);
        [Kf,~,~,Ff] = rf(ce, me, be, 0, Ue, [], gp, w, MN, MC, opts);
        [Ka,~,~,Fa] = ra(ce, me, be, 0, Ue, [], gp, w, MN, MC, opts);
        dF(i) = norm(Fa - Ff);
        nF(i) = norm(Ff);
        eK(i) = norm(Ka - Kf, 'fro') / max(norm(Kf, 'fro'), eps) * 100;
    end
    relF    = dF ./ max(nF, 0.02 * sqrt(mean(nF.^2))) * 100;
    fintErr = mean(relF);
    keErr   = mean(eK);
    fintP99 = percentile_local(relF, 99);
    keP99   = percentile_local(eK, 99);
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

% ========================================================================
% Hilfsfunktionen: Gates
% ========================================================================
function s = gate_verdict(ok)
if ok, s = 'GRUEN'; else, s = 'ROT'; end
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
