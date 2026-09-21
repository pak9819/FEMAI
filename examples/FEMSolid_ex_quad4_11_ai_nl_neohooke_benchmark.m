%FEMSOLID_EX_QUAD4_11_AI_NL_NEOHOOKE_BENCHMARK Starke materielle Nichtlinearitaet (Neo-Hooke).
% ------------------------------------------------------------------------
% DESCRIPTION
%   FEM- vs. KI-Vergleich mit dem Neo-Hooke-Energienetz
%   (train_quad4_nl_W_network_neohooke.py). Ein 2D-Rechteck wird sehr stark
%   gezogen bzw. gedrueckt, dazu ein langer Kragtraeger als Kontrolle fuer
%   grosse Rotationen:
%
%     1) Rechteck 4 x 2, beidseitig eingespannt, Zug   +60 %
%     2) Rechteck 4 x 2, beidseitig eingespannt, Zug  +100 %
%     3) Rechteck 4 x 2, beidseitig eingespannt, Druck -30 %
%     4) Rechteck 4 x 2, beidseitig eingespannt, Druck -40 %
%     5) Langer Kragtraeger L/h = 40, Endlast (w/L ~ 0.9)
%
%   Die Rechtecke sind weggesteuert und beidseitig eingespannt (auch uy = 0):
%   das erzeugt einen inhomogenen Zustand und verhindert das Ausknicken
%   unter Druck (siehe FEMSolid_ex_quad4_10_ai_nl_large_deformation_benchmark.m).
%   Die inneren Knoten werden um DISTORTION verschoben (0 = regulaer).
%
%   Zum Vergleich rechnet das ANALYTISCHE Element dieselben Druckfaelle mit
%   St.-Venant-Kirchhoff: dort klappen Elemente um (J < 0) -- der Grund,
%   warum fuer starke Stauchung ein Neo-Hooke-Netz noetig ist.
%
%   Ausgewertet werden:
%     (K) KONVERGENZ  Newton-Iterationen FEM vs. KI (gleiches rel. Kriterium)
%     (G) GENAUIGKEIT dU, dVM und Element-Fehler von Finte und Ke
%                     (Frobenius, Mittel und P99) am deformierten Endzustand
%
%   Material: Neo-Hooke ('NeoHooke' = 'NeoHookean1'), planeStrain, nu = 0.3.
%
% PREREQUISITE
%   training/quad4/train_quad4_nl_W_network_neohooke.py muss
%   quad4_nl_W_network_NeoHookean1.mat erzeugt haben.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-17
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

close all; clear all; clc; %#ok<CLALL>

fprintf('=== quad4 NL-Benchmark: Neo-Hooke, starke Zug- und Druckverformung ===\n');
fprintf('    Neo-Hooke, planeStrain, nu = 0.3, Total Lagrange\n\n');

% ------------------------------------------------------------------------
% Feste Annahmen
% ------------------------------------------------------------------------
E_MOD   = 1.0e3;
NU      = 0.3;
D_THICK = 1.0;
MATNAME = 'NeoHooke';

MAX_ITER   = 50;       % max. Newton-Iterationen je Lastschritt
TOL_REL    = 1e-6;     % rel. Konvergenzkriterium, identisch fuer beide Backends
KE_SAMPLE  = 1000;     % Element-Fehler auf einer Stichprobe
DISTORTION = 0.2;      % innere Knoten +-20 % der Elementgroesse (0 = regulaer)

netFile = quad4_nl_ai_network_file(MATNAME);
if ~exist(netFile, 'file')
    error('%s fehlt -- zuerst train_quad4_nl_W_network_neohooke.py ausfuehren.', netFile);
end
N = load(netFile, 'hidden', 'depth');
fprintf('Netz h%d d%d\n', round(N.hidden), round(N.depth));

warning('off', 'element_quad4_nl_ai:OutOfHull');

% ------------------------------------------------------------------------
% Strukturen
% ------------------------------------------------------------------------
S  = define_neohooke_structures(E_MOD, NU, D_THICK, DISTORTION, MATNAME);
nS = numel(S);

[ratioMax, angMin, angMax] = mesh_quality(S);
fprintf('Netzverzerrung %.0f %%: max. detJ-Verhaeltnis %.2f, Innenwinkel %.0f..%.0f Grad\n\n', ...
    100*DISTORTION, ratioMax, angMin, angMax);

figure('Name', 'Neo-Hooke-Benchmark: Strukturen', 'NumberTitle', 'off', ...
       'Position', [60 60 1500 520]);
for k = 1:nS
    subplot(2, ceil(nS/2), k);
    model_k = make_nl_model(S(k), 'matlab', MAX_ITER, []);
    plot_structure(model_k, sprintf('%d) %s', k, S(k).name));
end
sgtitle('Neo-Hooke-Benchmark: Ausgangsgeometrie, Lagerung (rot), Last (blau)');
drawnow;

% ------------------------------------------------------------------------
% Benchmark-Schleife
% ------------------------------------------------------------------------
meas    = zeros(nS,1);
itFEM   = zeros(nS,1);  itAI    = zeros(nS,1);
okFEM   = false(nS,1); okAI    = false(nS,1);
relU    = nan(nS,1);    relVM   = nan(nS,1);
fintM   = zeros(nS,1);  fintP99 = zeros(nS,1);
keM     = zeros(nS,1);  keP99   = zeros(nS,1);
tSolFEM = zeros(nS,1);  tSolAI  = zeros(nS,1);
nelem   = zeros(nS,1);
U_all   = cell(nS,2);
M_all   = cell(nS,1);

fprintf(' Nr | Struktur                        |  NEL | Kennwert     | itFEM | itKI |  dU     | dVM\n');
fprintf(' ---|---------------------------------|------|--------------|-------|------|---------|--------\n');

for k = 1:nS
    tolR_k    = reference_tolR(S(k), MAX_ITER, TOL_REL);
    model_fem = make_nl_model(S(k), 'matlab', MAX_ITER, tolR_k);
    model_ai  = make_nl_model(S(k), 'ai',     MAX_ITER, tolR_k);
    nelem(k)  = model_fem.info.NEL;

    t0 = tic;  [U_fem, res_fem, itFEM(k), okFEM(k)] = solve_nl_quiet(model_fem);
    tSolFEM(k) = toc(t0);
    t0 = tic;  [U_ai,  res_ai,  itAI(k),  okAI(k) ] = solve_nl_quiet(model_ai);
    tSolAI(k)  = toc(t0);

    if okAI(k)
        relU(k)  = norm(U_ai - U_fem) / max(norm(U_fem), eps) * 100;
        vmF = res_fem.vonMises.node;  vmA = res_ai.vonMises.node;
        relVM(k) = norm(vmA - vmF) / max(norm(vmF), eps) * 100;
    end

    meas(k) = S(k).measure(U_fem);
    [fintM(k), keM(k), fintP99(k), keP99(k)] = ...
        elem_nl_error(model_fem, model_ai, U_fem, KE_SAMPLE);

    U_all(k,:) = {U_fem, U_ai};
    M_all{k}   = model_fem;

    fprintf(' %2d | %-31s | %4d | %-4s %7.3f | %5d | %4d | %6.3f%% | %6.3f%%\n', ...
        k, S(k).name, nelem(k), S(k).mlabel, meas(k), itFEM(k), itAI(k), relU(k), relVM(k));
end

% ------------------------------------------------------------------------
% Element-Fehler
% ------------------------------------------------------------------------
fprintf('\n--- Element-Fehler am deformierten Endzustand [%%] ---\n');
fprintf(' Nr | Struktur                        | Finte mean | Finte P99 | Ke mean | Ke P99 | t FEM   | t KI\n');
fprintf(' ---|---------------------------------|------------|-----------|---------|--------|---------|--------\n');
for k = 1:nS
    fprintf(' %2d | %-31s | %9.2f  | %8.2f  | %6.2f  | %5.2f  | %6.2fs | %6.2fs\n', ...
        k, S(k).name, fintM(k), fintP99(k), keM(k), keP99(k), tSolFEM(k), tSolAI(k));
end

% ------------------------------------------------------------------------
% Vergleich: St.-Venant (analytisch) unter denselben Druckfaellen
% ------------------------------------------------------------------------
fprintf('\n--- Vergleich: analytisches Element mit St.-Venant-Kirchhoff ---\n');
fprintf(' Struktur                        | konvergiert | J min   | Hinweis\n');
fprintf(' --------------------------------|-------------|---------|------------------------------\n');
for k = find(contains({S.name}, 'Druck'))
    sv = S(k);  sv.matname = 'StVenant';
    tolR_k = reference_tolR(S(k), MAX_ITER, TOL_REL);
    m_sv = make_nl_model(sv, 'matlab', MAX_ITER, tolR_k);
    [U_sv, ~, ~, ok_sv] = solve_nl_quiet(m_sv);
    if all(isfinite(U_sv))
        jl = min_detF(m_sv, U_sv);
    else
        jl = NaN;
    end
    note = 'Elemente klappen um (J <= 0)';
    if ok_sv && jl > 0, note = 'physikalisch zulaessig'; end
    fprintf(' %-31s | %-11s | %7.3f | %s\n', S(k).name, yesno(ok_sv), jl, note);
end

% ------------------------------------------------------------------------
% Gates
% ------------------------------------------------------------------------
gateK = all(okFEM) && all(okAI) && all(itAI <= itFEM + 3);
gateG = gateK && max(relU) < 1 && max(fintP99) < 5 && max(keP99) < 5;

fprintf('\n--- Gates ---\n');
fprintf('  K (alle konvergiert, KI <= FEM + 3 Iterationen je Struktur) : %s\n', ...
    gate_verdict(gateK));
fprintf('  G (dU < 1 %%, Finte-P99 < 5 %%, Ke-P99 < 5 %%)                 : %s\n', ...
    gate_verdict(gateG));
% ------------------------------------------------------------------------
% Grafiken
% ------------------------------------------------------------------------
figure('Name', 'Neo-Hooke-Benchmark: Verformung FEM vs. KI', 'NumberTitle', 'off', ...
       'Position', [60 60 1500 620]);
for k = 1:nS
    subplot(2, ceil(nS/2), k);
    plot_deformed_pair(M_all{k}, U_all{k,1}, U_all{k,2});
    title(sprintf('%d) %s  (dU %.2f %%)', k, S(k).name, relU(k)), ...
          'FontSize', 9, 'Interpreter', 'none');
end
sgtitle('Verformte Konfiguration (Massstab 1): FEM (blau) vs. KI-Knoten (rot)');

figure('Name', 'Neo-Hooke-Benchmark: Kennzahlen', 'NumberTitle', 'off', ...
       'Position', [60 60 1000 420]);
subplot(1,2,1);
bar([itFEM, itAI]);
xlabel('Struktur Nr.'); ylabel('Newton-Iterationen (gesamt)');
title('Konvergenz'); legend({'FEM', 'KI'}, 'Location', 'northwest'); grid on;

subplot(1,2,2);
bar([relU, fintP99, keP99]);
xlabel('Struktur Nr.'); ylabel('Fehler [%]');
title('Genauigkeit'); legend({'dU', 'Finte P99', 'Ke P99'}, 'Location', 'northwest');
grid on;

fprintf('\nFertig. Grafiken erzeugt.\n');


% ========================================================================
% Strukturdefinition
% ========================================================================
function S = define_neohooke_structures(E, nu, d, distortion, matname)
%DEFINE_NEOHOOKE_STRUCTURES Starke Zug-/Druckverformung und langer Kragtraeger.
%   Lastschritte je Fall fest und getestet: newton.m setzt die Rand-
%   verschiebung ohne Praediktor, zu grosse Schritte klappen Randelemente
%   in der ersten Iteration um. Der Druckfall reagiert empfindlich
%   (Eckelemente nahe J -> 0): bei -40 % konvergiert die FEM mit 23
%   Schritten, mit 90 nicht; -45 % nur mit bestimmten Schrittzahlen.

matcard = [E, nu, d, 0];
S = struct('name', {}, 'coord', {}, 'elem', {}, 'mat', {}, 'bcond', {}, ...
           'fnode', {}, 'fvol', {}, 'numSteps', {}, 'fscale', {}, ...
           'measure', {}, 'mlabel', {}, 'matname', {});

% --- 1-4) Rechteck, beidseitig eingespannt, weggesteuert ------------------
Lx = 4.0; Ly = 2.0;
[coord, elem, mat] = mesh_rect(Lx, Ly, 16, 8, matcard, distortion);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
rect = { 'Rechteck Zug +60 %',     0.60, 30
         'Rechteck Zug +100 %',    1.00, 50
         'Rechteck Druck -30 %',  -0.30, 15
         'Rechteck Druck -40 %',  -0.40, 23 };
for i = 1:size(rect, 1)
    [name, strain, nst] = rect{i,:};
    nr = numel(right);
    bcond = [fix_xy(left);
             right, ones(nr,1),   strain*Lx*ones(nr,1);
             right, 2*ones(nr,1), zeros(nr,1)];
    S(end+1) = pack(name, coord, elem, mat, bcond, zeros(0,3), [0 0], nst, ...
        E*d*Ly*abs(strain), @(U) strain, 'eps', matname); %#ok<AGROW>
end

% --- 5) Langer Kragtraeger mit Endquerlast --------------------------------
Lx = 40.0; Ly = 1.0;
[coord, elem, mat] = mesh_rect(Lx, Ly, 80, 4, matcard, distortion);
left  = edge_nodes(coord, 'left',  Lx, Ly);
right = edge_nodes(coord, 'right', Lx, Ly);
tip   = right(abs(coord(right,2) - Ly/2) < 1e-9);
vals  = line_load_nodes(right, coord(right,2), -1.5/Ly);
S(end+1) = pack('Langer Kragtraeger L/h=40', coord, elem, mat, fix_xy(left), ...
    [right, 2*ones(numel(right),1), vals], [0 0], 40, 0, ...
    @(U) -U(2*tip)/Lx, 'w/L', matname);
end

function s = yesno(ok)
    if ok, s = 'ja'; else, s = 'nein'; end
end

% ========================================================================
% Hilfsfunktionen: Strukturaufbau (wie FEMSolid_ex_quad4_10_ai_nl_large_deformation_benchmark)
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

function vals = line_load_nodes(nodeIds, pos, q) %#ok<INUSL>
%LINE_LOAD_NODES Linienlast q -> aequivalente Knotenlasten (Trapezregel).
    [ps, order] = sort(pos(:));
    n = numel(ps);
    trib = zeros(n,1);
    if n > 1
        trib(1)   = (ps(2) - ps(1)) / 2;
        trib(end) = (ps(end) - ps(end-1)) / 2;
        for i = 2:n-1
            trib(i) = (ps(i+1) - ps(i-1)) / 2;
        end
    end
    vals = zeros(n,1);
    vals(order) = q * trib;
end

function s = pack(name, coord, elem, mat, bcond, fnode, fvol, numSteps, fscale, measure, mlabel, matname)
    s.name  = name;
    s.coord = coord;  s.elem = elem;  s.mat = mat;
    s.bcond = bcond;  s.fnode = fnode;  s.fvol = fvol;
    s.numSteps = numSteps;
    s.fscale   = fscale;
    s.measure  = measure;
    s.mlabel   = mlabel;
    s.matname  = matname;
end

% ========================================================================
% Hilfsfunktionen: Modell, Loesung, Fehler
% ========================================================================
function model = make_nl_model(s, backend, maxIter, tolR)
    setup = init_setup;
    setup.analysis.nl        = true;
    setup.analysis.numSteps  = s.numSteps;
    setup.element.type       = 'quad4';
    setup.element.backend    = backend;
    setup.material.name      = s.matname;
    setup.material.condition = 'planeStrain';
    setup.solver.maxIter     = maxIter;
    setup.solver.verbose     = false;
    if ~isempty(tolR)
        setup.solver.tolR = tolR;
    end
    model = init_model(s.coord, s.elem, s.mat, s.bcond, s.fnode, s.fvol, [], setup);
end

function tolR = reference_tolR(s, maxIter, tolRel)
%REFERENCE_TOLR Relatives Kriterium: ||Fext|| (kraftgesteuert) bzw. die
%   Kraftskala s.fscale (weggesteuert, Fext = 0). Identisch fuer beide Backends.
    if s.fscale > 0
        tolR = tolRel * s.fscale;
        return
    end
    m0 = make_nl_model(s, 'matlab', maxIter, []);
    [~, Fext0, ~] = assemble(m0, zeros(m0.info.NDOF, 1));
    tolR = tolRel * max(norm(Fext0(m0.dofs.free)), eps);
end

function [U, res, iters, ok] = solve_nl_quiet(model)
%SOLVE_NL_QUIET Nichtlineare Loesung ohne Konsolenausgabe; Fehler -> ok = false.
    try
        txt = evalc(['[U, sr] = solve_FE(model); ', ...
                     'res = compute_model_results(model, sr(end));']); %#ok<NASGU>
        iters = sum([sr.nIter]);
        ok    = all([sr.converged]) && all(isfinite(U));
    catch
        U = nan(model.info.NDOF, 1);  res = [];  iters = NaN;  ok = false;
    end
end

function jmin = min_detF(model, U)
%MIN_DETF Kleinstes J = det F ueber alle Elemente und Gausspunkte
%   (J <= 0: Element umgeklappt).
    DOF  = model.info.DOF;
    gp   = model.element.gp;
    jmin = Inf;
    for e = 1:model.info.NEL
        ce = model.coord(model.elem(e,:),:);
        Ue = U(get_element_dofs(e, model.elem, DOF));
        for g = 1:size(gp,1)
            [~, dh, ~] = shape_quad4(ce, gp(g,:));
            jmin = min(jmin, det(eye(2) + compute_gradU(dh, Ue, 2, DOF)));
        end
    end
end

function [fintErr, keErr, fintP99, keP99] = elem_nl_error(model_fem, model_ai, U, nSample)
%ELEM_NL_ERROR Rel. Element-Fehler von Finte und Ke (Frobenius) am Zustand U.
%   Finte mit Floor (2 % der RMS-Kraft), damit fast unbelastete Elemente den
%   Mittelwert nicht aufblaehen (wie in FEMSolid_ex_quad4_07).
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
    x = sort(x(:));
    n = numel(x);
    if n == 1, p = x; return; end
    pos = max(1, min(n, (q/100) * n + 0.5));
    lo  = floor(pos);  hi = ceil(pos);
    p   = x(lo) + (pos - lo) * (x(hi) - x(lo));
end

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
    Larr  = 0.18 * max(max(coord,[],1) - min(coord,[],1));
    if isempty(model.fnode), return; end
    F = zeros(model.info.NNODE, 2);
    for i = 1:size(model.fnode,1)
        F(model.fnode(i,1), model.fnode(i,2)) = ...
            F(model.fnode(i,1), model.fnode(i,2)) + model.fnode(i,3);
    end
    mag = sqrt(sum(F.^2, 2));
    if max(mag) > 0
        sc = Larr / max(mag);
        nz = find(mag > 0);
        quiver(coord(nz,1) - sc*F(nz,1), coord(nz,2) - sc*F(nz,2), ...
               sc*F(nz,1), sc*F(nz,2), 0, ...
               'Color', [0 0.2 0.8], 'LineWidth', 1.0, 'MaxHeadSize', 0.4);
    end
end

function plot_deformed_pair(model, U_fem, U_ai)
%PLOT_DEFORMED_PAIR Unverformt (grau), FEM verformt (blau), KI-Knoten (rot).
    hold on;
    patch('Faces', model.elem, 'Vertices', model.coord, ...
          'FaceColor', 'none', 'EdgeColor', [0.75 0.75 0.75]);
    cf = model.coord + reshape(U_fem, 2, []).';
    patch('Faces', model.elem, 'Vertices', cf, ...
          'FaceColor', [0.55 0.7 0.95], 'EdgeColor', [0.2 0.3 0.6]);
    if all(isfinite(U_ai))
        ca = model.coord + reshape(U_ai, 2, []).';
        plot(ca(:,1), ca(:,2), 'r.', 'MarkerSize', 5);
    end
    axis equal; axis tight; grid on;
end
