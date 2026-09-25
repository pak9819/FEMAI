%FEMSOLID_EX_BRICK8_07_AI_NL_BENCHMARK Nichtlinearer FEM- vs. KI-Vergleich (brick8).
% ------------------------------------------------------------------------
% DESCRIPTION
%   3D-Gegenstueck zu FEMSolid_ex_quad4_07_ai_nl_benchmark.m: vergleicht
%   das analytische element_brick8_nl (Total Lagrange, StVenant) mit dem
%   Deep-Learned-Element element_brick8_nl_ai (Voll-Energie-Netz, modale
%   Metrik-Kette) auf sechs geometrisch nichtlinearen Strukturen:
%
%     1 Kragbalken, Endquerlast      4 Kragbalken, Axialzug
%     2 Kragbalken, Eigengewicht     5 Torsionsstab
%     3 Block unter Schub            6 Platte, beidseitig eingespannt, Druck
%
%   Innere Knoten sind um DISTORTION der Elementgroesse verschoben (Seed 7,
%   ausserhalb der Trainingstrajektorien). Das Lastniveau wird EINMAL mit
%   dem analytischen Element auf max ||E_green|| ~ E_TARGET eingeregelt
%   (innerhalb der Trainingshuelle 0.2) und dann fuer beide Backends
%   identisch verwendet.
%
%   Ausgewertet (wie quad4-07):
%     (K) Newton-Iterationen FEM vs. KI
%     (G) dU, dVM (von Mises analytisch aus U), Element-Fehler Finte/Ke
%         am deformierten Endzustand (Mittel und P99)
%     (Z) Assemblierungs- und Gesamtzeit
%   Konvergenzkriterium relativ: TOL_REL * ||Fext|| (beide Backends gleich).
%
%   Grafiken (helles Design, einheitliche Farben FEM = blau, KI = orange):
%     1  Strukturen: Ausgangsnetz, Lagerung (rot), Flaechenlast (blau),
%        Torsionsmoment (blauer Drehpfeil), Eigengewicht (gruen)
%     2  Kennzahlen: Iterationen, dU/dVM, Element-Fehler, Zeiten + Speedup
%        (Gates nur in der Konsolenausgabe, nicht in den Grafiken)
%     3  Endkonfiguration (Massstab 1): von Mises (FEM) auf der verformten
%        Oberflaeche, KI-Loesung als rotes Gitter, Ausgangslage grau
%     4  Knotenweiser Verschiebungsfehler |U_KI - U_FEM| / max|U_FEM|
%     5  Last-Verschiebungs-Pfade am Kontrollknoten (FEM Linie, KI Marker)
%   BRICK8_FIG_DIR=<Ordner> speichert die Grafiken zusaetzlich als PNG.
%
% PREREQUISITE
%   training/brick8/train_brick8_nl_W_network.py
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

close all; clear all; clc; %#ok<CLALL>
fprintf('=== brick8 NL-Benchmark: Klassische FEM vs. Deep Learned Energie ===\n');
fprintf('    St.-Venant-Kirchhoff 3D, nu = 0.3, Total Lagrange\n\n');

E_MOD      = 1.0e3;
NU         = 0.3;
NUM_STEPS  = 5;
MAX_ITER   = 60;
TOL_REL    = 1e-6;
MESH_SCALE = 1;          % skaliert alle Netze gemeinsam
DISTORTION = 0.2;
E_TARGET   = 0.08;       % max ||E_green|| am Endzustand (Huelle 0.2)
R_ELEM     = 5;
KE_SAMPLE  = 400;
if ~isempty(getenv('BRICK8_MESH_SCALE'))
    MESH_SCALE = str2double(getenv('BRICK8_MESH_SCALE'));
end

S  = define_structures(E_MOD, NU, MESH_SCALE, DISTORTION);
nS = numel(S);

SHORT = {S.short};
fig1 = new_figure('brick8 NL-Benchmark: Strukturen', [40 40 1500 820]);
tl1 = tiledlayout(fig1, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:nS
    ax = nexttile(tl1);
    m0 = make_model(S(k), 'matlab', 1, NUM_STEPS, MAX_ITER, []);
    plot_structure(ax, m0, S(k));
    title(ax, sprintf('%d  %s', k, S(k).name), 'FontWeight', 'bold');
    subtitle(ax, sprintf('%d Elemente, %d Knoten', m0.info.NEL, m0.info.NNODE), 'Color', [0.4 0.4 0.4]);
end
title(tl1, 'brick8 NL-Benchmark: Ausgangsnetz, Lagerung und Lasten', 'FontSize', 14, 'FontWeight', 'bold');
subtitle(tl1, sprintf('Innenknoten um %.0f %% der Elementgroesse verzerrt  |  rot: eingespannt, blau: Last, gruen: Eigengewicht', ...
    100*DISTORTION), 'Color', [0.35 0.35 0.35]);
drawnow;

mods = cell(nS,1);  UFs = cell(nS,1);  UAs = cell(nS,1);  rFs = cell(nS,1);
pathF = cell(nS,1);  pathA = cell(nS,1);

relU = nan(nS,1); relVM = nan(nS,1);
fintErr = zeros(nS,1); fintP99 = fintErr; keErr = fintErr; keP99 = fintErr;
itFEM = zeros(nS,1); itAI = itFEM; okFEM = false(nS,1); okAI = okFEM;
tElemFEM = zeros(nS,1); tElemAI = tElemFEM; tSolFEM = tElemFEM; tSolAI = tElemFEM;
nelem = zeros(nS,1); mEfin = zeros(nS,1); scl = zeros(nS,1);

fprintf(' Nr | Struktur                    |  NEL | Skala  | maxE  | itFEM | itKI |   dU    |   dVM\n');
fprintf(' ---|-----------------------------|------|--------|-------|-------|------|---------|--------\n');
for k = 1:nS
    scl(k) = calibrate_scale(S(k), NUM_STEPS, MAX_ITER, E_TARGET);
    tolR = reference_tolR(S(k), scl(k), NUM_STEPS, MAX_ITER, TOL_REL);
    mF = make_model(S(k), 'matlab', scl(k), NUM_STEPS, MAX_ITER, tolR);
    mA = make_model(S(k), 'ai',     scl(k), NUM_STEPS, MAX_ITER, tolR);
    nelem(k) = mF.info.NEL;

    t0 = tic; [UF, rF, itFEM(k), okFEM(k), srF] = solve_quiet(mF); tSolFEM(k) = toc(t0);
    t0 = tic; [UA, rA, itAI(k),  okAI(k),  srA] = solve_quiet(mA); tSolAI(k)  = toc(t0);
    mEfin(k) = max_green(mF, UF);

    mods{k} = mF;  UFs{k} = UF;  UAs{k} = UA;  rFs{k} = rF;
    pathF{k} = srF;  pathA{k} = srA;
    relU(k)  = norm(UA - UF) / norm(UF) * 100;
    relVM(k) = norm(rA.vonMises.node - rF.vonMises.node) / norm(rF.vonMises.node) * 100;
    [fintErr(k), keErr(k), fintP99(k), keP99(k)] = elem_error(mF, mA, UF, KE_SAMPLE);
    tElemFEM(k) = time_assembly(mF, UF, R_ELEM);
    tElemAI(k)  = time_assembly(mA, UF, R_ELEM);

    fprintf(' %2d | %-27s | %4d | %6.3f | %5.3f | %5d | %4d | %6.3f%% | %6.3f%%\n', ...
        k, S(k).name, nelem(k), scl(k), mEfin(k), itFEM(k), itAI(k), relU(k), relVM(k));
end

spElem = tElemFEM ./ tElemAI;
spSol  = tSolFEM ./ tSolAI;
fprintf('\n--- Element-Fehler am deformierten Endzustand [%%] und Zeiten ---\n');
fprintf(' Nr | Struktur                    | Finte mean | Finte P99 | Ke mean | Ke P99 | t_asm FEM/KI [ms] | Asm-Speedup | Ges-Speedup\n');
for k = 1:nS
    fprintf(' %2d | %-27s | %9.2f  | %8.2f  | %6.2f  | %5.2f  | %7.1f / %6.1f  | %10.2fx | %9.2fx\n', ...
        k, S(k).name, fintErr(k), fintP99(k), keErr(k), keP99(k), ...
        1e3*tElemFEM(k), 1e3*tElemAI(k), spElem(k), spSol(k));
end

gateK = all(okFEM) && all(okAI) && all(itAI <= itFEM + 3);
gateG = gateK && max(relU) < 0.5 && max(fintP99) < 5 && max(keP99) < 5;
gateZ = sum(tSolAI < tSolFEM) >= max(1, nS - 1);
fprintf('\n--- Gates ---\n');
fprintf('  K (alle konvergiert, KI <= FEM + 3 Iterationen)     : %s\n', verdict(gateK));
fprintf('  G (dU < 0.5 %%, Finte-P99 < 5 %%, Ke-P99 < 5 %%)       : %s\n', verdict(gateG));
fprintf('  Z (Gesamtloesung KI schneller auf >= %d von %d)      : %s\n', max(1,nS-1), nS, verdict(gateZ));
fprintf('  Newton-Iterationen gesamt: FEM %d | KI %d\n', sum(itFEM), sum(itAI));
fprintf('  Assemblierung Speedup Median %.2fx | Gesamtloesung Speedup Median %.2fx\n', ...
    median(spElem), median(spSol));

% ------------------------------------------------------------------------
% Grafiken
% ------------------------------------------------------------------------
C_FEM = [0.00 0.45 0.74];  C_KI = [0.85 0.33 0.10];  C_RED = [0.80 0.10 0.10];
x = 1:nS;

% --- 2) Kennzahlen -------------------------------------------------------
fig2 = new_figure('brick8 NL-Benchmark: Kennzahlen', [40 40 1500 820]);
tl2 = tiledlayout(fig2, 2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax = nexttile(tl2);
hb = bar(ax, x, [itFEM, itAI], 'grouped', 'EdgeColor', 'none');
hb(1).FaceColor = C_FEM;  hb(2).FaceColor = C_KI;
bar_labels(hb, '%d');
style_axes(ax, SHORT);  ylabel(ax, 'Newton-Iterationen (Summe)');
ylim(ax, [0, 1.18*max([itFEM; itAI])]);
legend(ax, {'FEM analytisch', 'KI-Element'}, 'Location', 'northwest');
title(ax, 'Konvergenz');
subtitle(ax, sprintf('Summe FEM %d | KI %d  ->  Konsistenz Ke = dFinte/dU', sum(itFEM), sum(itAI)));

ax = nexttile(tl2);
hb = bar(ax, x, [relU, relVM], 'grouped', 'EdgeColor', 'none');
hb(1).FaceColor = [0.30 0.60 0.85];  hb(2).FaceColor = [0.55 0.35 0.75];
bar_labels(hb, '%.2f');
style_axes(ax, SHORT);  ylabel(ax, 'rel. Fehler [%]');
ylim(ax, [0, 1.25*max([relU; relVM])]);
legend(ax, {'Verschiebung dU', 'von Mises dVM'}, 'Location', 'northwest');
title(ax, 'Loesungsgenauigkeit');

ax = nexttile(tl2);
hb = bar(ax, x, [fintP99, keP99], 'grouped', 'EdgeColor', 'none');
hb(1).FaceColor = [0.93 0.60 0.20];  hb(2).FaceColor = [0.45 0.70 0.35];
hold(ax, 'on');
xo = [hb(1).XEndPoints; hb(2).XEndPoints].';
plot(ax, xo(:,1), fintErr, 'kd', 'MarkerFaceColor', 'w', 'MarkerSize', 6);
plot(ax, xo(:,2), keErr,   'kd', 'MarkerFaceColor', 'w', 'MarkerSize', 6, 'HandleVisibility', 'off');
set(ax, 'YScale', 'log');
style_axes(ax, SHORT);  ylabel(ax, 'rel. Element-Fehler [%]');
ylim(ax, [0.5, 2*max([fintP99; keP99])]);
legend(ax, {'Finte P99', 'Ke P99', 'Mittelwert'}, 'Location', 'northwest');
title(ax, 'Element-Fehler am deformierten Endzustand');

ax = nexttile(tl2);
hb = bar(ax, x, 1e3*[tElemFEM, tElemAI], 'grouped', 'EdgeColor', 'none');
hb(1).FaceColor = C_FEM;  hb(2).FaceColor = C_KI;
for k = 1:nS
    text(ax, k, 1e3*max(tElemFEM(k), tElemAI(k)), sprintf('%.2fx', spElem(k)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontWeight', 'bold', 'FontSize', 9);
end
style_axes(ax, SHORT);  ylabel(ax, 'Assemblierung [ms]');
ylim(ax, [0, 1.2e3*max([tElemFEM; tElemAI])]);
legend(ax, {'FEM analytisch', 'KI-Element'}, 'Location', 'northwest');
title(ax, 'Laufzeit (Zahl = Speedup)');
subtitle(ax, sprintf('Median Assemblierung %.2fx | Gesamtloesung %.2fx', median(spElem), median(spSol)));

title(tl2, 'Deep Learned Energie, brick8 nichtlinear: FEM vs. KI', 'FontSize', 14, 'FontWeight', 'bold');

% --- 3) Endkonfiguration --------------------------------------------------
fig3 = new_figure('brick8 NL-Benchmark: Verformung', [40 40 1500 820]);
tl3 = tiledlayout(fig3, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:nS
    ax = nexttile(tl3);
    m = mods{k};
    draw_surface(ax, m, zeros(size(UFs{k})), [], [0.55 0.55 0.55], 'none');     % Ausgangslage
    draw_surface(ax, m, UFs{k}, rFs{k}.vonMises.node, [0.25 0.25 0.25], 'interp');
    draw_surface(ax, m, UAs{k}, [], C_RED, 'none');                             % KI-Gitter
    colormap(ax, turbo(256));
    cb = colorbar(ax);  cb.Label.String = 'von Mises (FEM)';
    finish_3d(ax);
    title(ax, sprintf('%d  %s', k, S(k).name), 'FontWeight', 'bold');
    subtitle(ax, sprintf('dU = %.2f %%   dVM = %.2f %%', relU(k), relVM(k)));
end
title(tl3, 'Endkonfiguration (Massstab 1)', 'FontSize', 14, 'FontWeight', 'bold');
subtitle(tl3, 'Farbe: von Mises FEM  |  rotes Gitter: KI-Loesung  |  graues Gitter: Ausgangslage', ...
    'Color', [0.35 0.35 0.35]);

% --- 4) Verschiebungsfehler ------------------------------------------------
fig4 = new_figure('brick8 NL-Benchmark: Verschiebungsfehler', [40 40 1500 820]);
tl4 = tiledlayout(fig4, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:nS
    ax = nexttile(tl4);
    dUn = reshape(UAs{k} - UFs{k}, 3, []).';
    Ufe = reshape(UFs{k}, 3, []).';
    ern = sqrt(sum(dUn.^2, 2)) / max(sqrt(sum(Ufe.^2, 2))) * 100;
    draw_surface(ax, mods{k}, UFs{k}, ern, [0.3 0.3 0.3], 'interp');
    colormap(ax, flipud(hot(256)));
    clim(ax, [0, max(max(ern), 1e-6)]);
    cb = colorbar(ax);  cb.Label.String = 'Fehler [% von max|U|]';
    finish_3d(ax);
    title(ax, sprintf('%d  %s', k, S(k).name), 'FontWeight', 'bold');
    subtitle(ax, sprintf('max %.2f %%   Mittel %.2f %%', max(ern), mean(ern)));
end
title(tl4, 'Knotenweiser Verschiebungsfehler |U_{KI} - U_{FEM}| / max|U_{FEM}|', ...
    'FontSize', 14, 'FontWeight', 'bold');

% --- 5) Last-Verschiebungs-Pfade -------------------------------------------
fig5 = new_figure('brick8 NL-Benchmark: Lastpfade', [40 40 1500 820]);
tl5 = tiledlayout(fig5, 2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');
for k = 1:nS
    ax = nexttile(tl5);
    Un = reshape(UFs{k}, 3, []).';
    [~, node] = max(sum(Un.^2, 2));                  % Knoten mit groesster Verschiebung
    [lamF, uF] = load_path(pathF{k}, node);
    [lamA, uA] = load_path(pathA{k}, node);
    plot(ax, uF, lamF, '-o', 'Color', C_FEM, 'LineWidth', 1.8, 'MarkerFaceColor', C_FEM, 'MarkerSize', 5);
    hold(ax, 'on');
    plot(ax, uA, lamA, 'x', 'Color', C_KI, 'LineWidth', 1.8, 'MarkerSize', 10);
    grid(ax, 'on');  box(ax, 'on');
    xlabel(ax, sprintf('|u| am Knoten %d', node));  ylabel(ax, 'Lastfaktor \lambda');
    legend(ax, {'FEM analytisch', 'KI-Element'}, 'Location', 'southeast');
    title(ax, sprintf('%d  %s', k, S(k).name), 'FontWeight', 'bold');
    subtitle(ax, sprintf('Endwert: FEM %.4f | KI %.4f', uF(end), uA(end)));
end
title(tl5, 'Last-Verschiebungs-Pfade am Knoten mit groesster Verschiebung', ...
    'FontSize', 14, 'FontWeight', 'bold');
drawnow;

figDir = getenv('BRICK8_FIG_DIR');
if ~isempty(figDir)
    if ~exist(figDir, 'dir'), mkdir(figDir); end
    names = {'strukturen', 'kennzahlen', 'verformung', 'fehler', 'lastpfade'};
    figs  = [fig1, fig2, fig3, fig4, fig5];
    for i = 1:numel(figs)
        try
            exportgraphics(figs(i), fullfile(figDir, ['brick8_07_' names{i} '.png']), 'Resolution', 120);
        catch me
            warning('brick8_07:Export', 'PNG-Export %s fehlgeschlagen: %s', names{i}, me.message);
        end
    end
    fprintf('Grafiken gespeichert in %s\n', figDir);
end
fprintf('\nFertig. Grafiken erzeugt.\n');


% ========================================================================
function s = verdict(ok)
if ok, s = 'GRUEN'; else, s = 'ROT'; end
end

function S = define_structures(E, nu, sc, dist)
S = struct('name', {}, 'short', {}, 'coord', {}, 'elem', {}, 'mat', {}, 'bcond', {}, 'fnode', {}, 'fvol', {}, 'torsion', {});

% 1) Kragbalken, Endquerlast
[c, e, f, q] = create_model_data_box(6, 1, 1, 12*sc, 2*sc, 2*sc, dist, 7);
S(1) = pack('Kragbalken (Endquerlast)', 'Krag. Querlast', c, e, E, nu, clamp(f.x0), box_face_load(c, q.x1, 3, -1.0), [0 0 0]);

% 2) Kragbalken, Eigengewicht
[c, e, f] = create_model_data_box(5, 1, 1, 10*sc, 2*sc, 2*sc, dist, 7);
S(2) = pack('Kragbalken (Eigengewicht)', 'Krag. Eigengew.', c, e, E, nu, clamp(f.x0), [], [0 0 -1.0]);

% 3) Block unter Schub
[c, e, f, q] = create_model_data_box(2, 2, 2, 4*sc, 4*sc, 4*sc, dist, 7);
S(3) = pack('Block (Schub)', 'Block Schub', c, e, E, nu, clamp(f.z0), box_face_load(c, q.z1, 1, 10), [0 0 0]);

% 4) Kragbalken, Axialzug
[c, e, f, q] = create_model_data_box(5, 1, 1, 10*sc, 2*sc, 2*sc, dist, 7);
S(4) = pack('Kragbalken (Axialzug)', 'Krag. Zug', c, e, E, nu, clamp(f.x0), box_face_load(c, q.x1, 1, 50), [0 0 0]);

% 5) Torsionsstab
[c, e, f, q] = create_model_data_box(6, 1, 1, 12*sc, 3*sc, 3*sc, dist, 7);
S(5) = pack('Torsionsstab', 'Torsion', c, e, E, nu, clamp(f.x0), torsion_load(c, q.x1, [0.5 0.5], 40), [0 0 0]);
S(5).torsion = [6 0.5 0.5];                     % Angriffspunkt des Moments (x1-Flaeche)

% 6) Platte, beidseitig eingespannt, Flaechendruck
[c, e, f, q] = create_model_data_box(4, 3, 0.4, 10*sc, 6*sc, 2*sc, dist, 7);
S(6) = pack('Platte (Druck, 2x eingesp.)', 'Platte', c, e, E, nu, clamp([f.x0; f.x1]), ...
            box_face_load(c, q.z1, 3, -1.0), [0 0 0]);
end

function bc = clamp(nodes)
nodes = unique(nodes(:));
bc = [];
for d = 1:3
    bc = [bc; nodes, d*ones(numel(nodes),1), zeros(numel(nodes),1)]; %#ok<AGROW>
end
end

function fnode = torsion_load(coord, quads, yz0, q)
a = 1/sqrt(3);  gp = [-a -a; a -a; a a; -a a];
f = zeros(size(coord,1), 3);
for e = 1:size(quads, 1)
    X = coord(quads(e,:), :);
    for g = 1:4
        r = gp(g,1);  s = gp(g,2);
        h   = 0.25 * [(1-r)*(1-s); (1+r)*(1-s); (1+r)*(1+s); (1-r)*(1+s)];
        h_r = 0.25 * [-(1-s); (1-s); (1+s); -(1+s)];
        h_s = 0.25 * [-(1-r); -(1+r); (1+r); (1-r)];
        dA  = norm(cross(X.' * h_r, X.' * h_s));
        p   = X.' * h;
        f(quads(e,:), :) = f(quads(e,:), :) + h * (q * [0, -(p(3)-yz0(2)), p(2)-yz0(1)]) * dA;
    end
end
fnode = [];
for d = 2:3
    nz = find(f(:,d) ~= 0);
    fnode = [fnode; nz, d*ones(numel(nz),1), f(nz,d)]; %#ok<AGROW>
end
end

function s = pack(name, short, c, e, E, nu, bc, fn, fv)
s.name = name;  s.short = short;  s.coord = c;  s.elem = e;  s.torsion = [];
s.mat = repmat([E nu NaN 0], size(e,1), 1);
s.bcond = bc;  s.fnode = fn;  s.fvol = fv;
end

function model = make_model(s, backend, scale, nSteps, maxIter, tolR)
setup = init_setup;
setup.analysis.nl        = true;
setup.analysis.numSteps  = nSteps;
setup.element.type       = 'brick8';
setup.element.backend    = backend;
setup.material.name      = 'StVenant';
setup.material.condition = '3D';
setup.solver.maxIter     = maxIter;
setup.solver.verbose     = false;
if nargin > 5 && ~isempty(tolR), setup.solver.tolR = tolR; end
fn = s.fnode;
if ~isempty(fn), fn(:,3) = fn(:,3) * scale; end
model = init_model(s.coord, s.elem, s.mat, s.bcond, fn, s.fvol * scale, [], setup);
end

function scale = calibrate_scale(s, nSteps, maxIter, Et)
%CALIBRATE_SCALE Lastfaktor, so dass max ||E_green|| (analytisch) ~ Et.
scale = 1;
for it = 1:6
    m = make_model(s, 'matlab', scale, nSteps, maxIter, []);
    tolR = 1e-8 * max(norm(assemble_fext(m)), eps);
    m.solver.tolR = tolR;
    [U, ~, ~, ok] = solve_quiet(m, false);
    if ~ok
        scale = 0.4 * scale;
        continue
    end
    mE = max_green(m, U);
    if abs(mE - Et) < 0.1 * Et, return; end
    scale = scale * min(4, max(0.25, Et / mE));
end
end

function F = assemble_fext(m)
[~, F] = assemble(m, zeros(m.info.NDOF, 1));
F = F(m.dofs.free);
end

function tolR = reference_tolR(s, scale, nSteps, maxIter, tolRel)
m0 = make_model(s, 'matlab', scale, nSteps, maxIter, []);
tolR = tolRel * max(norm(assemble_fext(m0)), eps);
end

function [U, res, iters, ok, sr] = solve_quiet(model, post)
if nargin < 2, post = true; end
if post
    txt = evalc('[U, sr] = solve_FE(model); res = compute_model_results(model, sr(end));'); %#ok<NASGU>
else
    txt = evalc('[U, sr] = solve_FE(model);'); %#ok<NASGU>
    res = [];
end
iters = sum([sr.nIter]);
ok = all([sr.converged]);
end

function mE = max_green(model, U)
mE = 0;
for e = 1:model.info.NEL
    ce = model.coord(model.elem(e,:), :);
    Un = reshape(U(get_element_dofs(e, model.elem, 3)), 3, 8);
    for g = 1:size(model.element.gp, 1)
        [~, dh] = shape_brick8(ce, model.element.gp(g,:));
        F = eye(3) + Un * dh;
        mE = max(mE, norm(0.5*(F.'*F - eye(3)), 'fro'));
    end
end
end

function t = time_assembly(model, U, R)
assemble(model, U);
ts = zeros(R,1);
for r = 1:R
    t0 = tic; assemble(model, U); ts(r) = toc(t0);
end
t = median(ts);
end

function [fintErr, keErr, fintP99, keP99] = elem_error(mF, mA, U, nSample)
NEL = mF.info.NEL;
rf = mF.element.routine;  ra = mA.element.routine;
gp = mF.element.gp;  w = mF.element.w;  MN = mF.material.name;  MC = mF.material.condition;
if NEL > nSample, idx = randperm(RandStream('mt19937ar','Seed',1), NEL, nSample); else, idx = 1:NEL; end
dF = zeros(numel(idx),1); nF = dF; eK = dF;
for i = 1:numel(idx)
    e = idx(i);
    dofs = get_element_dofs(e, mF.elem, 3);
    ce = mF.coord(mF.elem(e,:), :);  me = mF.mat(e,:);  be = mF.fvol(e,:)';
    [Kf,~,~,Ff] = rf(ce, me, be, 0, U(dofs), [], gp, w, MN, MC, struct());
    [Ka,~,~,Fa] = ra(ce, me, be, 0, U(dofs), [], gp, w, MN, MC, struct());
    dF(i) = norm(Fa - Ff);  nF(i) = norm(Ff);
    eK(i) = norm(Ka - Kf, 'fro') / norm(Kf, 'fro') * 100;
end
relF = dF ./ max(nF, 0.02 * sqrt(mean(nF.^2))) * 100;
fintErr = mean(relF);  keErr = mean(eK);
fintP99 = prctile_local(relF, 99);  keP99 = prctile_local(eK, 99);
end

% ========================================================================
% Hilfsfunktionen: Plot
% ========================================================================
function fig = new_figure(name, pos)
fig = figure('Name', name, 'NumberTitle', 'off', 'Position', pos, 'Color', 'w');
try
    theme(fig, 'light');          % helles Design auch im Batch-/Dark-Mode
catch
end
end

function style_axes(ax, labels)
set(ax, 'XTick', 1:numel(labels), 'XTickLabel', labels, 'XTickLabelRotation', 20, ...
        'FontSize', 10, 'Box', 'off', 'TickDir', 'out');
grid(ax, 'on');  ax.YGrid = 'on';  ax.XGrid = 'off';  ax.GridAlpha = 0.15;
xlim(ax, [0.4, numel(labels) + 0.6]);
end

function bar_labels(hb, fmt)
for i = 1:numel(hb)
    text(hb(i).Parent, hb(i).XEndPoints, hb(i).YEndPoints, ...
        compose(fmt, hb(i).YData), 'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', [0.25 0.25 0.25]);
end
end

function Fb = boundary_faces(elem)
%BOUNDARY_FACES Aeussere Vierecksflaechen eines brick8-Netzes.
loc = [1 2 3 4; 5 6 7 8; 1 2 6 5; 2 3 7 6; 3 4 8 7; 4 1 5 8];
F = zeros(6*size(elem,1), 4);
for f = 1:6
    F(f:6:end, :) = elem(:, loc(f,:));
end
[~, ia, ic] = unique(sort(F, 2), 'rows');
cnt = accumarray(ic, 1);
Fb = F(ia(cnt == 1), :);
end

function draw_surface(ax, model, U, field, edgeColor, faceMode)
%DRAW_SURFACE Oberflaeche in verformter Lage; field (je Knoten) oder Drahtgitter.
Fb = boundary_faces(model.elem);
xd = model.coord + reshape(U, 3, []).';
hold(ax, 'on');
if isempty(field)
    patch(ax, 'Faces', Fb, 'Vertices', xd, 'FaceColor', 'none', ...
          'EdgeColor', edgeColor, 'LineWidth', 0.6, 'EdgeAlpha', 0.8);
else
    patch(ax, 'Faces', Fb, 'Vertices', xd, 'FaceVertexCData', field(:), ...
          'FaceColor', faceMode, 'EdgeColor', edgeColor, 'EdgeAlpha', 0.25, 'LineWidth', 0.4);
end
end

function finish_3d(ax)
view(ax, -35, 22);
axis(ax, 'equal');  axis(ax, 'tight');
grid(ax, 'on');  ax.GridAlpha = 0.12;  box(ax, 'off');
xlabel(ax, 'x');  ylabel(ax, 'y');  zlabel(ax, 'z');
ax.FontSize = 9;
end

function plot_structure(ax, model, s)
c = model.coord;
Fb = boundary_faces(model.elem);
hold(ax, 'on');
patch(ax, 'Faces', Fb, 'Vertices', c, 'FaceColor', [0.88 0.90 0.93], ...
      'EdgeColor', [0.35 0.38 0.42], 'FaceAlpha', 0.85, 'LineWidth', 0.5);
fixn = unique(model.bcond(:,1));
plot3(ax, c(fixn,1), c(fixn,2), c(fixn,3), 's', 'Color', [0.80 0.10 0.10], ...
      'MarkerFaceColor', [0.80 0.10 0.10], 'MarkerSize', 4);
span = max(max(c,[],1) - min(c,[],1));
L = 0.22 * span;
blue = [0.05 0.30 0.80];
if ~isempty(s.torsion)
    % Torsionsmoment als Drehpfeil um die x-Achse
    p0 = s.torsion;  r = 0.75;  t = linspace(0.2, 1.75*pi, 60);
    xa = p0(1) + 0.25;
    plot3(ax, xa*ones(size(t)), p0(2) + r*cos(t), p0(3) + r*sin(t), '-', 'Color', blue, 'LineWidth', 2);
    te = t(end);  d = [0, -sin(te), cos(te)];
    quiver3(ax, xa, p0(2) + r*cos(te), p0(3) + r*sin(te), 0, 0.3*d(2), 0.3*d(3), 0, ...
            'Color', blue, 'LineWidth', 2, 'MaxHeadSize', 3);
    text(ax, xa, p0(2), p0(3) + r + 0.25, 'M_T', 'Color', blue, 'FontWeight', 'bold');
elseif ~isempty(model.fnode)
    F = zeros(model.info.NNODE, 3);
    for i = 1:size(model.fnode,1)
        F(model.fnode(i,1), model.fnode(i,2)) = F(model.fnode(i,1), model.fnode(i,2)) + model.fnode(i,3);
    end
    nz = find(sqrt(sum(F.^2, 2)) > 0);
    dirv = sum(F(nz,:), 1);  dirv = dirv / norm(dirv);       % Lastrichtung
    pick = nz(round(linspace(1, numel(nz), min(9, numel(nz)))));
    quiver3(ax, c(pick,1) - L*dirv(1), c(pick,2) - L*dirv(2), c(pick,3) - L*dirv(3), ...
            L*dirv(1)*ones(numel(pick),1), L*dirv(2)*ones(numel(pick),1), L*dirv(3)*ones(numel(pick),1), 0, ...
            'Color', blue, 'LineWidth', 1.4, 'MaxHeadSize', 0.6);
end
if any(model.fvol(:) ~= 0)
    b = model.fvol(1,:) / norm(model.fvol(1,:));
    ce = zeros(model.info.NEL, 3);
    for e = 1:model.info.NEL
        ce(e,:) = mean(c(model.elem(e,:),:), 1);
    end
    top = ce(:,3) > max(ce(:,3)) - 1e-9;
    idx = find(top);  idx = idx(round(linspace(1, numel(idx), min(6, numel(idx)))));
    quiver3(ax, ce(idx,1), ce(idx,2), max(c(:,3)) + 0.55*L*ones(numel(idx),1), ...
            0.5*L*b(1)*ones(numel(idx),1), 0.5*L*b(2)*ones(numel(idx),1), 0.5*L*b(3)*ones(numel(idx),1), 0, ...
            'Color', [0.10 0.55 0.20], 'LineWidth', 1.4, 'MaxHeadSize', 0.8);
end
finish_3d(ax);
end

function [lam, u] = load_path(sr, node)
%LOAD_PATH Lastfaktor und |u| am Knoten ueber alle Lastschritte (mit Nullpunkt).
lam = [0, [sr.loadscale]];
u = zeros(1, numel(sr) + 1);
for i = 1:numel(sr)
    u(i+1) = norm(sr(i).U(3*node-2:3*node));
end
end

function p = prctile_local(x, q)
x = sort(x(:));  n = numel(x);
if n == 1, p = x; return; end
pos = max(1, min(n, (q/100)*n + 0.5));
lo = floor(pos);  hi = ceil(pos);
p = x(lo) + (pos - lo) * (x(hi) - x(lo));
end
