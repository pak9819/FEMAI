%GENERATE_NEWTON_TRAJECTORIES Sammelt echte Newton-Zustaende als Trainingsdaten.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Rechnet ZUFAELLIGE nichtlineare Strukturen mit dem ANALYTISCHEN Element
%   (element_quad4_nl) und zeichnet in JEDER Newton-Iteration die
%   Elementzustaende (coord_e, Ue) einer Stichprobe von Elementen auf.
%
%   Warum: der synthetische Sampler deckt den Zustandsraum breit ab, aber
%   nicht zielgenau den Teil, den der Solver spaeter tatsaechlich besucht
%   (Newton-Iterierte sind Nicht-Gleichgewichtszustaende). Diese Daten
%   schliessen genau diese Luecke (20 % des Trainingsmixes).
%
%   LEAKAGE-SPERRE: Die 5 Benchmark-Strukturen aus
%   FEMSolid_ex_quad4_07_ai_nl_benchmark.m sind HELD-OUT und kommen hier
%   NICHT vor -- es werden ausschliesslich zufaellige Geometrien, Lagerungen
%   und Lasten erzeugt. Die letzten N_VAL_STRUCT Strukturen werden per
%   is_val als VALIDIERUNGS-Strukturen markiert (nie im Training).
%
%   Die Newton-Schleife ist hier bewusst NACHGEBAUT (Logik aus
%   sourcecode/solver/newton.m), weil pro Iteration Zustaende abgegriffen
%   werden muessen -- newton.m selbst bleibt unangetastet.
%
% OUTPUT
%   training/quad4/newton_traj_states.mat mit
%     coords    (4 x 2 x N)  Elementknoten (physisch)
%     Ue        (8 x N)      Elementverschiebungen (physisch)
%     struct_id (1 x N)      Struktur-Nummer
%     is_val    (1 x N)      1 = Validierungs-Struktur
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-08-18
% ------------------------------------------------------------------------

close all; clear; clc;

thisDir = fileparts(mfilename('fullpath'));
run(fullfile(thisDir, '..', '..', 'startup.m'));

rng(2026);

N_STRUCT     = 25;      % Gesamtzahl Strukturen
N_VAL_STRUCT = 3;       % davon Validierung (die LETZTEN 3)
N_STEPS      = 6;       % Lastschritte
MAX_ITER     = 25;      % Newton-Iterationen je Lastschritt
TOL_R        = 1e-8;
SAMPLE_FRAC  = 0.10;    % Anteil Elemente je Iteration
E_TARGET     = 0.10;    % Ziel: max ||E_green|| am Endzustand

E_MOD = 1.0e3; NU = 0.3; D_THICK = 1.0;

allCoords = zeros(4, 2, 0);
allUe     = zeros(8, 0);
allSid    = zeros(1, 0);

fprintf('=== Newton-Trajektorien sammeln (%d Strukturen, %d Val) ===\n\n', ...
    N_STRUCT, N_VAL_STRUCT);
fprintf('  Nr | Geometrie      |  NEL | Lager | Last | Skala   | maxE  | Zustaende\n');
fprintf('  ---|----------------|------|-------|------|---------|-------|----------\n');

for k = 1:N_STRUCT

    % --------------------------------------------------------------------
    % Zufaellige Struktur (NICHT die Benchmark-Strukturen!)
    % --------------------------------------------------------------------
    Lx = 1.0 + 7.0 * rand();
    Ly = 1.0 + 5.0 * rand();
    nelTarget = 200 + round(600 * rand());
    nx = max(2, round(sqrt(nelTarget * Lx / Ly)));
    ny = max(2, round(sqrt(nelTarget * Ly / Lx)));
    supportType = randi(3);
    loadType    = randi(3);

    S = build_random_structure(Lx, Ly, nx, ny, supportType, loadType, ...
                               [E_MOD, NU, D_THICK, 0]);

    % --------------------------------------------------------------------
    % Lastamplitude einregeln: max ||E_green|| am Ende ~ E_TARGET
    % --------------------------------------------------------------------
    scale = 1.0;
    mE    = NaN;
    okRun = false;
    for trial = 1:5
        model = make_model(S, scale, N_STEPS, MAX_ITER, TOL_R);
        [U, ok] = run_newton_capture(model, N_STEPS, MAX_ITER, TOL_R, 0);
        if ~ok
            scale = scale * 0.35;
            continue;
        end
        mE = max_green_strain(model, U);
        okRun = true;
        if mE >= 0.05 && mE <= 0.15
            break;
        end
        scale = scale * min(4.0, max(0.25, E_TARGET / max(mE, 1e-9)));
    end

    if ~okRun
        fprintf('  %2d | verworfen (keine Konvergenz)\n', k);
        continue;
    end

    % --------------------------------------------------------------------
    % Lauf mit Aufzeichnung
    % --------------------------------------------------------------------
    model = make_model(S, scale, N_STEPS, MAX_ITER, TOL_R);
    [U, ok, cap] = run_newton_capture(model, N_STEPS, MAX_ITER, TOL_R, SAMPLE_FRAC);
    if ~ok || isempty(cap.Ue)
        fprintf('  %2d | verworfen (Aufzeichnungslauf)\n', k);
        continue;
    end
    mE = max_green_strain(model, U);

    allCoords = cat(3, allCoords, cap.coords);
    allUe     = [allUe, cap.Ue];                                  %#ok<AGROW>
    allSid    = [allSid, k * ones(1, size(cap.Ue, 2))];           %#ok<AGROW>

    fprintf('  %2d | %5.2f x %5.2f | %4d |   %d   |  %d   | %7.3f | %5.3f | %8d\n', ...
        k, Lx, Ly, model.info.NEL, supportType, loadType, scale, mE, size(cap.Ue, 2));
end

is_val = double(allSid > (N_STRUCT - N_VAL_STRUCT));

coords    = allCoords;
Ue        = allUe;
struct_id = allSid;

outFile = fullfile(thisDir, 'newton_traj_states.mat');
save(outFile, 'coords', 'Ue', 'struct_id', 'is_val', '-v7');

fprintf('\nGesamt: %d Zustaende (%d aus Val-Strukturen)\n', numel(allSid), sum(is_val));
fprintf('Datei:  %s\n', outFile);


% ========================================================================
% Strukturaufbau
% ========================================================================
function S = build_random_structure(Lx, Ly, nx, ny, supportType, loadType, matcard)
%BUILD_RANDOM_STRUCTURE Rechteckscheibe mit zufaelliger Lagerung und Last.

[coord, elem, ~, mat] = create_model_data_rectangle( ...
    Lx, Ly, nx, ny, [0 0 0 0], [0 0 0 0], matcard);

tol = 1e-6 * max(Lx, Ly);
left   = find(abs(coord(:,1))      < tol);
right  = find(abs(coord(:,1) - Lx) < tol);
bottom = find(abs(coord(:,2))      < tol);
top    = find(abs(coord(:,2) - Ly) < tol);

switch supportType
    case 1      % linke Kante voll eingespannt
        fixn = left;
    case 2      % Unterkante voll eingespannt
        fixn = bottom;
    otherwise   % beide unteren Ecken
        fixn = bottom([1, end]);
end
fixn  = fixn(:);
bcond = [fixn, ones(numel(fixn),1),  zeros(numel(fixn),1);
         fixn, 2*ones(numel(fixn),1), zeros(numel(fixn),1)];

fnode = [];
fvol  = [0 0];
switch loadType
    case 1      % Querlast an der rechten Kante
        vals  = line_load_nodes(coord(right,2), -0.4/Ly);
        fnode = [right, 2*ones(numel(right),1), vals];
    case 2      % Horizontalschub an der Oberkante
        vals  = line_load_nodes(coord(top,1), 3.0/Lx);
        fnode = [top, ones(numel(top),1), vals];
    otherwise   % Eigengewicht
        fvol = [0 -0.2];
end

S.coord = coord;  S.elem = elem;  S.mat = mat;
S.bcond = bcond;  S.fnode = fnode;  S.fvol = fvol;
end


function vals = line_load_nodes(pos, q)
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
v = q * trib;
vals = zeros(n,1);
vals(order) = v;
end


function model = make_model(S, scale, numSteps, maxIter, tolR)
%MAKE_MODEL Nichtlineares quad4-Modell (analytisches Backend), Last skaliert.
setup = init_setup;
setup.analysis.nl        = true;
setup.analysis.numSteps  = numSteps;
setup.element.type       = 'quad4';
setup.element.backend    = 'matlab';
setup.material.name      = 'StVenant';
setup.material.condition = 'planeStrain';
setup.solver.maxIter     = maxIter;
setup.solver.tolR        = tolR;
setup.solver.verbose     = false;

fnode = S.fnode;
if ~isempty(fnode)
    fnode(:,3) = fnode(:,3) * scale;
end
fvol = S.fvol * scale;

model = init_model(S.coord, S.elem, S.mat, S.bcond, fnode, fvol, [], setup);
end


% ========================================================================
% Newton mit Aufzeichnung (Logik aus sourcecode/solver/newton.m)
% ========================================================================
function [U, ok, cap] = run_newton_capture(model, nSteps, maxIter, tolR, sampleFrac)

NEL  = model.info.NEL;
DOF  = model.info.DOF;
free = model.dofs.free;
U    = zeros(model.info.NDOF, 1);
ok   = true;

% PREALLOKATION: inkrementelles Wachsen (cat/[...]) waere O(n^2) und
% dominiert sonst die Laufzeit voellig.
nSample = max(1, round(sampleFrac * NEL));
nMax    = nSample * nSteps * (maxIter + 1);
capC    = zeros(4, 2, nMax);
capU    = zeros(8, nMax);
nCap    = 0;
cap.coords = zeros(4, 2, 0);        % auch bei vorzeitigem Abbruch gueltig
cap.Ue     = zeros(8, 0);

for step = 1:nSteps
    loadscale = step / nSteps;
    converged = false;

    for it = 0:maxIter
        [K, Fext0, Fint] = assemble(model, U);
        R = Fint - loadscale * Fext0;
        normR = norm(R(free), 2);

        % Zustaende abgreifen (auch die NICHT konvergierten Iterierten --
        % genau die besucht der Solver spaeter)
        if sampleFrac > 0
            pick = randperm(NEL, nSample);
            for jj = 1:nSample
                e = pick(jj);
                nCap = nCap + 1;
                capC(:, :, nCap) = model.coord(model.elem(e, :), :);
                capU(:, nCap)    = U(get_element_dofs(e, model.elem, DOF));
            end
        end

        if it > 0 && normR <= tolR
            converged = true;
            break;
        end
        if it == maxIter
            break;
        end

        dUa = -K(free, free) \ R(free);
        U(free) = U(free) + dUa;

        if ~all(isfinite(U))
            ok = false;
            return;
        end
    end

    if ~converged
        ok = false;
        return;
    end
end

cap.coords = capC(:, :, 1:nCap);
cap.Ue     = capU(:, 1:nCap);
end


function mE = max_green_strain(model, U)
%MAX_GREEN_STRAIN Groesste Green-Lagrange-Norm ueber eine Element-Stichprobe.
mE  = 0;
NEL = model.info.NEL;
DOF = model.info.DOF;
idx = unique(round(linspace(1, NEL, min(NEL, 300))));
for e = idx
    coord_e = model.coord(model.elem(e, :), :);
    Un      = reshape(U(get_element_dofs(e, model.elem, DOF)), 2, 4).';
    for g = 1:size(model.element.gp, 1)
        [~, dh, ~] = shape_quad4(coord_e, model.element.gp(g, :));
        Fdef = eye(2) + Un.' * dh;
        Eg   = 0.5 * (Fdef.' * Fdef - eye(2));
        mE   = max(mE, norm(Eg, 'fro'));
    end
end
end
