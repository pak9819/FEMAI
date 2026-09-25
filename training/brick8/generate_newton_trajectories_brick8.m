%GENERATE_NEWTON_TRAJECTORIES_BRICK8 Echte Newton-Zustaende (brick8) als Trainingsdaten.
% ------------------------------------------------------------------------
% DESCRIPTION
%   3D-Gegenstueck zu training/quad4/generate_newton_trajectories.m:
%   rechnet ZUFAELLIGE nichtlineare Quader-Strukturen mit dem analytischen
%   element_brick8_nl (StVenant, E = 1000, nu = 0.3) und zeichnet in JEDER
%   Newton-Iteration (auch nicht konvergierte Iterierte) die Zustaende
%   (coord_e, Ue) einer Element-Stichprobe auf.
%
%   Strukturen: Quader Lx in [1.5, 6], Ly, Lz in [0.8, 2.5], 80-400
%   Elemente, Innenknoten-Verzerrung U(0, 0.25) der Elementgroesse.
%   Lagerung: 1 Flaeche x0 eingespannt | 2 Flaeche z0 eingespannt |
%             3 Flaechen x0 und x1 eingespannt
%   Last:     1 Querlast (Flaechenlast z) auf x1 | 2 Schub (x) auf z1 |
%             3 Eigengewicht (-z) | 4 Torsion um die x-Achse auf x1 |
%             5 Axialzug auf x1
%   Lasttyp gewichtet [0.22 0.14 0.18 0.30 0.16]; Lagerungen, die mit der
%   Last kollidieren, werden umgewuerfelt.
%   Lastamplitude wird auf ein zufaelliges Ziel max ||E_green|| in
%   [0.06, 0.14] (+-30 %) eingeregelt.
%
%   LEAKAGE-SPERRE: Die Benchmark-Strukturen aus
%   FEMSolid_ex_brick8_07_ai_nl_benchmark.m (feste Abmessungen, Seed 7)
%   kommen hier nicht vor. Die letzten N_VAL_STRUCT Strukturen sind
%   Validierung (is_val = 1).
%
% OUTPUT
%   training/brick8/newton_traj_states_brick8.mat
%     coords (8 x 3 x N), Ue (24 x N), struct_id (1 x N), is_val (1 x N)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

close all; clear; clc;

thisDir = fileparts(mfilename('fullpath'));
run(fullfile(thisDir, '..', '..', 'startup.m'));

rng(2031);

N_STRUCT     = 60;
N_VAL_STRUCT = 6;
N_STEPS      = 6;
MAX_ITER     = 25;
TOL_REL      = 1e-8;
SAMPLE_FRAC  = 0.30;

E_MOD = 1.0e3; NU = 0.3;

allCoords = zeros(8, 3, 0);
allUe     = zeros(24, 0);
allSid    = zeros(1, 0);

fprintf('=== brick8 Newton-Trajektorien (%d Strukturen, %d Val) ===\n\n', N_STRUCT, N_VAL_STRUCT);
fprintf('  Nr | Geometrie          |  NEL | Lager | Last | Skala   | maxE  | Zustaende | Zeit\n');

for k = 1:N_STRUCT
    t0 = tic;
    Lx = 1.5 + 4.5 * rand();
    Ly = 0.8 + 1.7 * rand();
    Lz = 0.8 + 1.7 * rand();
    nelTarget = 80 + round(320 * rand());
    E_TARGET  = 0.06 + 0.08 * rand();         % Ziel-Dehnungsniveau je Struktur
    h  = (Lx * Ly * Lz / nelTarget)^(1/3);
    nx = max(2, round(Lx / h));  ny = max(1, round(Ly / h));  nz = max(1, round(Lz / h));
    dist = 0.25 * rand();
    % Lasttyp gewichtet (Torsion und Querlast haeufiger: Verwindungs- und
    % Biegezustaende sind im synthetischen Sampler schwach vertreten)
    loadType = find(rand() < cumsum([0.22 0.14 0.18 0.30 0.16]), 1);
    while true
        supportType = randi(3);
        % Last auf eingespannter Flaeche (3/4, 3/5) oder sinnlos (2/4) ausschliessen
        if ~(supportType == 3 && any(loadType == [4 5])) && ~(supportType == 2 && loadType == 4)
            break;
        end
    end
    S = build_random_structure(Lx, Ly, Lz, nx, ny, nz, dist, k, supportType, loadType, E_MOD);

    scale = 1.0;  okRun = false;  mE = NaN;
    for trial = 1:5
        model = make_model(S, scale, N_STEPS, MAX_ITER, NU, E_MOD);
        [U, ok] = run_newton_capture(model, N_STEPS, MAX_ITER, TOL_REL, 0);
        if ~ok
            scale = scale * 0.35;
            continue;
        end
        mE = max_green_strain(model, U);
        okRun = true;
        if abs(mE - E_TARGET) <= 0.3 * E_TARGET
            break;
        end
        scale = scale * min(4.0, max(0.25, E_TARGET / max(mE, 1e-9)));
        okRun = false;
    end
    if ~okRun && ~(mE > 0.02 && mE < 0.2)
        fprintf('  %2d | verworfen (maxE %.3f)\n', k, mE);
        continue;
    end

    model = make_model(S, scale, N_STEPS, MAX_ITER, NU, E_MOD);
    [U, ok, cap] = run_newton_capture(model, N_STEPS, MAX_ITER, TOL_REL, SAMPLE_FRAC);
    if ~ok || isempty(cap.Ue)
        fprintf('  %2d | verworfen (Aufzeichnungslauf)\n', k);
        continue;
    end
    mE = max_green_strain(model, U);

    allCoords = cat(3, allCoords, cap.coords);
    allUe     = [allUe, cap.Ue];                                  %#ok<AGROW>
    allSid    = [allSid, k * ones(1, size(cap.Ue, 2))];           %#ok<AGROW>

    fprintf('  %2d | %4.1fx%4.1fx%4.1f d%.2f | %4d |   %d   |  %d   | %7.3f | %5.3f | %8d | %4.0f s\n', ...
        k, Lx, Ly, Lz, dist, model.info.NEL, supportType, loadType, scale, mE, ...
        size(cap.Ue, 2), toc(t0));
end

is_val = double(allSid > (N_STRUCT - N_VAL_STRUCT));
coords = allCoords;  Ue = allUe;  struct_id = allSid;
outFile = fullfile(thisDir, 'newton_traj_states_brick8.mat');
save(outFile, 'coords', 'Ue', 'struct_id', 'is_val', '-v7');
fprintf('\nGesamt: %d Zustaende (%d aus Val-Strukturen)\nDatei:  %s\n', ...
    numel(allSid), sum(is_val), outFile);


% ========================================================================
function S = build_random_structure(Lx, Ly, Lz, nx, ny, nz, dist, seed, supportType, loadType, E)
[coord, elem, f, fq] = create_model_data_box(Lx, Ly, Lz, nx, ny, nz, dist, 100 + seed);

switch supportType
    case 1, fixn = f.x0;
    case 2, fixn = f.z0;
    otherwise, fixn = [f.x0; f.x1];
end
bcond = [];
for d = 1:3
    bcond = [bcond; fixn, d*ones(numel(fixn),1), zeros(numel(fixn),1)]; %#ok<AGROW>
end

A = Ly * Lz;  fnode = [];  fvol = [0 0 0];
Lb = Lx;  if supportType == 3, Lb = Lx / 4; end
switch loadType
    case 1      % Querlast (Endquerkraft P ~ 10 b h^2 / L)
        P = 10 * Ly * Lz^2 / Lb;
        fnode = box_face_load(coord, fq.x1, 3, -P / A);
        if supportType == 3, fnode = box_face_load(coord, fq.z1, 3, -P / (Lx*Ly)); end
    case 2      % Schub auf der Oberseite
        fnode = box_face_load(coord, fq.z1, 1, 20);
    case 3      % Eigengewicht
        fvol = [0 0 -33 * Lz / Lb^2];
    case 4      % Torsion um die x-Achse, Schubspannung ~ q * r
        q = 40 / max(Ly, Lz);
        fnode = face_torsion(coord, fq.x1, [Ly/2, Lz/2], q);
    otherwise   % Axialzug
        fnode = box_face_load(coord, fq.x1, 1, 60);
end
S.coord = coord;  S.elem = elem;  S.mat = repmat([E 0.3 NaN 0], size(elem,1), 1);
S.bcond = bcond;  S.fnode = fnode;  S.fvol = fvol;
end


function fnode = face_torsion(coord, quads, yz0, q)
%FACE_TORSION Konsistente Knotenlasten der Traktion t = q*(0, -(z-z0), y-y0).
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
        t   = q * [0, -(p(3) - yz0(2)), p(2) - yz0(1)];
        f(quads(e,:), :) = f(quads(e,:), :) + h * t * dA;
    end
end
fnode = [];
for d = 2:3
    nz = find(f(:,d) ~= 0);
    fnode = [fnode; nz, d*ones(numel(nz),1), f(nz,d)]; %#ok<AGROW>
end
end


function model = make_model(S, scale, numSteps, maxIter, ~, ~)
setup = init_setup;
setup.analysis.nl        = true;
setup.analysis.numSteps  = numSteps;
setup.element.type       = 'brick8';
setup.element.backend    = 'matlab';
setup.material.name      = 'StVenant';
setup.material.condition = '3D';
setup.solver.maxIter     = maxIter;
setup.solver.verbose     = false;
fnode = S.fnode;
if ~isempty(fnode), fnode(:,3) = fnode(:,3) * scale; end
model = init_model(S.coord, S.elem, S.mat, S.bcond, fnode, S.fvol * scale, [], setup);
end


function [U, ok, cap] = run_newton_capture(model, nSteps, maxIter, tolRel, sampleFrac)
NEL  = model.info.NEL;  DOF = model.info.DOF;  free = model.dofs.free;
U    = zeros(model.info.NDOF, 1);  ok = true;
nSample = max(1, round(sampleFrac * NEL));
nMax = nSample * nSteps * (maxIter + 1);
capC = zeros(8, 3, nMax);  capU = zeros(24, nMax);  nCap = 0;
cap.coords = zeros(8, 3, 0);  cap.Ue = zeros(24, 0);
tolR = [];
for step = 1:nSteps
    loadscale = step / nSteps;
    converged = false;
    for it = 0:maxIter
        [K, Fext0, Fint] = assemble(model, U);
        if isempty(tolR)
            tolR = tolRel * max(norm(Fext0(free)), eps);
        end
        R = Fint - loadscale * Fext0;
        normR = norm(R(free), 2);
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
        if it == maxIter, break; end
        U(free) = U(free) - K(free, free) \ R(free);
        if ~all(isfinite(U)), ok = false; return; end
    end
    if ~converged, ok = false; return; end
end
cap.coords = capC(:, :, 1:nCap);
cap.Ue     = capU(:, 1:nCap);
end


function mE = max_green_strain(model, U)
mE = 0;  NEL = model.info.NEL;  DOF = model.info.DOF;
for e = 1:NEL
    coord_e = model.coord(model.elem(e, :), :);
    Un = reshape(U(get_element_dofs(e, model.elem, DOF)), 3, 8);
    for g = 1:size(model.element.gp, 1)
        [~, dh] = shape_brick8(coord_e, model.element.gp(g, :));
        F = eye(3) + Un * dh;
        mE = max(mE, norm(0.5 * (F.' * F - eye(3)), 'fro'));
    end
end
end
