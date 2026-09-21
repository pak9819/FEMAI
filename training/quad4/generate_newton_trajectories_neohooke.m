%GENERATE_NEWTON_TRAJECTORIES_NEOHOOKE Newton-Zustaende (Neo-Hooke) als Trainingsdaten.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Gegenstueck zu generate_newton_trajectories.m fuer das Neo-Hooke-Netz
%   (train_quad4_nl_W_network_neohooke.py). Rechnet ZUFAELLIGE Strukturen mit
%   dem ANALYTISCHEN Element (element_quad4_nl, 'NeoHooke') und zeichnet in
%   JEDER Newton-Iteration die Zustaende einer Element-Stichprobe auf.
%
%   Unterschiede zur St.-Venant-Variante:
%     * Lastfaelle mit starker materieller Nichtlinearitaet: beidseitig
%       eingespannte Scheiben unter weggesteuertem Zug UND Druck, dazu
%       Querlast, Oberkantenschub und Eigengewicht.
%     * VERZERRTE Netze (innere Knoten zufaellig verschoben): reine
%       Quadrat-Netze wuerden die Trainingsdaten auf eine Elementform
%       konzentrieren (Befund des Benchmark-Audits vom 16.09.2026).
%     * Lastniveau ueber die Neo-Hooke-Huelle eingeregelt: Hencky-Dehnung
%       ||ln U|| und Volumenverhaeltnis J an allen Gausspunkten.
%     * Tangenten-Praediktor je Lastschritt, damit weggesteuerte Raender
%       keine Elemente am Rand umklappen (J < 0).
%
%   LEAKAGE-SPERRE: keine Scheibe 4 x 2 (+-0.5 / +-0.3) und keine
%   Kragtraeger mit L/h > 8 -- das sind die Benchmark-Strukturen. Die
%   letzten N_VAL_STRUCT Strukturen sind VALIDIERUNG (is_val = 1).
%
% OUTPUT
%   training/quad4/newton_traj_states_neohooke.mat mit
%     coords (4x2xN), Ue (8xN), struct_id (1xN), is_val (1xN)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-17
% ------------------------------------------------------------------------

close all; clear; clc;

thisDir = fileparts(mfilename('fullpath'));
run(fullfile(thisDir, '..', '..', 'startup.m'));

rng(2027);

N_STRUCT     = 40;
N_VAL_STRUCT = 5;
N_STEPS      = 10;
MAX_ITER     = 25;
SAMPLE_FRAC  = 0.10;
MAX_DIST     = 0.25;        % Netzverzerrung bis +-25 % der Elementgroesse

% Neo-Hooke-Huelle (identisch zu quad4_nh_ref.py) und Ziel-Ausnutzung
H_MAX = 1.0;  J_MIN = 0.4;  J_MAX = 1.8;
H_TARGET = [0.25, 0.95];    % Zielband; je Struktur wird ein Ziel daraus gezogen

E_MOD = 1.0e3; NU = 0.3; D_THICK = 1.0;
MATNAME = 'NeoHooke';

allCoords = zeros(4, 2, 0);
allUe     = zeros(8, 0);
allSid    = zeros(1, 0);

loadNames = {'Querlast', 'Schub', 'Eigengew.', 'Zug (eing.)', 'Druck (eing.)'};

fprintf('=== Newton-Trajektorien Neo-Hooke (%d Strukturen, %d Val) ===\n\n', ...
    N_STRUCT, N_VAL_STRUCT);
fprintf('  Nr | Geometrie     | Verz. |  NEL | Last          | Skala    | ||lnU|| | J min..max  | Zustaende\n');
fprintf('  ---|---------------|-------|------|---------------|----------|---------|-------------|----------\n');

for k = 1:N_STRUCT

    % --------------------------------------------------------------------
    % Zufaellige Struktur (NICHT die Benchmark-Strukturen)
    % --------------------------------------------------------------------
    while true
        Lx = 1.0 + 7.0 * rand();
        Ly = 1.0 + 5.0 * rand();
        if ~(abs(Lx - 4) < 0.5 && abs(Ly - 2) < 0.3), break; end
    end
    nelTarget = 100 + round(300 * rand());
    nx = max(2, round(sqrt(nelTarget * Lx / Ly)));
    ny = max(2, round(sqrt(nelTarget * Ly / Lx)));
    dist = MAX_DIST * rand();
    loadType = randi(5);

    S = build_random_structure(Lx, Ly, nx, ny, dist, loadType, ...
                               [E_MOD, NU, D_THICK, 0]);

    % --------------------------------------------------------------------
    % Lastniveau einregeln: max ||ln U|| auf ein zufaelliges Ziel im Band,
    % J in der Huelle. Behalten wird der beste Lauf innerhalb der Huelle.
    % --------------------------------------------------------------------
    hTarget = H_TARGET(1) + diff(H_TARGET) * rand();
    if loadType >= 4
        scale = 0.3;                     % Anfangsdehnung (Weg)
    else
        scale = 5.0;                     % Anfangslast
    end
    bestScale = NaN;  bestErr = Inf;
    for trial = 1:10
        model = make_model(S, scale, MATNAME);
        [U, ok] = run_newton_capture(model, N_STEPS, MAX_ITER, 0);
        if ~ok
            scale = scale * 0.6;
            continue;
        end
        [hU, jmin, jmax] = hull_state(model, U);
        if ~(hU <= H_MAX && jmin >= J_MIN && jmax <= J_MAX)
            scale = scale * 0.7;
            continue;
        end
        err = abs(hU / hTarget - 1);
        if err < bestErr
            bestErr = err;  bestScale = scale;
        end
        if err < 0.1
            break;
        end
        scale = scale * min(3.0, max(0.4, (hTarget / max(hU, 1e-9))^1.2));
    end
    if isnan(bestScale)
        fprintf('  %2d | verworfen (keine Loesung in der Huelle)\n', k);
        continue;
    end
    scale = bestScale;

    % --------------------------------------------------------------------
    % Lauf mit Aufzeichnung
    % --------------------------------------------------------------------
    model = make_model(S, scale, MATNAME);
    [U, ok, cap] = run_newton_capture(model, N_STEPS, MAX_ITER, SAMPLE_FRAC);
    if ~ok || isempty(cap.Ue)
        fprintf('  %2d | verworfen (Aufzeichnungslauf)\n', k);
        continue;
    end
    [hU, jmin, jmax] = hull_state(model, U);

    allCoords = cat(3, allCoords, cap.coords);
    allUe     = [allUe, cap.Ue];                                  %#ok<AGROW>
    allSid    = [allSid, k * ones(1, size(cap.Ue, 2))];           %#ok<AGROW>

    fprintf('  %2d | %5.2f x %5.2f | %4.2f  | %4d | %-13s | %8.3f | %7.3f | %4.2f..%4.2f | %8d\n', ...
        k, Lx, Ly, dist, model.info.NEL, loadNames{loadType}, scale, hU, jmin, jmax, ...
        size(cap.Ue, 2));
end

is_val = double(allSid > (N_STRUCT - N_VAL_STRUCT));

coords    = allCoords;
Ue        = allUe;
struct_id = allSid;

outFile = fullfile(thisDir, 'newton_traj_states_neohooke.mat');
save(outFile, 'coords', 'Ue', 'struct_id', 'is_val', '-v7');

fprintf('\nGesamt: %d Zustaende (%d aus Val-Strukturen)\n', numel(allSid), sum(is_val));
fprintf('Datei:  %s\n', outFile);


% ========================================================================
% Strukturaufbau
% ========================================================================
function S = build_random_structure(Lx, Ly, nx, ny, dist, loadType, matcard)
%BUILD_RANDOM_STRUCTURE Verzerrte Rechteckscheibe mit Lagerung und Last.
%   loadType 1-3: kraftgesteuert (Last wird mit scale multipliziert)
%   loadType 4-5: weggesteuert, beidseitig eingespannt (scale = Dehnung)

[coord, elem, ~, mat] = create_model_data_rectangle( ...
    Lx, Ly, nx, ny, [0 0 0 0], [0 0 0 0], matcard);

tol   = 1e-9 * max(Lx, Ly);
inner = coord(:,1) > tol & coord(:,1) < Lx - tol & coord(:,2) > tol & coord(:,2) < Ly - tol;
coord(inner,1) = coord(inner,1) + dist * (Lx/nx) * (2*rand(nnz(inner),1) - 1);
coord(inner,2) = coord(inner,2) + dist * (Ly/ny) * (2*rand(nnz(inner),1) - 1);

left   = find(abs(coord(:,1))      < tol);
right  = find(abs(coord(:,1) - Lx) < tol);
bottom = find(abs(coord(:,2))      < tol);
top    = find(abs(coord(:,2) - Ly) < tol);

fix2 = @(n) [n(:), ones(numel(n),1), zeros(numel(n),1); n(:), 2*ones(numel(n),1), zeros(numel(n),1)];

S.fnode = zeros(0, 3);
S.fvol  = [0 0];
S.dispDof = zeros(0, 1);          % Zeilen in bcond mit skalierter Vorgabe
switch loadType
    case 1      % Kragscheibe mit Querlast
        S.bcond = fix2(left);
        S.fnode = [right, 2*ones(numel(right),1), line_load_nodes(coord(right,2), -1/Ly)];
    case 2      % Schub an der Oberkante
        S.bcond = fix2(bottom);
        S.fnode = [top, ones(numel(top),1), line_load_nodes(coord(top,1), 1/Lx)];
    case 3      % Eigengewicht, links eingespannt
        S.bcond = fix2(left);
        S.fvol  = [0 -1];
    otherwise   % beidseitig eingespannt, rechter Rand weggesteuert
        sgn = 1;
        if loadType == 5, sgn = -1; end
        nr = numel(right);
        S.bcond = [fix2(left);
                   right, ones(nr,1),   sgn * Lx * ones(nr,1);
                   right, 2*ones(nr,1), zeros(nr,1)];
        S.dispDof = size(fix2(left), 1) + (1:nr).';
end
S.coord = coord;  S.elem = elem;  S.mat = mat;
S.loadType = loadType;
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
vals = zeros(n,1);
vals(order) = q * trib;
end


function model = make_model(S, scale, MATNAME)
%MAKE_MODEL Neo-Hooke-Modell (analytisches Backend), Last bzw. Weg skaliert.
setup = init_setup;
setup.analysis.nl        = true;
setup.element.type       = 'quad4';
setup.element.backend    = 'matlab';
setup.material.name      = MATNAME;
setup.material.condition = 'planeStrain';
setup.solver.verbose     = false;

bcond = S.bcond;
fnode = S.fnode;
fvol  = S.fvol;
if S.loadType >= 4
    bcond(S.dispDof, 3) = bcond(S.dispDof, 3) * scale;
else
    if ~isempty(fnode), fnode(:,3) = fnode(:,3) * scale; end
    fvol = fvol * scale;
end
model = init_model(S.coord, S.elem, S.mat, bcond, fnode, fvol, [], setup);

% Kraftskala fuer das relative Konvergenzkriterium
if S.loadType >= 4
    model.fscale = model.mat(1,1) * max(abs(bcond(:,3))) ;
else
    [~, Fext0] = assemble(model, zeros(model.info.NDOF, 1));
    model.fscale = norm(Fext0(model.dofs.free));
end
end


% ========================================================================
% Newton mit Aufzeichnung und Tangenten-Praediktor
% ========================================================================
function [U, ok, cap] = run_newton_capture(model, nSteps, maxIter, sampleFrac)
%RUN_NEWTON_CAPTURE Newton je Lastschritt (Logik aus newton.m) mit Aufzeichnung.
%   Vor jedem Schritt verschiebt ein Tangenten-Praediktor die freien Knoten
%   passend zum Zuwachs der Randverschiebung -- ohne ihn klappen am
%   weggesteuerten Rand Elemente um, bevor Newton ueberhaupt startet.

NEL   = model.info.NEL;
DOF   = model.info.DOF;
free  = model.dofs.free;
fixed = model.dofs.fixed;
vals  = model.dofs.fixedValues;
tolR  = 1e-7 * max(model.fscale, eps);
U     = zeros(model.info.NDOF, 1);
ok    = true;

nSample = max(1, round(sampleFrac * NEL));
nMax    = nSample * nSteps * (maxIter + 1);
capC    = zeros(4, 2, nMax);
capU    = zeros(8, nMax);
nCap    = 0;
cap.coords = zeros(4, 2, 0);
cap.Ue     = zeros(8, 0);

lamOld = 0;
for step = 1:nSteps
    lam = step / nSteps;

    % Praediktor
    dUb = (lam - lamOld) * vals;
    if any(dUb)
        K = assemble(model, U);
        U(fixed) = U(fixed) + dUb;
        U(free)  = U(free) - K(free, free) \ (K(free, fixed) * dUb);
    end
    lamOld = lam;

    converged = false;
    for it = 0:maxIter
        try
            [K, Fext0, Fint] = assemble(model, U);
        catch
            ok = false;  return;
        end
        R = Fint - lam * Fext0;
        if ~all(isfinite(R)) || ~isreal(R)
            ok = false;  return;
        end

        if sampleFrac > 0
            pick = randperm(NEL, nSample);
            for jj = 1:nSample
                e = pick(jj);
                nCap = nCap + 1;
                capC(:, :, nCap) = model.coord(model.elem(e, :), :);
                capU(:, nCap)    = U(get_element_dofs(e, model.elem, DOF));
            end
        end

        if it > 0 && norm(R(free)) <= tolR
            converged = true;
            break;
        end
        if it == maxIter
            break;
        end
        U(free) = U(free) - K(free, free) \ R(free);
        if ~all(isfinite(U))
            ok = false;  return;
        end
    end
    if ~converged
        ok = false;  return;
    end
end

cap.coords = capC(:, :, 1:nCap);
cap.Ue     = capU(:, 1:nCap);
end


function [hmax, jmin, jmax] = hull_state(model, U)
%HULL_STATE max ||ln U||, min J und max J ueber alle Elemente und Gausspunkte.
hmax = 0;  jmin = Inf;  jmax = -Inf;
DOF = model.info.DOF;
for e = 1:model.info.NEL
    coord_e = model.coord(model.elem(e, :), :);
    Un      = reshape(U(get_element_dofs(e, model.elem, DOF)), 2, 4).';
    for g = 1:size(model.element.gp, 1)
        [~, dh, ~] = shape_quad4(coord_e, model.element.gp(g, :));
        Fdef = eye(2) + Un.' * dh;
        J    = det(Fdef);
        jmin = min(jmin, J);  jmax = max(jmax, J);
        if J <= 0
            hmax = Inf;  return;
        end
        hmax = max(hmax, norm(0.5 * log(eig(Fdef.' * Fdef))));
    end
end
end
