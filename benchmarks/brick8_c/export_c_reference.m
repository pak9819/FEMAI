function export_c_reference(netName, outFile)
%EXPORT_C_REFERENCE Referenzdaten fuer brick8_bench.c aus den MATLAB-Elementen.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Schreibt zwei Saetze (Eingaben + Ke/Finte aus element_brick8_nl und
%   element_brick8_nl_ai) im Binaerformat von brick8_bench.c:
%     1  200 zufaellige verzerrte, frei gedrehte Hexaeder, moderate Zustaende
%     2  Endzustand der Benchmark-Struktur 5 (Torsionsstab, 108 Elemente)
%
%   netName  Netzdatei in sourcecode/elements/brick8 (z.B.
%            'brick8_nl_W_network_h32d3.mat'); dieselbe Datei muss per
%            export_c_weights.py fuer das C-Programm exportiert werden.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-25
% ------------------------------------------------------------------------

setenv('BRICK8_NET_FILE', netName);
clear element_brick8_nl_ai brick8_nl_ai_energy          % persistente Netze verwerfen
rng(2026);
E = 1000;  nu = 0.3;  mat_e = [E nu NaN 0];
g = gauss_library('brick8', 'default');

fid = fopen(outFile, 'w', 'ieee-le');
fwrite(fid, int32(2), 'int32');

% --- Satz 1: Zufallselemente ---------------------------------------------
ne = 200;  X = zeros(24, ne);  U = zeros(24, ne);
for e = 1:ne
    Xe = random_hex();
    X(:, e) = reshape(Xe.', [], 1);
    U(:, e) = random_state(Xe);
end
write_set(fid, X, U, E, nu, mat_e, g);

% --- Satz 2: Torsionsstab (Benchmark-Struktur 5), Endzustand ---------------
[c, el, f, q] = create_model_data_box(6, 1, 1, 12, 3, 3, 0.2, 7);
bc = [];
for d = 1:3
    bc = [bc; f.x0, d*ones(numel(f.x0),1), zeros(numel(f.x0),1)]; %#ok<AGROW>
end
fn = torsion_load(c, q.x1, [0.5 0.5], 40 * 2.278);     % Lastniveau wie Benchmark 07
setup = init_setup;
setup.analysis.nl = true;  setup.analysis.numSteps = 5;
setup.element.type = 'brick8';  setup.element.backend = 'matlab';
setup.material.name = 'StVenant';  setup.material.condition = '3D';
setup.solver.verbose = false;  setup.solver.maxIter = 60;
model = init_model(c, el, repmat(mat_e, size(el,1), 1), bc, fn, [], [], setup);
evalc('[Ug, ~] = solve_FE(model);');
ne = size(el, 1);  X = zeros(24, ne);  U = zeros(24, ne);
for e = 1:ne
    X(:, e) = reshape(c(el(e,:), :).', [], 1);
    U(:, e) = Ug(get_element_dofs(e, el, 3));
end
write_set(fid, X, U, E, nu, mat_e, g);
fclose(fid);
fprintf('%s geschrieben (Netz %s)\n', outFile, netName);
end


function write_set(fid, X, U, E, nu, mat_e, g)
ne = size(X, 2);
Ka = zeros(576, ne);  Fa = zeros(24, ne);  Kk = Ka;  Fk = Fa;
for e = 1:ne
    Xe = reshape(X(:, e), 3, 8).';
    [K1, ~, ~, F1] = element_brick8_nl(Xe, mat_e, zeros(3,1), 0, U(:,e), [], g.gp, g.w, 'StVenant', '3D', struct());
    [K2, ~, ~, F2] = element_brick8_nl_ai(Xe, mat_e, zeros(3,1), 0, U(:,e), [], g.gp, g.w, 'StVenant', '3D', struct());
    Ka(:, e) = K1(:);  Fa(:, e) = F1;  Kk(:, e) = K2(:);  Fk(:, e) = F2;
end
fwrite(fid, int32([ne, 1]), 'int32');
fwrite(fid, [E, nu], 'double');
fwrite(fid, X, 'double');   fwrite(fid, U, 'double');
fwrite(fid, Ka, 'double');  fwrite(fid, Fa, 'double');
fwrite(fid, Kk, 'double');  fwrite(fid, Fk, 'double');
end


function fnode = torsion_load(coord, quads, yz0, q)
a = 1/sqrt(3);  gp = [-a -a; a -a; a a; -a a];
f = zeros(size(coord,1), 3);
for e = 1:size(quads, 1)
    Xq = coord(quads(e,:), :);
    for k = 1:4
        r = gp(k,1);  s = gp(k,2);
        h   = 0.25 * [(1-r)*(1-s); (1+r)*(1-s); (1+r)*(1+s); (1-r)*(1+s)];
        h_r = 0.25 * [-(1-s); (1-s); (1+s); -(1+s)];
        h_s = 0.25 * [-(1-r); -(1+r); (1+r); (1-r)];
        dA  = norm(cross(Xq.' * h_r, Xq.' * h_s));
        p   = Xq.' * h;
        f(quads(e,:), :) = f(quads(e,:), :) + h * (q * [0, -(p(3)-yz0(2)), p(2)-yz0(1)]) * dA;
    end
end
fnode = [];
for d = 2:3
    nz = find(f(:,d) ~= 0);
    fnode = [fnode; nz, d*ones(numel(nz),1), f(nz,d)]; %#ok<AGROW>
end
end


function R = rotmat(ax, th)
ax = ax / norm(ax);
Kx = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
R = eye(3) + sin(th)*Kx + (1-cos(th))*Kx*Kx;
end


function X = random_hex()
base = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1] / 2;
a = 1/sqrt(3);
gp = [-a -a -a; a -a -a; a a -a; -a a -a; -a -a a; a -a a; a a a; -a a a];
for trial = 1:500
    S = diag(exp(log(0.5) + (log(2.0) - log(0.5)) * rand(1,3)));
    G = eye(3) + 0.3 * (rand(3) - 0.5) .* (1 - eye(3));
    X = base * S * G.' + 0.05 * (rand(8,3) - 0.5);
    dJ = zeros(8,1);
    for k = 1:8
        [~, ~, dJ(k)] = shape_brick8(X, gp(k,:));
    end
    if min(dJ) > 0 && max(dJ) / min(dJ) < 3
        X = X * rotmat(randn(3,1), 2*pi*rand()).' * (0.3 + 3*rand()) + 5*randn(1,3);
        return
    end
end
error('keine gueltige Geometrie');
end


function Ue = random_state(X)
Lc = mean(sqrt(sum((X - mean(X,1)).^2, 2)));
G  = 0.05 * randn(3,3);
U  = (X - mean(X,1)) * G.' + 0.015 * Lc * randn(8,3);
R  = rotmat(randn(3,1), pi/3 * rand());
U  = (X + U) * R.' - X;
Ue = reshape(U.', [], 1);
end
