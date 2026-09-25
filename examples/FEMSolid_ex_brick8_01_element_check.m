%FEMSOLID_EX_BRICK8_01_ELEMENT_CHECK Verifikation der analytischen brick8-Elemente.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Phase 1 der brick8-Uebertragung: prueft element_brick8_lin/_nl ohne Netz.
%
%   p1  Ke_nl vs. zentrale Differenzen von Finte (StVenant und NeoHooke,
%       zufaellige verzerrte, frei gedrehte Hexaeder)            <= 1e-6
%   p2  Ke_nl symmetrisch                                       <= 1e-12
%   p3  Ke_nl(u = 0) = Ke_lin (StVenant)                        <= 1e-12
%   p4  Finte = 0 bei grosser Starrkoerperdrehung (StVenant)    <= 1e-10
%   p5  Patch-Test: linearer Verschiebungsansatz auf dem Rand eines
%       verzerrten 3x3x3-Netzes -> exakt reproduziert (linear)  <= 1e-10
%   p6  Schicht (uz = 0) == quad4 planeStrain, linear und nichtlinear
%       (gleiches Netz, gleiche Lasten)                         <= 1e-8
%   Info: Kragbalken (Endquerlast, linear) vs. Balkentheorie.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

close all; clear all; clc; %#ok<CLALL>
rng(11);
fprintf('=== brick8: Verifikation der analytischen Elemente (Phase 1) ===\n\n');

E = 1000; NU = 0.3;
mat_e = [E, NU, NaN, 0];
g8 = gauss_library('brick8', 'default');
gp = g8.gp;  w = g8.w;
b0 = zeros(3,1);
opts = struct();

% ------------------------------------------------------------------------
% p1 - p4: Einzelelemente
% ------------------------------------------------------------------------
N = 15;  h = 1e-6;
e1 = zeros(N,2); e2 = zeros(N,1); e3 = zeros(N,1); e4 = zeros(N,1);
mats = {'StVenant', 'NeoHooke'};
for k = 1:N
    X  = random_hex();
    Ue = random_state(X);
    for mi = 1:2
        [Ke, ~, ~, Fi] = element_brick8_nl(X, mat_e, b0, 0, Ue, [], gp, w, mats{mi}, '3D', opts);
        J = zeros(24);
        for j = 1:24
            up = Ue; up(j) = up(j) + h;
            um = Ue; um(j) = um(j) - h;
            [~,~,~,Fp] = element_brick8_nl(X, mat_e, b0, 0, up, [], gp, w, mats{mi}, '3D', opts);
            [~,~,~,Fm] = element_brick8_nl(X, mat_e, b0, 0, um, [], gp, w, mats{mi}, '3D', opts);
            J(:,j) = (Fp - Fm) / (2*h);
        end
        e1(k,mi) = norm(J - Ke, 'fro') / norm(Ke, 'fro');
        if mi == 1
            e2(k) = norm(Ke - Ke.', 'fro') / norm(Ke, 'fro');
        end
    end
    K0  = element_brick8_nl(X, mat_e, b0, 0, zeros(24,1), [], gp, w, 'StVenant', '3D', opts);
    Kl  = element_brick8_lin(X, mat_e, b0, 0, zeros(24,1), [], gp, w, 'Hooke', '3D', opts);
    e3(k) = norm(K0 - Kl, 'fro') / norm(Kl, 'fro');
    R  = rotmat(randn(3,1), pi * rand());
    Ur = (X * R.' + randn(1,3)) - X;
    [~,~,~,Fr] = element_brick8_nl(X, mat_e, b0, 0, reshape(Ur.', [], 1), [], gp, w, 'StVenant', '3D', opts);
    e4(k) = norm(Fr) / (norm(Kl, 'fro') * norm(Ur(:)));
end
ok1 = max(e1(:)) <= 1e-6;  ok2 = max(e2) <= 1e-12;
ok3 = max(e3) <= 1e-12;    ok4 = max(e4) <= 1e-10;
fprintf('p1 Ke vs FD(Finte) StVenant/NeoHooke : %.2e / %.2e  %s\n', max(e1(:,1)), max(e1(:,2)), verdict(ok1));
fprintf('p2 Ke symmetrisch                   : %.2e  %s\n', max(e2), verdict(ok2));
fprintf('p3 Ke_nl(0) = Ke_lin                : %.2e  %s\n', max(e3), verdict(ok3));
fprintf('p4 Finte(Starrkoerper) = 0          : %.2e  %s\n', max(e4), verdict(ok4));

% ------------------------------------------------------------------------
% p5: Patch-Test (linear, verzerrtes Netz, linearer Randansatz)
% ------------------------------------------------------------------------
[coord, elem, faces] = create_model_data_box(1, 1, 1, 3, 3, 3, 0.25, 3);
A = 1e-3 * randn(3,3);  c0 = 1e-3 * randn(1,3);
uex = coord * A.' + c0;
bnd = unique([faces.x0; faces.x1; faces.y0; faces.y1; faces.z0; faces.z1]);
bcond = [];
for d = 1:3
    bcond = [bcond; bnd, d*ones(numel(bnd),1), uex(bnd,d)]; %#ok<AGROW>
end
model = make_model(coord, elem, bcond, [], false, 'Hooke', 1);
[~, res] = evalc_solve(model);
U = reshape(res(end).U, 3, []).';
e5 = norm(U - uex, 'fro') / norm(uex, 'fro');
ok5 = e5 <= 1e-10;
fprintf('p5 Patch-Test (verzerrt, 3x3x3)     : %.2e  %s\n', e5, verdict(ok5));

% ------------------------------------------------------------------------
% p6: Schicht mit uz = 0 == quad4 planeStrain
% ------------------------------------------------------------------------
Lx = 4; Ly = 1; nx = 8; ny = 2;
[c3, el3, f3] = create_model_data_box(Lx, Ly, 1, nx, ny, 1);
[c2, el2] = create_model_data_rectangle(Lx, Ly, nx, ny, [0 0 0 0], [0 0 0 0], [E NU 1 0]);
left2 = find(abs(c2(:,1)) < 1e-9);  right2 = find(abs(c2(:,1) - Lx) < 1e-9);
bc2 = [left2, ones(numel(left2),1), zeros(numel(left2),1);
       left2, 2*ones(numel(left2),1), zeros(numel(left2),1)];
P = -3;                                   % Endquerlast gesamt (Dicke 1)
fn2 = [right2, 2*ones(numel(right2),1), P/numel(right2)*ones(numel(right2),1)];
nn2 = size(c2,1);
bc3 = [f3.x0, ones(numel(f3.x0),1), zeros(numel(f3.x0),1);
       f3.x0, 2*ones(numel(f3.x0),1), zeros(numel(f3.x0),1);
       (1:size(c3,1)).', 3*ones(size(c3,1),1), zeros(size(c3,1),1)];
fn3 = [fn2(:,1), fn2(:,2), fn2(:,3)/2;
       fn2(:,1) + nn2, fn2(:,2), fn2(:,3)/2];
e6 = zeros(1,2);
for nl = [false true]
    matn = 'Hooke'; if nl, matn = 'StVenant'; end
    m2 = make_model2(c2, el2, bc2, fn2, nl, matn, [E NU 1 0]);
    m3 = make_model(c3, el3, bc3, fn3, nl, matn, 4);
    [~, r2] = evalc_solve(m2);  [~, r3] = evalc_solve(m3);
    U2 = reshape(r2(end).U, 2, []).';
    U3 = reshape(r3(end).U, 3, []).';
    e6(nl+1) = max(norm(U3(1:nn2,1:2) - U2, 'fro'), norm(U3(nn2+1:end,1:2) - U2, 'fro')) ...
               / norm(U2, 'fro');
    if nl
        tipd = min(U2(:,2));
    end
end
ok6 = max(e6) <= 1e-8;
fprintf('p6 Schicht == quad4 planeStrain     : lin %.2e | nl %.2e  %s\n', e6(1), e6(2), verdict(ok6));
fprintf('   (nl Spitzenabsenkung %.4f)\n', tipd);

% ------------------------------------------------------------------------
% Info: Kragbalken vs. Balkentheorie (linear, 2x2x2 -> Schubversteifung)
% ------------------------------------------------------------------------
L = 10; b = 1; hh = 1;
for ne = [2 4]
    [cc, ee, ff, fq] = create_model_data_box(L, b, hh, 10*ne, ne, ne);
    bc = [];
    for d = 1:3
        bc = [bc; ff.x0, d*ones(numel(ff.x0),1), zeros(numel(ff.x0),1)]; %#ok<AGROW>
    end
    Pt = -1;
    fn = box_face_load(cc, fq.x1, 3, Pt / (b*hh));
    mm = make_model(cc, ee, bc, fn, false, 'Hooke', 1);
    [~, rr] = evalc_solve(mm);
    Ub = reshape(rr(end).U, 3, []).';
    wFE = mean(Ub(ff.x1, 3));
    I = b*hh^3/12;  G = E/(2*(1+NU));
    wTh = Pt*L^3/(3*E*I) + Pt*L/(5/6*G*b*hh);
    fprintf('Info Kragbalken %2dx%dx%d: w_FE/w_Theorie = %.4f\n', 10*ne, ne, ne, wFE/wTh);
end

allOK = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
fprintf('\n=== Gesamt: %s ===\n', verdict(allOK));


% ========================================================================
function s = verdict(ok)
if ok, s = 'GRUEN'; else, s = 'ROT'; end
end

function R = rotmat(ax, th)
ax = ax / norm(ax);
Kx = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
R = eye(3) + sin(th)*Kx + (1-cos(th))*Kx*Kx;
end

function X = random_hex()
base = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1] / 2;
for trial = 1:200
    S = diag(exp(log(0.4) + (log(2.5) - log(0.4)) * rand(1,3)));
    G = eye(3) + 0.3 * (rand(3) - 0.5);
    X = base * S * G.' + 0.08 * (rand(8,3) - 0.5);
    ok = true;
    for g = [-1 1]
        for q = [-1 1]
            for r = [-1 1]
                [~, ~, dJ] = shape_brick8(X, [g q r]);
                if dJ <= 0, ok = false; end
            end
        end
    end
    if ok
        X = X * rotmat(randn(3,1), 2*pi*rand()).' * (0.3 + 3*rand()) + 5*randn(1,3);
        return
    end
end
error('keine gueltige Geometrie');
end

function Ue = random_state(X)
Lc = mean(sqrt(sum((X - mean(X,1)).^2, 2)));
G  = 0.06 * randn(3,3);
U  = (X - mean(X,1)) * G.' + 0.02 * Lc * randn(8,3);
R  = rotmat(randn(3,1), pi/3 * rand());
U  = (X + U) * R.' - X;
Ue = reshape(U.', [], 1);
end

function model = make_model(coord, elem, bcond, fnode, nl, matname, nsteps)
setup = init_setup;
setup.element.type = 'brick8';
setup.element.backend = 'matlab';
setup.analysis.nl = nl;
setup.analysis.numSteps = nsteps;
setup.material.name = matname;
setup.material.condition = '3D';
setup.solver.verbose = false;
setup.solver.maxIter = 30;
mat = repmat([1000 0.3 NaN 0], size(elem,1), 1);
model = init_model(coord, elem, mat, bcond, fnode, [], [], setup);
end

function model = make_model2(coord, elem, bcond, fnode, nl, matname, matcard)
setup = init_setup;
setup.element.type = 'quad4';
setup.element.backend = 'matlab';
setup.analysis.nl = nl;
setup.analysis.numSteps = 4;
if ~nl, setup.analysis.numSteps = 1; end
setup.material.name = matname;
setup.material.condition = 'planeStrain';
setup.solver.verbose = false;
setup.solver.maxIter = 30;
mat = repmat(matcard, size(elem,1), 1);
model = init_model(coord, elem, mat, bcond, fnode, [], [], setup);
end

function [txt, res] = evalc_solve(model)
[txt, ~, res] = evalc_wrap(model);
end

function [txt, U, res] = evalc_wrap(model)
txt = evalc('[U, res] = solve_FE(model);');
end
