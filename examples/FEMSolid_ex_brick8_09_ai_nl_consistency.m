%FEMSOLID_EX_BRICK8_09_AI_NL_CONSISTENCY Verifikation des KI-brick8 (Gates d, e).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Prueft das nichtlineare KI-Element element_brick8_nl_ai (modale
%   Metrik-Kette + Voll-Energie-Netz) OHNE Solver -- 3D-Gegenstueck zu
%   FEMSolid_ex_quad4_09_ai_nl_consistency.m:
%
%   Gate d  MATLAB-Kern (dlfe_gram_energy) gegen die fp64-Oracle-Vektoren
%           aus Python (test_C/test_U -> test_W/F/K)            <= 1e-10
%   Gate e  20 zufaellige PHYSISCHE Hexaeder (frei gedreht, skaliert,
%           verschoben, E = 1000):
%             e1  Finte vs. zentrale Differenzen von W_phys      <= 1e-6
%             e2  Ke    vs. zentrale Differenzen von Finte       <= 1e-6
%             e3  Finte = 0 bei Starrkoerperbewegung             <= 1e-12
%             e4  Ke(u = 0) vs. lineares Element (gelernt)       < 1 %
%   Zusaetzlich (Info): Ke/Finte vs. analytisches element_brick8_nl an
%   den Zufallszustaenden (Netzgenauigkeit im physischen Rahmen).
%
% PREREQUISITE
%   training/brick8/train_brick8_nl_W_network.py
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

close all; clear all; clc; %#ok<CLALL>
fprintf('=== brick8 NL-KI: Konsistenz- und Ketten-Verifikation (Gates d, e) ===\n\n');
rng(4243);

MATNAME = 'StVenant';
E_MOD = 1000;  NU = 0.3;
N_ELEM = 20;  FD_STEP = 1e-6;

netFile = brick8_nl_ai_network_file(MATNAME);
NET  = load(netFile);
NETG = dlfe_load_network(netFile, 8, 3);
fprintf('Netz: %s | %s | h%d d%d | %s\n\n', netFile, strtrim(char(NET.state_form)), ...
    round(NET.hidden), round(NET.depth), strtrim(char(NET.timestamp)));

% ------------------------------------------------------------------------
% Gate d
% ------------------------------------------------------------------------
M = zeros(24);  M(triu(true(24))) = 1:300;  M = M + triu(M,1).';
recon = M(:);
nT = size(NET.test_C, 1);
eW = zeros(nT,1); eF = eW; eK = eW;
floorW = 0.01 * sqrt(mean(NET.test_W(:).^2));
floorF = 0.01 * sqrt(mean(sum(NET.test_F.^2, 2)));
for i = 1:nT
    chat = NET.test_C(i,:).';
    [Wm, Fm, Km] = dlfe_gram_energy(NETG, chat, reshape(chat, 3, 8).', ...
                                    reshape(NET.test_U(i,:), 3, 8).');
    eW(i) = abs(Wm - NET.test_W(i)) / max(abs(NET.test_W(i)), floorW);
    eF(i) = norm(Fm - NET.test_F(i,:).') / max(norm(NET.test_F(i,:)), floorF);
    Kr = reshape(NET.test_K(i, recon), 24, 24);
    eK(i) = norm(Km - Kr, 'fro') / norm(Kr, 'fro');
end
okD = max([eW; eF; eK]) <= 1e-10;
fprintf('--- Gate d: MATLAB vs. Python-Oracle ---\n  W %.2e | F %.2e | K %.2e  -> %s\n\n', ...
    max(eW), max(eF), max(eK), verdict(okD));

% ------------------------------------------------------------------------
% Gate e
% ------------------------------------------------------------------------
mat_e = [E_MOD, NU, NaN, 0];
g8 = gauss_library('brick8', 'default');
e1 = zeros(N_ELEM,1); e2 = e1; e3 = e1; e4 = e1; eKa = e1; eFa = e1;
for k = 1:N_ELEM
    X  = random_hex();
    Ue = random_state(X);
    [~, Fi, Ke] = brick8_nl_ai_energy(X, mat_e, Ue, MATNAME);
    gFD = zeros(24,1);  JFD = zeros(24);
    for j = 1:24
        up = Ue; up(j) = up(j) + FD_STEP;
        um = Ue; um(j) = um(j) - FD_STEP;
        [Wp, Fp] = brick8_nl_ai_energy(X, mat_e, up, MATNAME);
        [Wm, Fm] = brick8_nl_ai_energy(X, mat_e, um, MATNAME);
        gFD(j) = (Wp - Wm) / (2*FD_STEP);
        JFD(:,j) = (Fp - Fm) / (2*FD_STEP);
    end
    e1(k) = norm(gFD - Fi) / norm(Fi);
    e2(k) = norm(JFD - Ke, 'fro') / norm(Ke, 'fro');

    R = rotmat(randn(3,1), 2*pi*rand());
    Ur = (X * R.' + 3*randn(1,3)) - X;
    [~, Fr, Kr] = brick8_nl_ai_energy(X, mat_e, reshape(Ur.', [], 1), MATNAME);
    e3(k) = norm(Fr) / norm(Kr, 'fro');

    [~, ~, K0] = brick8_nl_ai_energy(X, mat_e, zeros(24,1), MATNAME);
    Kl = element_brick8_lin(X, mat_e, zeros(3,1), 0, zeros(24,1), [], g8.gp, g8.w, 'Hooke', '3D', struct());
    e4(k) = norm(K0 - Kl, 'fro') / norm(Kl, 'fro');

    [Ka, ~, ~, Fa] = element_brick8_nl(X, mat_e, zeros(3,1), 0, Ue, [], g8.gp, g8.w, MATNAME, '3D', struct());
    eKa(k) = norm(Ke - Ka, 'fro') / norm(Ka, 'fro');
    eFa(k) = norm(Fi - Fa) / norm(Fa);
end
ok1 = max(e1) <= 1e-6;  ok2 = max(e2) <= 1e-6;  ok3 = max(e3) <= 1e-12;  ok4 = max(e4) < 0.01;
fprintf('--- Gate e: physische Hexaeder (E = %.0f, frei gedreht/skaliert) ---\n', E_MOD);
fprintf('  e1  Finte vs FD(W_phys)    : max %.3e  (<= 1e-6)   %s\n', max(e1), verdict(ok1));
fprintf('  e2  Ke    vs FD(Finte)     : max %.3e  (<= 1e-6)   %s\n', max(e2), verdict(ok2));
fprintf('  e3  Finte bei Starrkoerper : max %.3e  (<= 1e-12)  %s\n', max(e3), verdict(ok3));
fprintf('  e4  Ke(u=0) vs linear      : max %.3f %% | Mittel %.3f %% (max < 1 %%)  %s\n', ...
    100*max(e4), 100*mean(e4), verdict(ok4));
fprintf('  Info: vs. analytisch  Finte max %.3f %% / Mittel %.3f %% | Ke max %.3f %% / Mittel %.3f %%\n', ...
    100*max(eFa), 100*mean(eFa), 100*max(eKa), 100*mean(eKa));

allOK = okD && ok1 && ok2 && ok3 && ok4;
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
%RANDOM_HEX Wohlgestellter verzerrter Hexaeder (innerhalb der Trainingshuelle).
base = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1] / 2;
a = 1/sqrt(3);
gp = [-a -a -a; a -a -a; a a -a; -a a -a; -a -a a; a -a a; a a a; -a a a];
for trial = 1:500
    S = diag(exp(log(0.5) + (log(2.0) - log(0.5)) * rand(1,3)));
    G = eye(3) + 0.3 * (rand(3) - 0.5) .* (1 - eye(3));
    X = base * S * G.' + 0.05 * (rand(8,3) - 0.5);
    dJ = zeros(8,1);
    for g = 1:8
        [~, ~, dJ(g)] = shape_brick8(X, gp(g,:));
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
