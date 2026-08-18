%FEMSOLID_EX_QUAD4_09_AI_NL_CONSISTENCY Verifikation des Residual-Energie-Elements.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Prueft das nichtlineare KI-Element (element_quad4_nl_ai, Residual-Energie
%   mit K0-Split) OHNE Solver -- Gates d und e der Verifikationsleiter:
%
%   Gate d  Netzauswertung in MATLAB gegen die im .mat mitgelieferten
%           Oracle-Testvektoren aus Python (test_C/test_Z -> test_W/F/K).
%           Trennt "MATLAB rechnet das Netz falsch" von "Netz ist ungenau".
%
%   Gate e  20 zufaellige PHYSISCHE Elemente (frei rotiert, skaliert,
%           verschoben, E = 1000, d = 2):
%             e1  Finte vs. zentrale Differenzen von W_phys
%             e2  Ke    vs. zentrale Differenzen von Finte  (globale
%                 Koordinaten -- validiert die komplette Kette inkl. d2theta)
%             e3  Finte = 0 bei reiner Starrkoerperbewegung (Translation +
%                 grosse Rotation) -- muss Maschinengenauigkeit sein
%             e4  Ke bei u -> 0 gegen das LINEARE Element (element_quad4_lin)
%                 -- prueft, ob K_NL(c,0) ~ 0 tatsaechlich erreicht wird
%                 (weiche Nebenbedingung, siehe Plan Phase 2)
%
%   e1/e2 pruefen die KONSISTENZ (Ke = dFinte/dUe), die beim Energie-Ansatz
%   strukturell gelten MUSS -- unabhaengig davon, wie gut das Netz trainiert
%   ist. Ein Fehler hier ist ein Implementierungsfehler, kein Lernfehler.
%
% PREREQUISITE
%   training/quad4/train_quad4_nl_W_network.py ausfuehren
%   (erzeugt quad4_nl_W_network.mat).
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-08-18
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

% "clear all": das KI-Element haelt das Netz in PERSISTENT-Variablen, die ein
% einfaches "clear" nicht loescht (sonst Vergleich gegen veraltete Gewichte).
close all; clear all; clc; %#ok<CLALL>

fprintf('=== quad4 NL-KI: Konsistenz- und Ketten-Verifikation (Gates d, e) ===\n\n');

rng(4242);

MATNAME = 'StVenant';
MATCOND = 'planeStrain';
NU      = 0.3;
E_MOD   = 1000.0;
D_THICK = 2.0;

N_ELEM  = 20;
FD_STEP = 1e-6;

netFile = fullfile(fileparts(which('element_quad4_nl_ai')), 'quad4_nl_W_network.mat');
if ~exist(netFile, 'file')
    error('quad4_nl_W_network.mat fehlt -- zuerst train_quad4_nl_W_network.py ausfuehren.');
end
NET = load(netFile);

fprintf('Netz: %s\n', strtrim(char(NET.model_form)));
fprintf('      GELU, %d Gewichtslagen, Hidden %d, Tiefe %d\n', ...
    round(NET.num_linear_layers), round(NET.hidden), round(NET.depth));
if isfield(NET, 'git_hash')
    fprintf('      git %s | %s\n', strtrim(char(NET.git_hash)), strtrim(char(NET.timestamp)));
end
fprintf('\n');

% ========================================================================
% Gate d: MATLAB-Netzauswertung vs. Python-Oracle-Vektoren
% ========================================================================
% Der Export erfolgt in fp64 (test_precision), damit hier wirklich die
% MATLAB-Rekurrenzen gegen PyTorch geprueft werden und nicht der
% fp32-Rundungsfehler von K0*z (Ausloeschung durch den Nullraum von K0).
GATE_D_TOL = 1e-10;
if ~isfield(NET, 'test_precision') || ~strcmpi(strtrim(char(NET.test_precision)), 'float64')
    GATE_D_TOL = 1e-5;      % aelterer fp32-Export
end
fprintf('--- Gate d: MATLAB vs. Python-Oracle (Schwelle %.0e rel) ---\n', GATE_D_TOL);

nTest = size(NET.test_C, 1);
eW = zeros(nTest,1);  eF = zeros(nTest,1);  eK = zeros(nTest,1);

% Floors gegen Division durch (fast) null: der Export waehlt zwar nur
% nicht-degenerierte Zustaende, aber W und F koennen einzeln klein werden.
% 1 % der RMS-Groesse der Testmenge -- misst dann absolut statt relativ.
floorW = 0.01 * sqrt(mean(NET.test_W(:).^2));
floorF = 0.01 * sqrt(mean(sum(NET.test_F.^2, 2)));

% Rekonstruktionstabelle triu(36) -> volle 8x8 (column-major, wie Python)
M = zeros(8);  M(triu(true(8))) = 1:36;  M = M + triu(M,1).';
recon = M(:);

for i = 1:nTest
    chat = NET.test_C(i,:).';
    z    = NET.test_Z(i,:).';

    % Das kanonische Modell wird DIREKT geprueft (ohne Kette) -- genau die
    % Groessen, die Python als Oracle exportiert hat.
    [Wm, Fm, Km] = quad4_nl_ai_model(chat, z);

    eW(i) = abs(Wm - NET.test_W(i)) / max(abs(NET.test_W(i)), floorW);
    eF(i) = norm(Fm - NET.test_F(i,:).') / max(norm(NET.test_F(i,:)), floorF);
    Kref  = reshape(NET.test_K(i, recon), 8, 8);
    eK(i) = norm(Km - Kref, 'fro') / max(norm(Kref, 'fro'), 1e-12);
end

fprintf('  W: max %.3e | F: max %.3e | K: max %.3e   -> %s\n\n', ...
    max(eW), max(eF), max(eK), verdict(max([eW; eF; eK]) <= GATE_D_TOL));

% ========================================================================
% Gate e: physische Elemente, FD-Konsistenz und Grenzfaelle
% ========================================================================
fprintf('--- Gate e: physische Elemente (E = %.0f, d = %.1f, frei rotiert/skaliert) ---\n', ...
    E_MOD, D_THICK);

mat_e = [E_MOD, NU, D_THICK, 0];
gp = [-1 -1; 1 -1; 1 1; -1 1] / sqrt(3);
w  = ones(4,1);

e1 = zeros(N_ELEM,1);   % Finte vs FD(W)
e2 = zeros(N_ELEM,1);   % Ke vs FD(Finte)
e3 = zeros(N_ELEM,1);   % Finte bei Starrkoerperbewegung (absolut, normiert)
e4 = zeros(N_ELEM,1);   % Ke(u->0) vs lineares Element

for k = 1:N_ELEM
    coord_e = random_element();
    Ue      = random_state(coord_e);

    % --- e1: Finte vs. zentrale Differenzen der Energie ------------------
    [~, Finte, Ke] = quad4_nl_ai_energy(coord_e, mat_e, Ue);
    gFD = zeros(8,1);
    for j = 1:8
        up = Ue; up(j) = up(j) + FD_STEP;
        um = Ue; um(j) = um(j) - FD_STEP;
        Wp = quad4_nl_ai_energy(coord_e, mat_e, up);
        Wm = quad4_nl_ai_energy(coord_e, mat_e, um);
        gFD(j) = (Wp - Wm) / (2*FD_STEP);
    end
    e1(k) = norm(gFD - Finte) / max(norm(Finte), 1e-12);

    % --- e2: Ke vs. zentrale Differenzen von Finte (globale Koordinaten) --
    JFD = zeros(8,8);
    for j = 1:8
        up = Ue; up(j) = up(j) + FD_STEP;
        um = Ue; um(j) = um(j) - FD_STEP;
        [~, Fp] = quad4_nl_ai_energy(coord_e, mat_e, up);
        [~, Fm] = quad4_nl_ai_energy(coord_e, mat_e, um);
        JFD(:,j) = (Fp - Fm) / (2*FD_STEP);
    end
    e2(k) = norm(JFD - Ke, 'fro') / max(norm(Ke, 'fro'), 1e-12);

    % --- e3: reine Starrkoerperbewegung -> Finte = 0 ----------------------
    th = 2*pi*rand();
    R  = [cos(th) -sin(th); sin(th) cos(th)];
    shift = 5*(rand(1,2) - 0.5);
    Urig  = (coord_e * R.' + shift) - coord_e;
    [~, Frig, Krig] = quad4_nl_ai_energy(coord_e, mat_e, reshape(Urig.', [], 1));
    e3(k) = norm(Frig) / max(norm(Krig, 'fro'), 1e-12);   % dimensionslos

    % --- e4: Ke bei u -> 0 gegen das LINEARE Element ----------------------
    [~, ~, K0ai] = quad4_nl_ai_energy(coord_e, mat_e, zeros(8,1));
    Klin = element_quad4_lin(coord_e, mat_e, [0;0], 0, zeros(8,1), [], gp, w, ...
                             'Hooke', MATCOND, struct());
    e4(k) = norm(K0ai - Klin, 'fro') / norm(Klin, 'fro');
end

fprintf('  e1  Finte vs FD(W_phys)   : max %.3e  (<= 1e-6)   %s\n', ...
    max(e1), verdict(max(e1) <= 1e-6));
fprintf('  e2  Ke    vs FD(Finte)    : max %.3e  (<= 1e-6)   %s\n', ...
    max(e2), verdict(max(e2) <= 1e-6));
fprintf('  e3  Finte bei Starrkoerper: max %.3e  (<= 1e-12)  %s\n', ...
    max(e3), verdict(max(e3) <= 1e-12));
fprintf('  e4  Ke(u->0) vs linear    : max %.3f %% | Mittel %.3f %% (max < 1 %%)  %s\n', ...
    100*max(e4), 100*mean(e4), verdict(max(e4) < 0.01));

fprintf('\n  Hinweis: e1/e2 pruefen die KONSISTENZ (strukturell garantiert),\n');
fprintf('           e4 die weiche Nebenbedingung K_NL(c,0) ~ 0 (gelernt).\n');

allOK = max([eW; eF; eK]) <= GATE_D_TOL && max(e1) <= 1e-6 && max(e2) <= 1e-6 ...
        && max(e3) <= 1e-12 && max(e4) < 0.01;
fprintf('\n=== Gesamt: %s ===\n', verdict(allOK));


% ========================================================================
% Hilfsfunktionen
% ========================================================================
function s = verdict(ok)
if ok, s = 'GRUEN'; else, s = 'ROT'; end
end


function coord_e = random_element()
%RANDOM_ELEMENT Zufaelliges, wohlgestelltes Viereck, frei platziert.
for trial = 1:200
    ax = exp(log(0.4) + (log(2.5)-log(0.4))*rand());
    H  = exp(log(0.4) + (log(2.5)-log(0.4))*rand());
    t  = exp(log(4.0) * rand() * sign(rand()-0.5));
    c  = [-ax/2 -H/2; ax/2 -H/2; ax*t/2 H/2; -ax*t/2 H/2];
    g1 = -0.4 + 0.8*rand();  g2 = -0.4 + 0.8*rand();
    c  = c * [1 g1; g2 1].';
    c  = c + (-0.08 + 0.16*rand(4,2));
    if polyarea(c(:,1), c(:,2)) <= 0, continue; end
    ang = interior_angles_local(c);
    if min(ang) < 20 || max(ang) > 160, continue; end
    % frei rotieren / skalieren / verschieben
    th = 2*pi*rand();  R = [cos(th) -sin(th); sin(th) cos(th)];
    coord_e = c * R.' * (0.3 + 4.7*rand()) + 10*(rand(1,2) - 0.5);
    return;
end
error('Keine gueltige Zufallsgeometrie gefunden.');
end


function ang = interior_angles_local(c)
ang = zeros(4,1);
for i = 1:4
    u = c(mod(i-2,4)+1,:) - c(i,:);
    v = c(mod(i,4)+1,:)   - c(i,:);
    ang(i) = acosd(max(-1, min(1, dot(u,v)/(norm(u)*norm(v)+1e-15))));
end
end


function Ue = random_state(coord_e)
%RANDOM_STATE Moderater Verschiebungszustand + zufaellige Starrkoerperrotation.
Lc = mean(sqrt(sum((coord_e - mean(coord_e,1)).^2, 2)));
G  = 0.06 * randn(2,2);
U  = (coord_e - mean(coord_e,1)) * G.' + 0.02 * Lc * randn(4,2);
th = (pi/3) * (2*rand() - 1);                      % bis 60 Grad Zustandsrotation
R  = [cos(th) -sin(th); sin(th) cos(th)];
U  = (coord_e + U) * R.' - coord_e;
Ue = reshape(U.', [], 1);
end
