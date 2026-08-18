%FEMSOLID_EX_QUAD4_08_AI_NL_CHECK Isolierter Element-Check (ohne Solver).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Vergleicht das nichtlineare KI-Element (element_quad4_nl_ai,
%   Residual-Energie-Netz mit K0-Split) DIREKT mit dem analytischen Element
%   (element_quad4_nl) -- an EINEM Element, ohne globalen Newton-Solver.
%
%   Zweck: trennt "Element/Netz korrekt eingebunden" von "Benchmark
%   konvergiert nicht". Der Verschiebungszustand wird von 0 bis weit ueber
%   die Trainings-Huelle hinaus hochskaliert; pro Amplitude werden
%   ausgegeben:
%     - max ||E_green|| und max lokale Rotation ueber die Gausspunkte
%     - ob der Zustand INNERHALB der Trainings-Huelle liegt
%       (||E_green|| <= E_MAX, Rotation <= ROT_MAX -- Metadaten des Netzes)
%     - relativer Frobenius-Fehler von Ke und Finte (KI vs. analytisch)
%
%   Interpretation:
%     * amp = 0 (Referenzzustand, dicht trainiert): Fehler MUSS klein sein.
%       Ist er hier schon gross -> Netz/Einbindung defekt (nicht die Huelle).
%     * Fehler klein in der Huelle, gross ausserhalb -> Element ist korrekt,
%       die Benchmark-Lasten liegen ausserhalb der Huelle (Lasten senken oder
%       Huelle E_MAX/ROT_MAX im Training vergroessern).
%
%   Hinweis: Die Rotation ist seit der Ko-Rotation KEINE Huellengrenze mehr
%   (state_rot_max_deg = 180); massgeblich ist allein ||E_green|| <= E_MAX.
%
% PREREQUISITE
%   Run training/quad4/train_quad4_nl_W_network.py first to generate
%   quad4_nl_W_network.mat.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-08-18
% ------------------------------------------------------------------------

% "clear all": das KI-Element haelt das Netz in PERSISTENT-Variablen, die ein
% einfaches "clear" nicht loescht (sonst Diagnose gegen veraltete Gewichte).
close all; clear all; clc; %#ok<CLALL>

E_MOD = 1000; NU = 0.3; D = 1.0;
MATNAME = 'StVenant'; MATCOND = 'planeStrain';
mat_e = [E_MOD NU D 0];
b0 = [0; 0];

g  = gauss_library('quad4', 'default');   % 2x2
gp = g.gp;  w = g.w;
opts = struct();

% Trainings-Huelle aus den Netz-Metadaten lesen (Fallback: Planwerte).
netFile = fullfile(fileparts(which('element_quad4_nl_ai')), 'quad4_nl_W_network.mat');
E_MAX = 0.2; ROT_MAX = 180;
if exist(netFile,'file')
    nd = load(netFile);
    if isfield(nd,'state_E_max'),       E_MAX   = nd.state_E_max;       end
    if isfield(nd,'state_Egreen_max'),  E_MAX   = nd.state_Egreen_max;  end
    if isfield(nd,'state_rot_max_deg'), ROT_MAX = nd.state_rot_max_deg; end
end

% Ein moderat verzerrtes Element (innerhalb der Geometrie-Huelle).
coord_e = [0.00 0.00; 1.00 0.10; 1.20 1.00; 0.15 0.90];

% Verschiebungs-"Form": affiner Anteil (Scherung/Streckung) + nicht-affin.
G = [0.10 0.08; -0.05 0.12];
Ubase_nodes = coord_e * G.' + [0 0; 0.03 0; 0.05 0.04; 0 0.02];
Ubase = reshape(Ubase_nodes.', [], 1);

fprintf('=== nl KI-Element Check (Ke & Finte vs. analytisch, ohne Solver) ===\n');
fprintf('Material %s, %s, E = %.0f, nu = %.2f\n', MATNAME, MATCOND, E_MOD, NU);
fprintf('Trainings-Huelle: ||E_green|| <= %.2f, Rotation <= %.0f Grad\n\n', E_MAX, ROT_MAX);

% KeErr in mehreren Matrixnormen: Frobenius (wie bisher), Spektral-2 (groesster
% Singulaerwert -> beschraenkt den rel. Fehler in jeder Verformungsmode),
% 1-Norm (max. Spaltenbetragssumme; = inf-Norm, da Ke symmetrisch) und MaxAbs
% (groesster Einzeleintrags-Fehler). So sieht man, ob der Netzfehler
% gleichverteilt (Frob ~ MaxAbs) oder in einer Mode konzentriert ist (Spektral hoch).
fprintf([' amp  | max||Eg|| | maxRot | inHull | KeErr[%%]  Frob | Spek-2 |  1-Norm |  MaxAbs | FinteErr %%\n']);
fprintf(['------|-----------|--------|--------|----------------|--------|---------|---------|-----------\n']);

amps = [0 0.1 0.25 0.5 0.75 1.0 1.5 2.0];
for a = amps
    Ue = a * Ubase;
    [mE, mR] = state_measures(coord_e, Ue, gp);
    inhull = (mE <= E_MAX) && (mR <= ROT_MAX);

    [Ka,~,~,Fa] = element_quad4_nl(   coord_e, mat_e, b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
    [Kk,~,~,Fk] = element_quad4_nl_ai(coord_e, mat_e, b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);

    Dk    = Kk - Ka;
    keFro = norm(Dk,'fro') / max(norm(Ka,'fro'), eps) * 100;
    keSpk = norm(Dk,2)     / max(norm(Ka,2),     eps) * 100;
    ke1   = norm(Dk,1)     / max(norm(Ka,1),     eps) * 100;
    keMax = max(abs(Dk(:)))/ max(max(abs(Ka(:))),eps) * 100;
    fErr  = norm(Fk - Fa)  / max(norm(Fa),       eps) * 100;

    hullStr = 'nein';
    if inhull, hullStr = 'JA'; end
    fprintf(' %4.2f | %9.4f | %6.1f | %6s | %13.2f | %6.2f | %7.2f | %7.2f | %9.2f\n', ...
        a, mE, mR, hullStr, keFro, keSpk, ke1, keMax, fErr);
end

% ------------------------------------------------------------------------
% E/d-Skalierungscheck: Ke ~ E*d, Finte ~ E*d bei identischem Zustand.
%   Prueft die Faktorisierung im KI-Element (unabhaengig von der Netzguete).
% ------------------------------------------------------------------------
fprintf('\nE/d-Skalierungscheck (amp = 0.25, KI-Element):\n');
Ue = 0.25 * Ubase;
[K1,~,~,F1] = element_quad4_nl_ai(coord_e, [1    NU 1 0], b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
[K2,~,~,F2] = element_quad4_nl_ai(coord_e, [1000 NU 2 0], b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
% Erwartung: K2 = 2000 * K1, F2 = 2000 * F1  (E*d: 1000*2 vs 1*1)
fprintf('  Ke:    ||K2 - 2000*K1|| / ||K2||    = %.3e  (soll ~0)\n', ...
    norm(K2 - 2000*K1,'fro')/max(norm(K2,'fro'),eps));
fprintf('  Finte: ||F2 - 2000*F1|| / ||F2||    = %.3e  (soll ~0)\n', ...
    norm(F2 - 2000*F1)/max(norm(F2),eps));

fprintf('\nHinweis: amp=0 gross => Netz/Einbindung defekt. Klein in Huelle, gross\n');
fprintf('         ausserhalb => Element ok, Benchmark-Lasten ausserhalb der Huelle.\n');

% ------------------------------------------------------------------------
function [mE, mR] = state_measures(coord_e, Ue, gp)
%STATE_MEASURES max ||E_green||_F und max lokale Rotation [Grad] ueber die GP.
    mE = 0; mR = 0;
    for i = 1:size(gp,1)
        [~, dh_dx, ~] = shape_quad4(coord_e, gp(i,:));
        gradU = compute_gradU(dh_dx, Ue, 2, 2);
        F  = eye(2) + gradU;
        Eg = 0.5 * (F.'*F - eye(2));
        mE = max(mE, norm(Eg, 'fro'));
        th = abs(atan2d(F(2,1) - F(1,2), F(1,1) + F(2,2)));
        mR = max(mR, th);
    end
end
