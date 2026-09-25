function [Ke, Fbe, Fte, Finte, history] = element_brick8_nl_ai(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_BRICK8_NL_AI Nichtlineares brick8-Element: Voll-Energie-Netz.
% ------------------------------------------------------------------------
% DESCRIPTION
%   "Deep Learned Finite Elements" fuer das trilineare Hexaeder in der
%   NICHTLINEAREN Analyse (Total Lagrange, St.-Venant-Kirchhoff). 3D-
%   Uebertragung von element_quad4_nl_ai.m: das Element ersetzt die
%   Gauss-Schleife durch ein SKALARES Energienetz,
%
%     Finte = dW/dUe        Ke = d2W/dUe2 = dFinte/dUe
%
%   Exakte Struktur (nicht gelernt):
%     - Konsistenz Ke = dFinte/dUe und Symmetrie (Potentialform)
%     - Kraeftegleichgewicht und Starrkoerper-Nullraum (Metrik-Eingang)
%     - Finte = 0 bei jeder Starrkoerperbewegung in Maschinengenauigkeit
%       (D = 0 exakt, Subtraktionsform)
%     - Groessen-, Translations-, Rotationsinvarianz (Kanonisierung +
%       rotationsinvarianter modaler Metrik-Eingang)
%   Faktorisierung: W = E*Lc^3*What, Finte = E*Lc^2*(...), Ke = E*Lc*(...)
%   nu = 0.3, 3D und das Materialgesetz sind eintrainiert und werden HART
%   geprueft; E ist frei.
%
%   Volumenlast Fbe analytisch (Fte = 0, wie im analytischen nl-Element).
%
% PREREQUISITE
%   training/brick8/train_brick8_nl_W_network.py
%
% INPUT / OUTPUT
%   Identisch zu element_brick8_nl.m.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

persistent cfg ood_warned

if isempty(ood_warned)
    ood_warned = false;
end

Fbe = zeros(24, 1);
Fte = zeros(24, 1);

if isempty(cfg) || ~strcmp(cfg.MATNAME, MATNAME) || cfg.nu ~= mat_e(2) ...
        || ~strcmp(cfg.MATCOND, MATCOND)
    [~, Finte, Ke, dg, meta] = brick8_nl_ai_energy(coord_e, mat_e, Ue, MATNAME);
    [~, matKey] = brick8_nl_ai_network_file(MATNAME);
    [~, netKey] = brick8_nl_ai_network_file(meta.material);
    if ~strcmp(matKey, netKey)
        error('element_brick8_nl_ai:Material', ...
            'Das Netz wurde fuer Material "%s" trainiert, das Modell nutzt "%s".', ...
            meta.material, MATNAME);
    end
    if abs(mat_e(2) - meta.nu_train) > 1e-6
        error('element_brick8_nl_ai:Nu', ...
            ['Das Netz wurde fuer nu = %.4f trainiert, das Modell nutzt nu = %.4f. ' ...
             'nu ist NICHT herausfaktorisiert -- Ergebnisse waeren stumm falsch.'], ...
            meta.nu_train, mat_e(2));
    end
    if ~strcmpi(strtrim(char(meta.condition)), MATCOND)
        error('element_brick8_nl_ai:Condition', ...
            'Das Netz wurde fuer "%s" trainiert, das Modell nutzt "%s".', ...
            meta.condition, MATCOND);
    end
    cfg = struct('MATNAME', MATNAME, 'nu', mat_e(2), 'MATCOND', MATCOND, ...
                 'hencky', strcmpi(meta.state_measure, 'hencky_J'), ...
                 'E_max', meta.E_max, 'H_max', meta.H_max, ...
                 'J_min', meta.J_min, 'J_max', meta.J_max);
else
    [~, Finte, Ke, dg] = brick8_nl_ai_energy(coord_e, mat_e, Ue, MATNAME);
end

% OOD-Proxy am Elementmittelpunkt (die volle Huellenpruefung laeuft in
% FEMSolid_ex_brick8_08_ai_nl_check.m, nicht im Hot-Path).
if cfg.hencky
    outside = dg.hencky > cfg.H_max || dg.J < cfg.J_min || dg.J > cfg.J_max;
else
    outside = dg.maxE > cfg.E_max;
end
if outside && ~ood_warned
    warning('element_brick8_nl_ai:OutOfHull', ...
        ['Elementzustand ausserhalb der trainierten Huelle (||E_green|| = %.3f > %.3f). ' ...
         'Das Netz extrapoliert. Weitere Meldungen werden unterdrueckt.'], dg.maxE, cfg.E_max);
    ood_warned = true;
end

if any(b_e ~= 0)
    for i = 1:size(gp, 1)
        [h, ~, detJ] = shape_brick8(coord_e, gp(i, :));
        Fbe = Fbe + kron(h, eye(3)) * b_e * (detJ * w(i));
    end
end
end
