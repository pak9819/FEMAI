function [Ke, Fbe, Fte, Finte, history] = element_quad4_nl_ai(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_QUAD4_NL_AI Nichtlineares quad4-Element: Voll-Energie-Netz.
% ------------------------------------------------------------------------
% DESCRIPTION
%   "Deep Learned Finite Elements" fuer das bilineare Viereckselement in der
%   NICHTLINEAREN Analyse (Total Lagrange; St.-Venant-Kirchhoff oder
%   Neo-Hooke, je ein eigenes Netz).
%
%   Das Element ersetzt die Gauss-Schleife des klassischen Elements durch ein
%   SKALARES Energiemodell auf der kanonischen Geometrie:
%
%     What(chat, z) = Wnet(chat, z)
%
%   Das Netz traegt die VOLLE Energie (quadratischer + nichtlinearer Anteil);
%   im Element gibt es KEINE numerische Energie-/Steifigkeitsberechnung mehr.
%   Beide Ausgabegroessen entstehen durch DIFFERENTIATION desselben
%   Potentials:
%
%     Finte = dW/dUe        Ke = d2W/dUe2 = dFinte/dUe
%
%   -> Ke ist PER KONSTRUKTION exakt die Jacobimatrix von Finte. Genau diese
%   Konsistenz fehlte der Zwei-Kopf-Variante (zwei unabhaengige Netzkoepfe
%   fuer Ke und Finte, gemessene Inkonsistenz ~7 %), die deshalb nur linear
%   mit Kontraktionsrate ~0.9 konvergierte und das Iterationslimit erreichte.
%
%   Die Tangente bei z = 0 (lineare Steifigkeit) ist GELERNT, nicht
%   analytisch -- ihre Genauigkeit wird in Gate e4
%   (FEMSolid_ex_quad4_09_ai_nl_consistency.m) gegen das lineare Element
%   geprueft.
%
%   Exakte Struktur (nicht gelernt):
%     - Konsistenz Ke = dFinte/dUe            (Potentialform)
%     - Symmetrie von Ke                      (Hessian eines Skalars)
%     - Kraeftegleichgewicht sum_i Finte_i = 0 und Translationsnullraum von
%       Ke                                    (Translationsprojektor P)
%     - Finte = 0 bei reiner Starrkoerperbewegung, in Maschinengenauigkeit
%       (Ko-Rotation liefert z = 0, Subtraktionsform liefert F(c,0) = 0)
%     - Groessen-, Translations- und Rotationsinvarianz sowie Objektivitaet
%       (Kanonisierung + Ko-Rotation, Ableitungen exakt mitgefuehrt)
%
%   Faktorisierungen (StVenant, exakt):
%     W     = E * d * Lc^2 * What
%     Finte = E * d * Lc   * Tc' * P * g_v
%     Ke    = E * d        * Tc' * P * K_v * P * Tc
%   nu = 0.3, planeStrain und das Materialgesetz sind FEST eintrainiert und
%   werden HART geprueft; E und Dicke d sind frei. Je Material gibt es ein
%   eigenes Netz (quad4_nl_ai_network_file): StVenant und NeoHookean1.
%
%   Volumenlast Fbe wird weiterhin analytisch berechnet (Fte = 0, wie im
%   analytischen nl-Element).
%
% PREREQUISITE
%   StVenant  : training/quad4/train_quad4_nl_W_network.py
%   NeoHooke  : training/quad4/train_quad4_nl_W_network_neohooke.py
%
% INPUT / OUTPUT
%   Identisch zu element_quad4_nl.m.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-17
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

persistent cfg ood_warned

if isempty(ood_warned)
    ood_warned = false;
end

DIM    = size(coord_e, 2);
DOF    = DIM;
NNEL   = size(coord_e, 1);
NDOFEL = NNEL * DOF;

Fbe = zeros(NDOFEL, 1);
Fte = zeros(NDOFEL, 1);

% ------------------------------------------------------------------------
% Konsistenzpruefung: bei JEDEM Aufruf gegen die zuletzt gepruefte
% Konfiguration (Material, nu, ebener Zustand). Frueher lief sie nur einmal
% je MATLAB-Sitzung -- ein spaeteres Modell mit anderem Material rechnete
% dann stumm mit dem falschen Netz. Der Vergleich kostet zwei String-
% Vergleiche; die volle Pruefung (mit Netz-Metadaten) laeuft nur, wenn sich
% die Konfiguration aendert.
% HART (error): alles, was stumm falsche Physik ergaebe -- jedes Netz ist
% materialgebunden (Material, nu, ebener Zustand sind eintrainiert).
% ------------------------------------------------------------------------
if isempty(cfg) || ~strcmp(cfg.MATNAME, MATNAME) || cfg.nu ~= mat_e(2) ...
        || ~strcmp(cfg.MATCOND, MATCOND)
    [~, Finte, Ke, dg, meta] = quad4_nl_ai_energy(coord_e, mat_e, Ue, MATNAME);

    [~, matKey] = quad4_nl_ai_network_file(MATNAME);
    [~, netKey] = quad4_nl_ai_network_file(meta.material);
    if ~strcmp(matKey, netKey)
        error('element_quad4_nl_ai:Material', ...
            ['Das Netz wurde fuer Material "%s" trainiert, das Modell nutzt "%s". ' ...
             'Das Netz ist materialgebunden -- Ergebnisse waeren stumm falsch.'], ...
            strtrim(char(meta.material)), MATNAME);
    end
    if abs(mat_e(2) - meta.nu_train) > 1e-6
        error('element_quad4_nl_ai:Nu', ...
            ['Das Netz wurde fuer nu = %.4f trainiert, das Modell nutzt nu = %.4f. ' ...
             'nu ist NICHT herausfaktorisiert -- Ergebnisse waeren stumm falsch.'], ...
            meta.nu_train, mat_e(2));
    end
    if ~strcmpi(strtrim(char(meta.condition)), MATCOND)
        error('element_quad4_nl_ai:Condition', ...
            ['Das Netz wurde fuer "%s" trainiert, das Modell nutzt "%s". ' ...
             'Der ebene Zustand ist eintrainiert -- Ergebnisse waeren stumm falsch.'], ...
            strtrim(char(meta.condition)), MATCOND);
    end

    cfg = struct('MATNAME', MATNAME, 'nu', mat_e(2), 'MATCOND', MATCOND, ...
                 'hencky', strcmpi(meta.state_measure, 'hencky_J'), ...
                 'E_max', meta.E_max, 'H_max', meta.H_max, ...
                 'J_min', meta.J_min, 'J_max', meta.J_max);
else
    % ------------------------------------------------------------------
    % Kern: Energie -> Finte, Ke (Kanonisierung, Ko-Rotation, Netz)
    % ------------------------------------------------------------------
    [~, Finte, Ke, dg] = quad4_nl_ai_energy(coord_e, mat_e, Ue, MATNAME);
end

% ------------------------------------------------------------------------
% OOD-Proxy am Elementmittelpunkt (der Verschiebungsgradient liegt fuer die
% Ko-Rotation ohnehin vor -> praktisch kostenlos). Das Mass kommt aus dem
% Netz: ||E_green|| (StVenant) bzw. Hencky-Dehnung ||ln U|| und J = det F
% (Neo-Hooke). Die volle 4-GP-Huellenpruefung bleibt bewusst DRAUSSEN aus
% dem Hot-Path; dafuer gibt es FEMSolid_ex_quad4_08_ai_nl_check.m.
% ------------------------------------------------------------------------
if cfg.hencky
    outside = dg.hencky > cfg.H_max || dg.J < cfg.J_min || dg.J > cfg.J_max;
else
    outside = dg.maxE > cfg.E_max;
end
if outside && ~ood_warned
    if cfg.hencky
        state = sprintf('||ln U|| = %.3f (max %.3f), J = %.3f (%.2f..%.2f)', ...
            dg.hencky, cfg.H_max, dg.J, cfg.J_min, cfg.J_max);
    else
        state = sprintf('||E_green|| = %.3f > %.3f', dg.maxE, cfg.E_max);
    end
    warning('element_quad4_nl_ai:OutOfHull', ...
        ['Elementzustand ausserhalb der trainierten Huelle (%s). Das Netz ' ...
         'extrapoliert -- Lasten reduzieren oder Huelle neu trainieren. ' ...
         'Weitere Meldungen werden unterdrueckt.'], state);
    ood_warned = true;
end

% ------------------------------------------------------------------------
% Volumenlast Fbe analytisch (nur wenn eine Last anliegt)
% ------------------------------------------------------------------------
if any(b_e ~= 0)
    numgp = size(gp, 1);
    for i = 1:numgp
        [h, ~, detJ] = shape_quad4(coord_e, gp(i, :));
        H = [ h(1) 0    h(2) 0    h(3) 0    h(4) 0
              0    h(1) 0    h(2) 0    h(3) 0    h(4) ];
        Fbe = Fbe + H' * b_e * (detJ * w(i) * mat_e(3));
    end
end

% ------------------------------------------------------------------------
end
