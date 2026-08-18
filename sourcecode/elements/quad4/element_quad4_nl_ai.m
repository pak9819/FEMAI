function [Ke, Fbe, Fte, Finte, history] = element_quad4_nl_ai(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_QUAD4_NL_AI Nichtlineares quad4-Element: Residual-Energie-Netz.
% ------------------------------------------------------------------------
% DESCRIPTION
%   "Deep Learned Finite Elements" fuer das bilineare Viereckselement in der
%   NICHTLINEAREN Analyse (Total Lagrange, St.-Venant-Kirchhoff).
%
%   Das Element ersetzt die Gauss-Schleife des klassischen Elements durch ein
%   SKALARES Energiemodell auf der kanonischen Geometrie:
%
%     What(chat, z) = 0.5*z' K0hat(chat) z  +  Wnl(chat, z)
%
%   K0hat ist die EXAKTE lineare Steifigkeit der kanonischen Geometrie
%   (analytisch pro Aufruf, kleine 4-GP-Schleife mit linearer B-Matrix --
%   bewusst OHNE Cache, damit assemble.m zustandslos bleibt). Wnl ist ein
%   kleines neuronales Residual-Netz. Beide Ausgabegroessen entstehen durch
%   DIFFERENTIATION desselben Potentials:
%
%     Finte = dW/dUe        Ke = d2W/dUe2 = dFinte/dUe
%
%   -> Ke ist PER KONSTRUKTION exakt die Jacobimatrix von Finte. Genau diese
%   Konsistenz fehlte der Vorgaengerversion (zwei unabhaengige Netzkoepfe fuer
%   Ke und Finte, gemessene Inkonsistenz ~7 %), die deshalb nur linear mit
%   Kontraktionsrate ~0.9 konvergierte und das Iterationslimit erreichte.
%
%   Warum der K0-Split: fuer StVenant ist What exakt ein Polynom 4. Grades in
%   z. Der Split entfernt den quadratischen Term -- das Netz lernt nur den
%   kubisch/quartischen Rest (leichtere Zielfunktion, kleineres Netz) und das
%   Kleinamplituden-Regime (Newton-Endphase) wird von K0hat EXAKT dominiert.
%
%   Exakte Struktur (nicht gelernt):
%     - Konsistenz Ke = dFinte/dUe            (Potentialform)
%     - Symmetrie von Ke                      (Hessian eines Skalars)
%     - Kraeftegleichgewicht sum_i Finte_i = 0 und Translationsnullraum von
%       Ke                                    (Translationsprojektor P)
%     - Finte = 0 bei reiner Starrkoerperbewegung, in Maschinengenauigkeit
%       (Ko-Rotation liefert z = 0, Subtraktionsform liefert Fnl(c,0) = 0)
%     - Groessen-, Translations- und Rotationsinvarianz sowie Objektivitaet
%       (Kanonisierung + Ko-Rotation, Ableitungen exakt mitgefuehrt)
%
%   Faktorisierungen (StVenant, exakt):
%     W     = E * d * Lc^2 * What
%     Finte = E * d * Lc   * Tc' * P * g_v
%     Ke    = E * d        * Tc' * P * K_v * P * Tc
%   nu = 0.3, planeStrain und das Materialgesetz (StVenant) sind FEST
%   eintrainiert und werden beim Laden HART geprueft; E und Dicke d sind frei.
%
%   Volumenlast Fbe wird weiterhin analytisch berechnet (Fte = 0, wie im
%   analytischen nl-Element).
%
% PREREQUISITE
%   Run training/quad4/train_quad4_nl_W_network.py first to generate
%   quad4_nl_W_network.mat.
%
% INPUT / OUTPUT
%   Identisch zu element_quad4_nl.m.
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

persistent cfg_checked ood_count ood_warned E_max_train

if isempty(cfg_checked)
    cfg_checked = false;
    ood_count   = 0;
    ood_warned  = false;
end

DIM    = size(coord_e, 2);
DOF    = DIM;
NNEL   = size(coord_e, 1);
NDOFEL = NNEL * DOF;

Fbe = zeros(NDOFEL, 1);
Fte = zeros(NDOFEL, 1);

% ------------------------------------------------------------------------
% Konsistenzpruefung (nur EINMAL pro Session, nicht im Hot-Path).
% HART (error): alles, was stumm falsche Physik ergaebe -- dieses Netz ist
% materialgebunden (StVenant, nu, ebener Zustand sind eintrainiert).
% ------------------------------------------------------------------------
if ~cfg_checked
    [~, Finte, Ke, dg, meta] = quad4_nl_ai_energy(coord_e, mat_e, Ue);

    if ~strcmpi(strtrim(char(meta.material)), MATNAME)
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

    E_max_train = meta.E_max;
    cfg_checked = true;
else
    % ------------------------------------------------------------------
    % Kern: Energie -> Finte, Ke (Kanonisierung, Ko-Rotation, K0-Split, Netz)
    % ------------------------------------------------------------------
    [~, Finte, Ke, dg] = quad4_nl_ai_energy(coord_e, mat_e, Ue);
end

% ------------------------------------------------------------------------
% OOD-Proxy: ||E_green|| am Elementmittelpunkt (der Verschiebungsgradient
% liegt fuer die Ko-Rotation ohnehin vor -> praktisch kostenlos). Die volle
% 4-GP-Huellenpruefung bleibt bewusst DRAUSSEN aus dem Hot-Path; dafuer gibt
% es FEMSolid_ex_quad4_08_ai_nl_check.m.
% ------------------------------------------------------------------------
if dg.maxE > E_max_train
    ood_count = ood_count + 1;
    if ~ood_warned
        warning('element_quad4_nl_ai:OutOfHull', ...
            ['Elementzustand ausserhalb der trainierten Huelle ' ...
             '(||E_green|| = %.3f > %.3f). Das Netz extrapoliert -- Lasten ' ...
             'reduzieren oder Huelle neu trainieren. Weitere Meldungen ' ...
             'werden unterdrueckt (Diagnose: FEMSolid_ex_quad4_08_ai_nl_check).'], ...
            dg.maxE, E_max_train);
        ood_warned = true;
    end
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
