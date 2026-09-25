function [Wphys, Finte, Ke, dg, meta] = brick8_nl_ai_energy(coord_e, mat_e, Ue, MATNAME)
%BRICK8_NL_AI_ENERGY Kanonisierungs- und Metrik-Kette des KI-brick8.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Bringt ein physisches Hexaeder in den kanonischen Rahmen, wertet dort
%   das Energienetz ueber die MODALE Metrik-Kette aus (dlfe_gram_energy)
%   und skaliert Energie, erste und zweite Ableitung EXAKT zurueck:
%
%     W_phys = E * Lc^3 * What(chat, D)
%     Finte  = E * Lc^2 * dWhat/du_hat       = dW_phys/dUe
%     Ke     = E * Lc   * d2What/du_hat2     = dFinte/dUe
%
%   Der Zustand D (Metrik von F0 im Zentrum plus Hourglass-Vektoren) ist
%   rotationsinvariant -- in 3D ersetzt das die 2D-Ko-Rotation, fuer die es
%   keine geschlossene Form gibt. Rc wird nur fuer die Geometrie gebraucht.
%
% INPUT
%   coord_e  (8x3) Knotenkoordinaten, mat_e [E, nu, NaN, alphaT],
%   Ue (24x1), MATNAME (Standard 'StVenant')
%
% OUTPUT
%   Wphys, Finte (24x1), Ke (24x24), dg (Diagnose am Mittelpunkt: maxE =
%   ||E_green||, J, hencky, Lc), meta (Netz-Metadaten)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

persistent NET lastName

if nargin < 4
    MATNAME = 'StVenant';
end
if isempty(lastName) || ~strcmp(lastName, MATNAME)
    netFile  = brick8_nl_ai_network_file(MATNAME);
    NET      = dlfe_load_network(netFile, 8, 3);
    lastName = MATNAME;
end

Emod = mat_e(1);

[chat, Lc, ~, xbar] = dlfe_canonical_frame(coord_e);
uhat = reshape(Ue, 3, 8).' / Lc;

if nargout > 2
    [What, gu, Ku] = dlfe_gram_energy(NET, chat, xbar, uhat);
    Ke = (Emod * Lc) * Ku;
else
    [What, gu] = dlfe_gram_energy(NET, chat, xbar, uhat);
end
Wphys = Emod * Lc^3 * What;
Finte = (Emod * Lc^2) * gu;

if nargout > 3
    [~, dh0] = shape_brick8(coord_e, [0 0 0]);
    Fdef = eye(3) + reshape(Ue, 3, 8) * dh0;
    Eg   = 0.5 * (Fdef.' * Fdef - eye(3));
    dg.maxE   = norm(Eg, 'fro');
    dg.J      = det(Fdef);
    lam2      = max(eig(Fdef.' * Fdef), realmin);
    dg.hencky = norm(0.5 * log(lam2));
    dg.Lc     = Lc;
end
if nargout > 4
    meta = struct('material', NET.material, 'nu_train', NET.nu_train, ...
                  'condition', NET.condition, 'E_max', NET.E_max, ...
                  'state_measure', NET.state_measure, 'H_max', NET.H_max, ...
                  'J_min', NET.J_min, 'J_max', NET.J_max, ...
                  'state_form', NET.state_form, 'num_linear_layers', NET.L);
end
end
