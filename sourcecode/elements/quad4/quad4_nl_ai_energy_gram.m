function [Wphys, Finte, Ke, dg, meta] = quad4_nl_ai_energy_gram(coord_e, mat_e, Ue, netFile)
%QUAD4_NL_AI_ENERGY_GRAM Gram-Kette des KI-quad4 (Metrik-Eingang statt Ko-Rotation).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Alternative zu quad4_nl_ai_energy.m (Ko-Rotation) -- Phase 0 der
%   brick8-Uebertragung. Gleiche Schnittstelle, gleiche Faktorisierung:
%
%     W_phys = E*d*Lc^2 * What(chat, D)
%     Finte  = E*d*Lc   * dWhat/du_hat
%     Ke     = E*d      * d2What/du_hat2
%
%   Der Zustand ist rotationsinvariant (dlfe_gram_energy), daher entfallen
%   Tc, Ko-Rotationswinkel und seine Ableitungen komplett. Gewaehlt wird
%   diese Kette ueber quad4_nl_ai_network_file (QUAD4_STATE_FORM=gram).
%
% INPUT / OUTPUT
%   wie quad4_nl_ai_energy.m; netFile = Pfad des Gram-Netzes
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

persistent NETS

key = matlab.lang.makeValidName(netFile);
if isempty(NETS) || ~isfield(NETS, key)
    NETS.(key) = dlfe_load_network(netFile, 4, 2);
end
NET = NETS.(key);

Emod = mat_e(1);
d    = mat_e(3);

[chat, Lc, ~, xbar] = dlfe_canonical_frame(coord_e);
uhat = reshape(Ue, 2, 4).' / Lc;

if nargout > 2
    [What, gu, Ku] = dlfe_gram_energy(NET, chat, xbar, uhat);
    Ke = Emod * d * Ku;
else
    [What, gu] = dlfe_gram_energy(NET, chat, xbar, uhat);
end
Wphys = Emod * d * Lc^2 * What;
Finte = Emod * d * Lc * gu;

if nargout > 3
    [~, dh0, ~] = shape_quad4(coord_e, [0 0]);
    Fdef = eye(2) + reshape(Ue, 2, 4) * dh0;
    Eg   = 0.5 * (Fdef.' * Fdef - eye(2));
    dg.maxE   = norm(Eg, 'fro');
    dg.J      = det(Fdef);
    lam2      = max(eig(Fdef.' * Fdef), realmin);
    dg.hencky = norm(0.5 * log(lam2));
    dg.Lc     = Lc;
    dg.theta  = NaN;
end
if nargout > 4
    meta = struct('material', NET.material, 'nu_train', NET.nu_train, ...
                  'condition', NET.condition, 'E_max', NET.E_max, ...
                  'state_measure', NET.state_measure, 'H_max', NET.H_max, ...
                  'J_min', NET.J_min, 'J_max', NET.J_max, ...
                  'num_linear_layers', NET.L);
end
end
