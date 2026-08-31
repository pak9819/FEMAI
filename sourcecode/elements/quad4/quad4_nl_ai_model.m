function [What, p, H, meta] = quad4_nl_ai_model(chat, z)
%QUAD4_NL_AI_MODEL Kanonisches Energiemodell des nichtlinearen KI-quad4.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Wertet das SKALARE Energiemodell auf der kanonischen Geometrie aus:
%
%     What(chat, z) = Wnet(chat, z)          volle Energie aus dem Netz
%     p             = dWhat/dz               (Gradient des Netzes)
%     H             = d2What/dz2             (Hessian des Netzes)
%
%   Es gibt KEINE numerische Energie-/Steifigkeitsberechnung mehr im Element:
%   das Netz traegt den quadratischen UND den nichtlinearen Anteil der
%   Energie, Finte und Ke entstehen ausschliesslich durch Differentiation.
%   Das Netz ist in Subtraktionsform
%
%     Wnet = f(ct,zt) - f(ct,0) - grad_zt f(ct,0)'*zt
%
%   -> Wnet(chat,0) = 0 und grad_z Wnet(chat,0) = 0 EXAKT (unabhaengig von
%   den Gewichten). Damit ist Finte bei z = 0 exakt null. Die Tangente bei
%   z = 0 (lineare Steifigkeit) ist GELERNT -- ihre Genauigkeit wird in
%   Gate e4 (FEMSolid_ex_quad4_09_ai_nl_consistency.m) gegen das lineare
%   Element geprueft.
%
%   Diese Funktion ist bewusst von der KETTE (Kanonisierung, Ko-Rotation,
%   Rueckskalierung) getrennt: sie ist direkt gegen die Python-Oracle-
%   Vektoren im .mat testbar (Gate d in
%   FEMSolid_ex_quad4_09_ai_nl_consistency.m).
%
% INPUT
%   chat  (8x1) kanonische Knotenkoordinaten [x1;y1;...;x4;y4]
%   z     (8x1) kanonischer, ko-rotierter, translations-projizierter Zustand
%
% OUTPUT
%   What  Skalar   Energie (kanonisch, E = d = 1)
%   p     (8x1)    dWhat/dz
%   H     (8x8)    d2What/dz2 (symmetrisch)
%   meta  struct   Netz-Metadaten (nur auf Anforderung)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-08-31
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

persistent NET

if isempty(NET)
    NET = load_network();
end

% --- Netz: volle Energie, Gradient, Hessian ------------------------------
[What, p, H] = net_total(NET, chat, z);

if nargout > 3
    meta = struct('material', NET.material, 'nu_train', NET.nu_train, ...
                  'condition', NET.condition, 'E_max', NET.E_max, ...
                  'num_linear_layers', NET.L);
end

end


% ========================================================================
function [Wt, pt, Ht] = net_total(NET, chat, z)
%NET_TOTAL Volle Energie und ihre ersten/zweiten Ableitungen nach z.
%
%   Subtraktionsform (identisch im Training):
%       Wt = f(ct,zt) - f(ct,0) - grad_zt f(ct,0)'*zt
%   Daraus:
%       grad_z Wt = ( g(zt) - g(0) ) ./ zs
%       hess_z Wt = H(zt) ./ (zs*zs')
%   mit zt = z./zs -- NUR Skalierung, KEIN Offset (sonst waere zt(z=0) ~= 0
%   und die strukturelle Nullform gebrochen).

ct = (chat - NET.c_mean) ./ NET.c_std;
zs = NET.z_scale;
zt = z ./ zs;

[f1, g1, H1] = mlp_val_grad_hess(NET, ct, zt, true);
[f0, g0]     = mlp_val_grad_hess(NET, ct, zeros(8, 1), false);   % ohne Hessian

Wt = f1 - f0 - g0.' * zt;
pt = (g1 - g0) ./ zs;
Ht = H1 ./ (zs * zs.');
Ht = 0.5 * (Ht + Ht.');
end


% ========================================================================
function [fval, gz, Hz] = mlp_val_grad_hess(NET, ct, zt, need_hess)
%MLP_VAL_GRAD_HESS Wert, Gradient und Hessian des rohen Skalar-MLP nach zt.
%
%   Handkodierte Rekurrenzen (tiefen-agnostisch ueber NET.L Gewichtslagen):
%     Forward :  z_l = W_l*a_{l-1} + b_l,  a_l = gelu(z_l)
%     Reverse :  r_{l-1} = W_l'*(gelu'(z_l).*r_l),  Seed r_L = 1
%     Forward-over-Reverse (alle 8 Richtungen als Matrix-Batch, keine
%     Richtungsschleife):
%       rd_{l-1} = W_l'*( gelu'(z_l).*rd_l + gelu''(z_l).*zd_l.*r_l )
%
%   need_hess = false ueberspringt die Tangenten komplett (der Null-Zustands-
%   Pass braucht nur Wert und Gradient).
%
%   GELU in exakter erf-Form (== PyTorch nn.GELU(), approximate='none'):
%     gelu(x)   = x*Phi(x)
%     gelu'(x)  = Phi(x) + x*phi(x)
%     gelu''(x) = (2 - x^2)*phi(x)
%   mit Phi = 0.5*(1+erf(x/sqrt(2))), phi = exp(-x^2/2)/sqrt(2*pi).

L  = NET.L;
W  = NET.W;
b  = NET.b;
nz = 8;

Zd = cell(L, 1);          % zd_l = dz_l/dzt   (n_l x 8)
D1 = cell(L, 1);          % gelu'(z_l)
D2 = cell(L, 1);          % gelu''(z_l)

% --- Forward (+ Tangenten nur wenn der Hessian gebraucht wird) ------------
a  = [ct; zt];
Ad = [];
if need_hess
    Ad = [zeros(8, nz); eye(nz)];   % da_0/dzt  (16 x 8)
end

for l = 1:L
    zl = W{l} * a + b{l};
    if need_hess
        Zd{l} = W{l} * Ad;
    end
    if l < L
        [gv, g1v, g2v] = gelu_derivs(zl);
        a = gv;
        D1{l} = g1v;
        D2{l} = g2v;
        if need_hess
            Ad = g1v .* Zd{l};
        end
    else
        fval = zl;                  % Ausgabeschicht: linear, Skalar
    end
end

% --- Reverse: Gradient (+ Forward-over-Reverse fuer den Hessian) ---------
%   Invariante beim Eintritt in Iteration l < L:  r = df/da_l
r  = 1.0;
rd = zeros(1, nz);
td = [];
for l = L:-1:1
    if l == L
        t = r;
        if need_hess, td = rd; end
    else
        t = D1{l} .* r;
        if need_hess
            td = D1{l} .* rd + (D2{l} .* Zd{l}) .* r;
        end
    end
    r = W{l}.' * t;
    if need_hess
        rd = W{l}.' * td;
    end
end

gz = r(9:16);                       % df/dzt
if need_hess
    Hz = rd(9:16, :);               % d2f/dzt2
    Hz = 0.5 * (Hz + Hz.');
else
    Hz = [];
end
end


% ========================================================================
function [g, g1, g2] = gelu_derivs(x)
%GELU_DERIVS GELU (exakte erf-Form) mit erster und zweiter Ableitung.
Phi = 0.5 * (1 + erf(x / sqrt(2)));
phi = exp(-0.5 * x.^2) / sqrt(2*pi);
g  = x .* Phi;
g1 = Phi + x .* phi;
g2 = (2 - x.^2) .* phi;
end


% ========================================================================
function NET = load_network()
%LOAD_NETWORK Netz + Metadaten einmalig laden und aufbereiten.
%   HARTE Fehler (kein Warning) bei allem, was stumm falsche Physik ergaebe.

netFile = fullfile(fileparts(mfilename('fullpath')), 'quad4_nl_W_network.mat');
if ~exist(netFile, 'file')
    error('quad4_nl_ai_model:NetworkMissing', ...
        ['quad4_nl_W_network.mat nicht gefunden.\n' ...
         'Bitte zuerst training/quad4/train_quad4_nl_W_network.py ausfuehren.']);
end
S = load(netFile);

req = {'model_form', 'activation', 'canonicalization', 'state_corotation', ...
       'num_linear_layers', 'input_norm_c_mean', 'input_norm_c_std', ...
       'input_norm_z_scale', 'material', 'nu_train', 'condition'};
for i = 1:numel(req)
    if ~isfield(S, req{i})
        error('quad4_nl_ai_model:MetaMissing', ...
            'Pflicht-Metadatum "%s" fehlt im Netz -- Netz neu trainieren.', req{i});
    end
end

assert_meta(S, 'model_form',       'total_energy_subtract_f0_gradf0');
assert_meta(S, 'activation',       'GELU_erf');
assert_meta(S, 'canonicalization', 'edge_n1n2_to_pos_x_after_centroid_Lc');
assert_meta(S, 'state_corotation', 'mean_polar_angle_at_center_removed__u_c=R(-th)(x+u)-x');

NET.L = round(double(S.num_linear_layers));
NET.W = cell(NET.L, 1);
NET.b = cell(NET.L, 1);
for l = 1:NET.L
    NET.W{l} = double(S.(sprintf('W%d', l)));
    bl = double(S.(sprintf('b%d', l)));
    NET.b{l} = bl(:);
end
if size(NET.W{1}, 2) ~= 16
    error('quad4_nl_ai_model:InputSize', ...
        'Erste Gewichtslage erwartet 16 Eingaenge, hat %d.', size(NET.W{1}, 2));
end
if size(NET.W{NET.L}, 1) ~= 1
    error('quad4_nl_ai_model:OutputSize', ...
        'Ausgabeschicht muss SKALAR sein (Energie), hat %d Ausgaenge.', ...
        size(NET.W{NET.L}, 1));
end

NET.c_mean  = double(S.input_norm_c_mean(:));
NET.c_std   = double(S.input_norm_c_std(:));
NET.z_scale = double(S.input_norm_z_scale(:));

NET.nu_train  = double(S.nu_train);
NET.material  = strtrim(char(S.material));
NET.condition = strtrim(char(S.condition));

NET.E_max = 0.2;
if isfield(S, 'state_E_max')
    NET.E_max = double(S.state_E_max);
end
end


% ========================================================================
function assert_meta(S, field, expected)
val = strtrim(char(S.(field)));
if ~strcmpi(val, expected)
    error('quad4_nl_ai_model:MetaMismatch', ...
        ['Netz-Metadatum "%s" ist "%s", erwartet "%s".\n' ...
         'Das Element wuerde damit stumm falsch rechnen -- Netz neu trainieren.'], ...
        field, val, expected);
end
end
