function [What, p, H, meta] = quad4_nl_ai_model(chat, z)
%QUAD4_NL_AI_MODEL Kanonisches Energiemodell des nichtlinearen KI-quad4.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Wertet das SKALARE Energiemodell auf der kanonischen Geometrie aus:
%
%     What(chat, z) = 0.5*z' K0(chat) z  +  Wnl(chat, z)
%     p             = dWhat/dz  = K0*z + grad_z Wnl
%     H             = d2What/dz2 = K0   + hess_z Wnl
%
%   K0 ist die EXAKTE lineare Steifigkeit der kanonischen Geometrie
%   (analytisch, 4-GP-Schleife mit linearer B-Matrix, E = d = 1, KEIN Cache).
%   Wnl ist das neuronale Residual-Netz in Subtraktionsform
%
%     Wnl = f(ct,zt) - f(ct,0) - grad_zt f(ct,0)'*zt
%
%   -> Wnl(chat,0) = 0 und grad_z Wnl(chat,0) = 0 EXAKT (unabhaengig von den
%   Gewichten). Damit ist Finte bei z = 0 exakt null und die Tangente dort
%   exakt K0 plus dem (gelernten, kleinen) Rest hess_z Wnl(chat,0).
%
%   Warum der K0-Split: fuer StVenant ist What exakt ein Polynom 4. Grades in
%   z. Der Split entfernt den quadratischen Term -- das Netz lernt nur den
%   kubisch/quartischen Rest. Das macht die Zielfunktion leichter (kleineres
%   Netz -> schnelleres Element) und laesst das Kleinamplituden-Regime
%   (Newton-Endphase) exakt von K0 dominieren.
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
%   2026-08-18
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

coords_canon = reshape(chat, 2, 4).';

% --- K0: exakte lineare Steifigkeit der kanonischen Geometrie ------------
K0 = k0_canonical(coords_canon, NET.C0);

% --- Residual-Netz: Wnl, Gradient, Hessian ------------------------------
[Wnl, pnl, Hnl] = net_residual(NET, chat, z);

% --- Modell-Summe --------------------------------------------------------
K0z  = K0 * z;
What = 0.5 * (z.' * K0z) + Wnl;
p    = K0z + pnl;
H    = K0  + Hnl;

if nargout > 3
    meta = struct('material', NET.material, 'nu_train', NET.nu_train, ...
                  'condition', NET.condition, 'E_max', NET.E_max, ...
                  'num_linear_layers', NET.L);
end

end


% ========================================================================
function K0 = k0_canonical(coords_canon, C)
%K0_CANONICAL Lineare Steifigkeit (Ke-Teil von element_quad4_lin, E = d = 1).
%   Fuer StVenant ist das exakt die Tangente bei z = 0: S(0) = 0 loescht den
%   geometrischen Term, und die nichtlineare B-Matrix faellt bei F = I auf die
%   lineare zurueck. Kein Cache -- die Schleife ist billig (4 GP, KEINE
%   Doppelknotenschleife).

persistent GP
if isempty(GP)
    s = 1/sqrt(3);
    GP = [-s -s; s -s; s s; -s s];
end

K0 = zeros(8, 8);
for i = 1:4
    [~, dh, detJ] = shape_quad4(coords_canon, GP(i, :));
    hx = dh(:, 1).';  hy = dh(:, 2).';
    B = zeros(3, 8);
    B(1, 1:2:7) = hx;
    B(2, 2:2:8) = hy;
    B(3, 1:2:7) = hy;
    B(3, 2:2:8) = hx;
    K0 = K0 + (B.' * C * B) * detJ;            % w = 1, d = 1
end
end


% ========================================================================
function [Wnl, pnl, Hnl] = net_residual(NET, chat, z)
%NET_RESIDUAL Residual-Energie und ihre ersten/zweiten Ableitungen nach z.
%
%   Subtraktionsform (identisch im Training):
%       Wnl = f(ct,zt) - f(ct,0) - grad_zt f(ct,0)'*zt
%   Daraus:
%       grad_z Wnl = ( g(zt) - g(0) ) ./ zs
%       hess_z Wnl = H(zt) ./ (zs*zs')
%   mit zt = z./zs -- NUR Skalierung, KEIN Offset (sonst waere zt(z=0) ~= 0
%   und die strukturelle Nullform gebrochen).

ct = (chat - NET.c_mean) ./ NET.c_std;
zs = NET.z_scale;
zt = z ./ zs;

[f1, g1, H1] = mlp_val_grad_hess(NET, ct, zt, true);
[f0, g0]     = mlp_val_grad_hess(NET, ct, zeros(8, 1), false);   % ohne Hessian

Wnl = f1 - f0 - g0.' * zt;
pnl = (g1 - g0) ./ zs;
Hnl = H1 ./ (zs * zs.');
Hnl = 0.5 * (Hnl + Hnl.');
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

assert_meta(S, 'model_form',       'resid_energy_K0split_subtract_f0_gradf0');
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
NET.C0 = hooke_C(1.0, NET.nu_train, NET.condition);

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


% ========================================================================
function C = hooke_C(E, nu, condition)
%HOOKE_C Materialmatrix in Voigt-Notation (Evec = [E11, E22, 2*E12]).
lam = E * nu / ((1 + nu) * (1 - 2*nu));
mu  = E / (2 * (1 + nu));
switch lower(condition)
    case 'planestrain'
        c11 = lam + 2*mu;  c12 = lam;
    case 'planestress'
        c11 = lam/(nu - 1) + 2*mu + 2*lam;
        c12 = lam/(nu - 1) + 2*lam;
    otherwise
        error('quad4_nl_ai_model:Condition', 'Unbekannter Zustand: %s', condition);
end
C = [c11 c12 0; c12 c11 0; 0 0 mu];
end
