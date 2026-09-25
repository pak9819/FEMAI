function [What, gu, Ku, D] = dlfe_gram_energy(NET, chat, xbar, uhat)
%DLFE_GRAM_ENERGY Gram-Kette: Energie, Gradient und Hessian nach u_hat.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Kern der dimensionsunabhaengigen DLFE-Kette (quad4 und brick8). Das
%   Netz sieht die kanonische Geometrie chat und einen METRIK-Zustand
%
%     Z = L' * Y,   Z0 = L' * X,   Y = X + U  (X = xbar, U = uhat)
%     D = triu( Z Z' - Z0 Z0' ) = triu( Z0 Uz' + Uz Z0' + Uz Uz' ),  Uz = L'U
%
%   mit einer festen (nur geometrieabhaengigen) Basis L:
%     'gram'       L = Pn = I - 11'/n           (Knoten-Gram, k = n)
%     'gram_modal' L = [dh0 | gamma]            (k = n-1)
%                  dh0   = Formfunktionsgradienten im Zentrum (kanonisch)
%                  gamma = Hourglass-Vektoren (Flanagan-Belytschko)
%                  -> D = triu([F0'F0 - I, F0'q ; q'F0, q'q])
%
%   D ist invariant unter jeder Starrkoerperbewegung; bei reiner
%   Starrkoerperbewegung ist D = 0 und (Subtraktionsform) W = grad W = 0
%   exakt. Eine Ko-Rotation / Polarzerlegung entfaellt.
%
%     What = f(c~, D~) - f(c~, 0) - grad_D~ f(c~, 0)' D~,   D~ = D ./ Ds
%     p    = dWhat/dD                          (m x 1)
%     S    = k x k symmetrisch, S_ij = p (i ~= j), S_ii = 2 p_ii
%     gu   = vec( L*S*Z )                      = dWhat/du_hat
%     Ku   = Jf' H Jf + kron(L*S*L', I_dim)    = d2What/du_hat2
%     Jf   = dD/dZ * kron(L', I_dim)           (m x ndof)
%
%   H wird nie explizit gebildet: dlfe_mlp liefert mit den ndof Richtungen
%   T = Jf./Ds direkt T'*H~*T aus den Vorwaerts-Tangenten (ohne
%   Rueckwaerts-Tangentenpass, 2026-09-25).
%
%   Alle Groessen sind KANONISCH (E = d = 1, Lc = 1); die Rueckskalierung
%   auf das physikalische Element macht der Aufrufer.
%
% INPUT
%   NET   Netz (dlfe_load_network)
%   chat  (ndof x 1) kanonische Geometrie
%   xbar  (n x dim) zentrierte, durch Lc geteilte Knotenkoordinaten
%   uhat  (n x dim) Verschiebungen / Lc (gleiche Orientierung wie xbar)
%
% OUTPUT
%   What  Skalar, gu (ndof x 1), Ku (ndof x ndof, symmetrisch), D (m x 1)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-25
% ------------------------------------------------------------------------

n = NET.n;  dim = NET.dim;  ndof = NET.ndof;  m = NET.m;  k = NET.k;
Ds = NET.D_scale;

% --- Basis L ------------------------------------------------------------
if NET.modal
    Xc  = reshape(chat, dim, n).';                 % kanonisch, zentriert
    Pl  = NET.Phi(:, 1:dim);
    Ph  = NET.Phi(:, dim+1:end);
    J0  = Xc.' * Pl;                               % J0(i,k) = dx_i/dr_k
    dh0 = Pl / J0;
    L   = [dh0, Ph - dh0 * (Xc.' * Ph)];           % (n x k)
else
    L   = NET.Pn;                                  % (n x n)
end

Z0 = L.' * xbar;                                   % (k x dim)  (L'1 = 0)
Uz = L.' * uhat;
Z  = Z0 + Uz;

M = Z0 * Uz.';
M = M + M.' + Uz * Uz.';                           % ausloeschungsfrei
D = M(NET.triuMask);

ct = (chat - NET.c_mean) ./ NET.c_std;
Dt = D ./ Ds;

needK = nargout > 2;
if needK
    Jz = zeros(m, k*dim);                          % dD/dZ
    Jz(NET.jyIdxJ) = Z(NET.jyValJ);
    Jz(NET.jyIdxI) = Jz(NET.jyIdxI) + Z(NET.jyValI);
    % Jf = Jz * kron(L', I_dim) ohne kron: Spalten (d, Mode) -> (d, Knoten)
    Jf  = reshape(reshape(Jz, m*dim, k) * L.', m, ndof);   % dD/du_hat
    Jfs = Jf ./ Ds;
    % projizierte Hessian Jfs' * H~ * Jfs direkt (nur D-Eingaenge tragen Tangenten)
    [f1, g1, Kmat] = dlfe_mlp(NET, [ct; Dt], Jfs, ndof+1:ndof+m);
else
    [f1, g1] = dlfe_mlp(NET, [ct; Dt], []);
end
[f0, g0] = dlfe_mlp(NET, [ct; zeros(m, 1)], []);

gD1 = g1(ndof+1:end);
gD0 = g0(ndof+1:end);
What = f1 - f0 - gD0.' * Dt;
p = (gD1 - gD0) ./ Ds;

S = zeros(k);
S(NET.triuMask) = p;
S = S + S.';                                       % Diagonale automatisch 2*p_ii

gY = L * (S * Z);                                  % (n x dim), translationsfrei
gu = reshape(gY.', [], 1);

if needK
    Ku = Kmat;
    G  = L * S * L.';                              % kron(G, I_dim) addieren
    Ku(NET.kronIdx) = Ku(NET.kronIdx) + repmat(G(:), dim, 1);
    Ku = 0.5 * (Ku + Ku.');
end
end
