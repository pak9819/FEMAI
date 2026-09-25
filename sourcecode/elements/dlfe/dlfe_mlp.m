function [fval, g, THT] = dlfe_mlp(NET, x, T, rows)
%DLFE_MLP Wert, Gradient und projizierte Hessian eines Skalar-MLP.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Skalar-MLP f(x) mit GELU (exakte erf-Form == nn.GELU()):
%
%     fval = f(x)                    Skalar
%     g    = df/dx                   (n_in x 1)
%     THT  = T' * d2f/dx2 * T        (ndir x ndir, symmetrisch)
%
%   Rekurrenzen:
%     Forward :  z_l = W_l*a_{l-1} + b_l,  a_l = gelu(z_l)
%     Reverse :  r_{l-1} = W_l'*(gelu'(z_l).*r_l),   r_l = df/da_l
%
%   Projizierte Hessian OHNE Rueckwaerts-Tangenten: Da nur die Aktivierung
%   nichtlinear ist, gilt exakt
%
%     T' H T = sum_{l<L} Zd_l' * diag( gelu''(z_l) .* r_l ) * Zd_l
%
%   mit den Vorwaerts-Tangenten Zd_l = dz_l/dx * T. Gegenueber dem frueheren
%   Forward-over-Reverse (H*T) entfaellt der Rueckwaerts-Tangentenpass
%   komplett (etwa die Haelfte der Hessian-Kosten); das Ergebnis ist
%   mathematisch identisch.
%
% INPUT
%   NET   Netz-Struktur (dlfe_load_network): L, W{l}, b{l}
%   x     (n_in x 1) Eingang (bereits normiert)
%   T     (numel(rows) x ndir) Tangentenrichtungen oder [] (nur f, g)
%   rows  Eingangsindizes, auf denen T lebt (Standard: alle). Die uebrigen
%         Eingaenge haben Tangente null -> Lage 1 multipliziert nur diese
%         Spalten von W_1.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-25
% ------------------------------------------------------------------------

L = NET.L;
W = NET.W;
b = NET.b;
needH = nargin > 2 && ~isempty(T);

Zd = cell(L-1, 1);
D1 = cell(L-1, 1);
D2 = cell(L-1, 1);

a = x;
for l = 1:L
    zl = W{l} * a + b{l};
    if l < L
        if needH
            if l == 1
                if nargin > 3 && ~isempty(rows)
                    Zd{1} = W{1}(:, rows) * T;
                else
                    Zd{1} = W{1} * T;
                end
            else
                Zd{l} = W{l} * (D1{l-1} .* Zd{l-1});
            end
        end
        Phi = 0.5 * (1 + erf(zl / sqrt(2)));
        phi = exp(-0.5 * zl.^2) / sqrt(2*pi);
        a     = zl .* Phi;
        D1{l} = Phi + zl .* phi;
        D2{l} = (2 - zl.^2) .* phi;
    else
        fval = zl;
    end
end

% Reverse: r = df/da_l beim Eintritt in Iteration l < L
r = W{L}.';
if needH
    THT = zeros(size(T, 2));
end
for l = L-1:-1:1
    if needH
        THT = THT + Zd{l}.' * ((D2{l} .* r) .* Zd{l});
    end
    r = W{l}.' * (D1{l} .* r);
end

g = r;
if needH
    THT = 0.5 * (THT + THT.');
else
    THT = [];
end
end
