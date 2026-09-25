function [Ke, Fbe, Fte, Finte, history] = element_brick8_lin(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_BRICK8_LIN Linear 8-node brick: stiffness, loads, internal forces.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Small-strain trilinear hexahedron (B-bar-free, full 2x2x2 by default).
%   Voigt order [e11 e22 e33 2e12 2e23 2e13] (Emat2Evec / Svec2Smat).
%   3D material card: [E, nu, NaN, alphaT] -- no thickness.
%
% INPUT / OUTPUT
%   Identical to element_quad4_lin.m.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
%
% COPYRIGHT AND LICENSE
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

Ke  = zeros(24, 24);
Fbe = zeros(24, 1);
Fte = zeros(24, 1);

alphaT = mat_e(4);
if isnan(alphaT), alphaT = 0; end
if isempty(DeltaT_e)
    DeltaT = 0;
else
    DeltaT = DeltaT_e(1);
end
epsT = alphaT * DeltaT * [1; 1; 1; 0; 0; 0];

[~, C] = material_elasticity(mat_e, zeros(3), MATNAME, MATCOND);

for i = 1:size(gp, 1)
    [h, dh, detJ] = shape_brick8(coord_e, gp(i,:));
    B = bmat_lin(dh);
    dV = detJ * w(i);
    Ke  = Ke + B.' * C * B * dV;
    Hm  = kron(h.', eye(3));
    Fbe = Fbe + Hm.' * b_e * dV;
    Fte = Fte + B.' * C * epsT * dV;
end

Finte = Ke * Ue;
end


function B = bmat_lin(dh)
hx = dh(:,1).';  hy = dh(:,2).';  hz = dh(:,3).';
B = zeros(6, 24);
B(1, 1:3:end) = hx;
B(2, 2:3:end) = hy;
B(3, 3:3:end) = hz;
B(4, 1:3:end) = hy;  B(4, 2:3:end) = hx;
B(5, 2:3:end) = hz;  B(5, 3:3:end) = hy;
B(6, 1:3:end) = hz;  B(6, 3:3:end) = hx;
end
