function [h, dh_dx, detJ] = shape_brick8(coord_e, rst_loc)
%SHAPE_BRICK8 Evaluates shape functions and derivatives of the 8-node brick.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Trilinear isoparametric hexahedron. Node numbering (natural coords):
%
%     1 (-1,-1,-1)  2 ( 1,-1,-1)  3 ( 1, 1,-1)  4 (-1, 1,-1)
%     5 (-1,-1, 1)  6 ( 1,-1, 1)  7 ( 1, 1, 1)  8 (-1, 1, 1)
%
%   i.e. nodes 1-4 counter-clockwise on the bottom face t = -1, nodes 5-8
%   above them on t = +1 (consistent with gauss_library.m and the face
%   table in plot_results.m).
%
% INPUT
%   coord_e  (8x3) global node coordinates
%   rst_loc  [r, s, t] local coordinates
%
% OUTPUT
%   h        (8x1) shape functions
%   dh_dx    (8x3) derivatives with respect to x, y, z
%   detJ     Jacobian determinant
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
%
% COPYRIGHT AND LICENSE
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

persistent RN SN TN
if isempty(RN)
    RN = [-1;  1;  1; -1; -1;  1;  1; -1];
    SN = [-1; -1;  1;  1; -1; -1;  1;  1];
    TN = [-1; -1; -1; -1;  1;  1;  1;  1];
end

r = rst_loc(1);  s = rst_loc(2);  t = rst_loc(3);

ar = 1 + RN*r;  as = 1 + SN*s;  at = 1 + TN*t;
h = ar .* as .* at / 8;

dh_dr = [ RN .* as .* at, ar .* SN .* at, ar .* as .* TN ] / 8;   % (8x3)

J = coord_e.' * dh_dr;              % J(i,k) = dx_i/dr_k
detJ = det(J);
dh_dx = dh_dr / J;                  % = dh_dr * inv(J)

end
