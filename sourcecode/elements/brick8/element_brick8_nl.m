function [Ke, Fbe, Fte, Finte, history] = element_brick8_nl(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_BRICK8_NL Nonlinear 8-node brick (Total Lagrange).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Total-Lagrange trilinear hexahedron, analogous to element_quad4_nl.m:
%
%     Finte = sum_gp  B_L' * S          * dV
%     Ke    = sum_gp (B_L' * C * B_L + kron(dh*S*dh', I3)) * dV
%
%   B_L(:, node a, comp k) = d(Evec)/d(u_ak) with
%     row 1..3 : F_k1 h_a,1 | F_k2 h_a,2 | F_k3 h_a,3
%     row 4    : F_k1 h_a,2 + F_k2 h_a,1        (2 E12)
%     row 5    : F_k2 h_a,3 + F_k3 h_a,2        (2 E23)
%     row 6    : F_k1 h_a,3 + F_k3 h_a,1        (2 E13)
%   S, C from material_elasticity (StVenant, NeoHooke, ... in 3D).
%   Fte = 0 (temperature only in the linear analysis, like quad4 nl).
%
% INPUT / OUTPUT
%   Identical to element_quad4_nl.m. mat_e = [E, nu, NaN, alphaT].
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
%
% COPYRIGHT AND LICENSE
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

Ke    = zeros(24, 24);
Fbe   = zeros(24, 1);
Fte   = zeros(24, 1);
Finte = zeros(24, 1);

Un = reshape(Ue, 3, 8);                    % (3x8), column a = u_a

for i = 1:size(gp, 1)
    [h, dh, detJ] = shape_brick8(coord_e, gp(i,:));
    dV = detJ * w(i);

    gradU = Un * dh;                       % du_k/dX_j
    F = eye(3) + gradU;

    [S, C] = material_elasticity(mat_e, gradU, MATNAME, MATCOND);
    Svec = [S(1,1); S(2,2); S(3,3); S(1,2); S(2,3); S(1,3)];

    hx = dh(:,1).';  hy = dh(:,2).';  hz = dh(:,3).';
    B = zeros(6, 24);
    for k = 1:3
        c = k:3:24;
        B(1, c) = F(k,1) * hx;
        B(2, c) = F(k,2) * hy;
        B(3, c) = F(k,3) * hz;
        B(4, c) = F(k,1) * hy + F(k,2) * hx;
        B(5, c) = F(k,2) * hz + F(k,3) * hy;
        B(6, c) = F(k,1) * hz + F(k,3) * hx;
    end

    Finte = Finte + B.' * Svec * dV;
    Ke    = Ke + (B.' * C * B + kron(dh * S * dh.', eye(3))) * dV;

    if any(b_e ~= 0)
        Fbe = Fbe + kron(h, eye(3)) * b_e * dV;
    end
end
end
