function gradU = compute_gradU(dh_dx, Ue, DIM, DOF)
%COMPUTE_GRADU Computes the displacement gradient.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Computes the displacement gradient.
%
% INPUT
%   dh_dx    Shape-function derivatives with respect to global coordinates
%   Ue       Element displacement vector
%   DIM      Spatial dimension
%   DOF      Degrees of freedom per node
%
% OUTPUT
%   gradU    Displacement gradient
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-05-06
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

if nargin < 4
    DOF = DIM;
end

% --- 1D -----------------------------------------------------------------
if DIM == 1
    u1 = Ue(1:DOF:end);
    h_x = dh_dx(:,1);

    gradU = h_x'*u1;

% --- 2D -----------------------------------------------------------------
elseif DIM == 2
    u1 = Ue(1:DOF:end);
    u2 = Ue(2:DOF:end);

    h_x = dh_dx(:,1);
    h_y = dh_dx(:,2);

    gradU = [ u1'*h_x   u1'*h_y
              u2'*h_x   u2'*h_y ];

% --- 3D -----------------------------------------------------------------
elseif DIM == 3
    u1 = Ue(1:DOF:end);
    u2 = Ue(2:DOF:end);
    u3 = Ue(3:DOF:end);

    h_x = dh_dx(:,1);
    h_y = dh_dx(:,2);
    h_z = dh_dx(:,3);

    gradU = [ u1'*h_x   u1'*h_y   u1'*h_z
              u2'*h_x   u2'*h_y   u2'*h_z
              u3'*h_x   u3'*h_y   u3'*h_z ];

% ------------------------------------------------------------------------
else
    error('Unknown spatial dimension DIM.');
end

end
