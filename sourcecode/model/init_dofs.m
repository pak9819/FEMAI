function dofs = init_dofs(model)
%INIT_DOFS Initializes FEM-Solid Edu data structures.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Initializes FEM-Solid Edu data structures.
%
% INPUT
%   model    FE model structure
%
% OUTPUT
%   dofs     Degree-of-freedom indices
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

gdof  = model.info.gdof;
bcond = model.bcond;

nBC = size(bcond,1);

dofs.fixed        = zeros(nBC,1);
dofs.fixedValues  = zeros(nBC,1);

% -------------------------------------------------------------------------
% Dirichlet-Randbedingungen → globale DOF-Nummern
% -------------------------------------------------------------------------
for i = 1:nBC

    node  = bcond(i,1);   % Knotennummer
    dir   = bcond(i,2);   % Richtung (1=x, 2=y, 3=z)
    value = bcond(i,3);   % vorgegebene Verschiebung

    dofs.fixed(i)       = gdof(node,dir);
    dofs.fixedValues(i) = value;

end

% -------------------------------------------------------------------------
% Freie DOFs
% -------------------------------------------------------------------------
allDofs   = (1:model.info.NDOF)';
dofs.free = setdiff(allDofs, dofs.fixed);

end
