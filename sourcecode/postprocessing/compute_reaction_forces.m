function Freact = compute_reaction_forces(model, Fint, Fext)
%COMPUTE_REACTION_FORCES Computes reaction forces at prescribed displacement DOFs.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Computes the global reaction force vector from internal and external
%   forces. Reaction forces are only stored at degrees of freedom with
%   prescribed displacements.
%
% INPUT
%   model    FE model structure
%   Fint     Global internal force vector
%   Fext     Global external force vector
%
% OUTPUT
%   Freact   Global reaction force vector
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

dofsfix = model.dofs.fixed;

R = Fint - Fext;

Freact = zeros(model.info.NDOF,1);
Freact(dofsfix) = R(dofsfix);

end
