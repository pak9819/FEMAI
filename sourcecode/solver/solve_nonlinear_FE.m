function [U,results] = solve_nonlinear_FE(model)
%SOLVE_NONLINEAR_FE Solves a nonlinear static FE problem.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Solves the nonlinear static FE problem by applying the load factors and
%   calling the Newton solver for each load step.
%
% INPUT
%   model    FE model structure
%
% OUTPUT
%   U        Global displacement vector
%   results  Results structure
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

numloadsteps = model.analysis.numSteps;
results = init_results(model);
U = zeros(model.info.NDOF,1);

for it = 1:numloadsteps
   loadscale = model.analysis.loadFactor(it);
   [U,results(it)] = newton(model, U, loadscale);
end

end
