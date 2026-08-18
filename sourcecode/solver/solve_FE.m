function [U, results] = solve_FE(model)
%SOLVE_FE Solves a linear or nonlinear FE model.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Solves the FE model and dispatches to the linear or nonlinear solver
%   depending on model.analysis.nl.
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

timer = tic;

print_model_summary(model);

switch model.analysis.nl

    % --------------------------------------------------------------------
    case false
        analysisName = 'linear analysis';
        fprintf('\nStart %s ...\n', analysisName);
        [U, results] = solve_linear_FE(model);

    % --------------------------------------------------------------------
    case true
        analysisName = 'nonlinear analysis';
        fprintf('\nStart %s with %i load step(s) ...\n', analysisName, model.analysis.numSteps);
        [U, results] = solve_nonlinear_FE(model);

end

elapsedTime = toc(timer);
fprintf('End %s. Calculation time: %.3f s.\n', analysisName, elapsedTime);

end
