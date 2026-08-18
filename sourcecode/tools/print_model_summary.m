function print_model_summary(model)
%PRINT_MODEL_SUMMARY Prints a summary of model and solver settings.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Prints the most important analysis, element, material, mesh and solver settings of the FE model.
%
% INPUT
%   model    FE model structure
%
% OUTPUT
%   none
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-05-08
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

if model.analysis.nl
    analysisMode = 'nonlinear';
else
    analysisMode = 'linear';
end

fprintf('\n');
fprintf('===========================================\n');
fprintf('FE model summary\n');
fprintf('===========================================\n');
fprintf('Analysis\n');
fprintf('  type:        %s\n', model.analysis.type);
fprintf('  mode:        %s\n', analysisMode);
fprintf('  load steps:  %i\n', model.analysis.numSteps);
fprintf('  load factor: %s\n', format_load_factors(model.analysis.loadFactor));
fprintf('\n');
fprintf('Element\n');
fprintf('  type:        %s\n', model.element.type);
if model.element.nl
    fprintf('  formulation: nonlinear\n');
else
    fprintf('  formulation: linear\n');
end
fprintf('  backend:     %s\n', model.element.backend);
fprintf('  integration: %s\n', model.element.gauss.rule);
fprintf('  gauss pts:   %i\n', model.element.ngp);
fprintf('\n');
fprintf('Material\n');
fprintf('  name:        %s\n', model.material.name);
fprintf('  condition:   %s\n', model.material.condition);
fprintf('\n');
fprintf('Mesh / DOFs\n');
fprintf('  dimension:   %iD\n', model.info.DIM);
fprintf('  nodes:       %i\n', model.info.NNODE);
fprintf('  elements:    %i\n', model.info.NEL);
fprintf('  DOFs/node:   %i\n', model.info.DOF);
fprintf('  total DOFs:  %i\n', model.info.NDOF);
fprintf('\n');
fprintf('Solver\n');
if model.analysis.nl
    fprintf('  nonlinear:     %s\n', model.solver.type);
    fprintf('  Newton maxit:  %i\n', model.solver.maxIter);
    fprintf('  Newton tol R:  %.4e\n', model.solver.tolR);
    fprintf('  Newton tol U:  %.4e (diagnostic)\n', model.solver.tolU);
    fprintf('  system solver: %s\n', model.solver.linear.type);
    if ~strcmpi(model.solver.linear.type, 'direct')
        fprintf('  PCG tol:       %.4e\n', model.solver.linear.tol);
        fprintf('  PCG maxit:     %i\n', model.solver.linear.maxIter);
    end
else
    fprintf('  analysis:      linear static\n');
    fprintf('  system solver: %s\n', model.solver.linear.type);
    if ~strcmpi(model.solver.linear.type, 'direct')
        fprintf('  PCG tol:       %.4e\n', model.solver.linear.tol);
        fprintf('  PCG maxit:     %i\n', model.solver.linear.maxIter);
    end
end
fprintf('===========================================\n\n');

end

function txt = format_load_factors(loadFactor)

if isempty(loadFactor)
    txt = '[]';
elseif numel(loadFactor) <= 8
    txt = mat2str(loadFactor,4);
else
    txt = sprintf('[%g ... %g] (%i values)', ...
                  loadFactor(1), loadFactor(end), numel(loadFactor));
end

end
