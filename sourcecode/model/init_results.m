function results = init_results(model)
%INIT_RESULTS Initializes FEM-Solid Edu data structures.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Initializes FEM-Solid Edu data structures.
%
% INPUT
%   model    FE model structure
%
% OUTPUT
%   results  Results structure
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

nSteps = model.analysis.numSteps;

template = struct();

% Loesung
template.U = [];

% Last
template.loadscale = [];

% Systemgroessen
template.K = [];
template.Fext = [];
template.Fint = [];

% Residuum und Lagerreaktionen
template.R = [];
template.Freact = [];

% Solver-Informationen
template.nIter = 0;
template.converged = false;
template.normR = [];
template.normdU = [];

% Postprocessing-Ergebnisse
template.stress.gp = {};
template.stress.node = [];
template.strain.gp = {};
template.strain.node = [];
template.strain.total.gp = {};
template.strain.total.node = [];
template.strain.thermal.gp = {};
template.strain.thermal.node = [];
template.strain.elastic.gp = {};
template.strain.elastic.node = [];
template.vonMises.gp = {};
template.vonMises.node = [];

results = repmat(template, nSteps, 1);

end
