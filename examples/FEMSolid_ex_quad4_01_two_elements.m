%FEMSOLID_EX_QUAD4_01_TWO_ELEMENTS Runs an FEM-Solid Edu example.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Runs an FEM-Solid Edu example.
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

close all; clear; clc;

% ------------------------------------------------------------------------
%% Preprocessing
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
% Hinweise:
% - Linear: Material wird automatisch auf 'Hooke' gesetzt.
% - Nichtlinear: 'StVenant', 'NeoHookean1' oder 'NeoHookean2'.
% - 'planeStress' ist nur fuer 'Hooke' und 'StVenant' verfuegbar.
%
% siehe init_setup.m und readme.md für weitere Optionen
% ------------------------------------------------------------------------

setup = init_setup; 

% ------------------------------------------------------------------------
% Analyse
setup.analysis.nl = false;                % lineare Analyse (default)
setup.analysis.nl = true;                 % nichtlineare Analyse

% ------------------------------------------------------------------------
% Solver
setup.solver.linear.type = 'direct';         % 'direct' (default)
%setup.solver.linear.type = 'pcg';

% ------------------------------------------------------------------------
% Material
setup.material.condition = 'planeStrain'; % 2D: 'planeStrain', 'planeStress'
setup.material.name = 'NeoHookean1';      % nichtlinear: 'StVenant', 'NeoHookean1', 'NeoHookean2'

% ------------------------------------------------------------------------
% Lastschritte und Newton-Verfahren
setup.analysis.numSteps = 1;
%setup.analysis.loadFactor = [0.2 0.8];    % Alternativ können beliebige Lastfaktoren vorgegeben werden
setup.solver.maxIter = 15;

% ------------------------------------------------------------------------
% Netzfeinheit
meshFactor = 1;
numELx = meshFactor*10;
numELy = meshFactor*2;

% ------------------------------------------------------------------------
% FE-Modell erzeugen
model = data_ex_quad4_01_two_elements(setup);


% ------------------------------------------------------------------------
%% Processing
% ------------------------------------------------------------------------

[U, stepResults] = solve_FE(model);

% Fuer das vorhandene Postprocessing wird der letzte Lastschritt verwendet.
results = stepResults(end);


% ------------------------------------------------------------------------
%% Postprocessing
% ------------------------------------------------------------------------

% Ergebnisse (Spannungen & Verzerrungen) berechnen
results = compute_model_results(model, results);

% ------------------------------------------------------------------------
figure('Name','FEM-Solid: mesh','NumberTitle','off');
plot_results(model,'undeformed');
scale = 1.0;
plot_results(model,'support',scale);
plot_results(model,'nodenumber');
plot_results(model,'elementnumber');

% ------------------------------------------------------------------------
figure('Name','FEM-Solid: equivalent nodal forces','NumberTitle','off');
scale = 0.5;
plot_results(model,'undeformed');
plot_results(model,'nodeforcesFext',results,scale);

% ------------------------------------------------------------------------
figure('Name','FEM-Solid: deformed mesh','NumberTitle','off');
scale = 1.0;
plot_results(model,'undeformed');
plot_results(model,'deformed',results,scale);

% % ------------------------------------------------------------------------
% figure('Name','FEM-Solid: strain','NumberTitle','off');
% % Plotten der Verzerrung in X-Richtung (Exx)
% plot_results(model, 'strain_xx', results, scale);

% ------------------------------------------------------------------------
figure('Name','FEM-Solid: Von-Mises stress','NumberTitle','off');
% Plotten der Von-Mises-Spannung
scale = 1.0;
plot_results(model, 'vonMises', results, scale);