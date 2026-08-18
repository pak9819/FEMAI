function setup = init_setup()
%INIT_SETUP Initializes FEM-Solid Edu data structures.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Initializes FEM-Solid Edu data structures.
%
% INPUT
%   none
%
% OUTPUT
%   setup    Setup structure with analysis, material and solver options
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

setup.physics.type = 'elasticity';
%   'poisson'
%   'elasticity'

% -------------------------------------------------------------------------
% Element
% -------------------------------------------------------------------------
setup.element.type = 'quad4';
% 
%   1D:
%       'bar2'
%
%   2D:
%       'truss2D'
%       'tria3'
%       'quad4'
%
%   3D:
%       'truss3D'
%       'tetra4'
%       'brick8'

setup.element.integration = 'default';
% abhaengig vom Element:
%   'default'
%   'full'
%   'reduced'
%   '1point'
%   '2point'
%   '1x1'
%   '2x2'
%   '2x2x2'

setup.element.backend = 'matlab';
% Auswahl, welche Element-Version verwendet wird
%
%   'matlab'
%   'vectorized'
%   'mex'
%   'AI'

% -------------------------------------------------------------------------
% Material
% -------------------------------------------------------------------------
setup.material.name = 'StVenant';
%   'Hooke'
%   'StVenant'
%   'NeoHookean1'
%   'NeoHookean2'


setup.material.condition = 'planeStrain';
%   'planeStrain'
%   'planeStress'
%   '3D'

% -------------------------------------------------------------------------
% Analyse
% -------------------------------------------------------------------------
setup.analysis.type = 'static';
%   'static'
%   'transient'  (nicht in Edu-Version)

setup.analysis.nl = false;
%   false   lineares Problem
%   true    nichtlineares Problem

% Anzahl Lastschritte
setup.analysis.numSteps = 1;

% Lastfaktoren der einzelnen Lastschritte
% [] bedeutet: automatisch gleichmaessig von 1/numSteps bis 1
setup.analysis.loadFactor = [];



% -------------------------------------------------------------------------
% Solver
% -------------------------------------------------------------------------
setup.solver.type = 'newton';
%   'newton'
%   'modified_newton'  (nicht in Edu-Version)

% maximale Newton-Iterationen pro Lastschritt
setup.solver.maxIter = 20;

% Toleranz fuer das Residuum
setup.solver.tolR = 1e-8;

% Toleranz fuer das Verschiebungsinkrement
% derzeit nur Diagnosewert; Konvergenz wird ueber tolR entschieden
setup.solver.tolU = 1e-10;

% Ausgabe steuern
setup.solver.verbose = true;
%   true    Iterationsausgabe aktiv
%   false   keine Iterationsausgabe


% -------------------------------------------------------------------------
% Linearer Gleichungsloeser
% -------------------------------------------------------------------------
setup.solver.linear.type = 'direct';
%   'direct'
%   'pcg'

% Toleranz fuer iterative lineare Loeser
setup.solver.linear.tol = 1e-12;

% maximale Iterationen fuer iterative lineare Loeser
setup.solver.linear.maxIter = 500;


end
