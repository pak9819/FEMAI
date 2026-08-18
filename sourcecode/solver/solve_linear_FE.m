function [U,results] = solve_linear_FE(model)
%SOLVE_LINEAR_FE Solves a linear static FE problem.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Solves the linear static FE problem with prescribed displacement
%   boundary conditions.
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

[K,Fext] = assemble(model);

dofsfree = model.dofs.free;
dofsfix = model.dofs.fixed;

U = zeros(model.info.NDOF,1);

U(dofsfix) = model.dofs.fixedValues; % vorgegebene Verschiebungen

% ------------------------------------------------------------------------
% Partitionierung des Gleichungssystems
%
% partitioniertes System
%  [ Kaa Kab   [ Ua   = [ Fa
%    Kba Kbb ]   Ub ]     Fb ]

Kaa = K(dofsfree,dofsfree);
%Kbb = K(dofsfix,dofsfix);
Kab = K(dofsfree,dofsfix);
%Kba = K(dofsfix,dofsfree);   % = Kab', da K symmetrisch

Fa = Fext(dofsfree);         % bekannte Kraefte
%Fb = Fext(dofsfix);          % unbekannte Kraefte

%Ua = U(dofsfree);           % unbekannte Verschiebungen
Ub = U(dofsfix);              % bekannte (vorgegebene) Verschiebungen

% ------------------------------------------------------------------------
% Gleichungssystem loesen
% Reduziertes Gleichungssystem:   Kaa*Ua + Kab*Ub = Fa

% Bestimmung der unbekannten Verschiebungen Ua 
% aus dem LGS:  Kaa*Ua = Fa - Kab*Ub
Ua = solve_system(model, Kaa, Fa - Kab*Ub);

% Gesamtverschiebungsvektor
U(dofsfree) = Ua;

% ------------------------------------------------------------------------
% Innere Kräfte
Fint = K*U;

% ------------------------------------------------------------------------
% Residuum
R = Fint - Fext;

% ------------------------------------------------------------------------
% Lagerkräfte (nur an FHG mit Verschiebungs-Randbedingungen)
Freact = compute_reaction_forces(model, Fint, Fext);

% ------------------------------------------------------------------------
results = init_results(model);
results = results(1);

results.U = U;
results.loadscale = 1.0;
results.K = K;
results.Fext = Fext;
results.Fint = Fint;
results.R = R;
results.Freact = Freact;
results.nIter = 1;
results.converged = true;
results.normR = norm(R(dofsfree),2);
results.normdU = norm(U(model.dofs.free),2);


% ------------------------------------------------------------------------
end
