function model = data_ex_quad4_02_beam_nel(numELx, numELy, setup)
%DATA_EX_QUAD4_02_BEAM_NEL Creates model input data for an FEM-Solid Edu example.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Creates model input data for an FEM-Solid Edu example.
%
% INPUT
%   numELx   Number of elements in x-direction
%   numELy   Number of elements in y-direction
%   setup    Setup structure with analysis, material and solver options
%
% OUTPUT
%   model    FE model structure
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-05-07
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

if nargin < 1 || isempty(numELx)
    numELx = 10;       % Anzahl Elemente in X-Richtung
end

if nargin < 2 || isempty(numELy)
    numELy = 1;        % Anzahl Elemente in Y-Richtung
end

if nargin < 3 || isempty(setup)
    setup = init_setup;
end

setup.element.type = 'quad4';

% ------------------------------------------------------------------------
% Abmessungen und Materialparameter

Lx = 10;           % Breite der Scheibe
Ly = 1;            % Höhe 

Emod = 1000;       % E-Modul
nue = 0.3;         % Querdehnzahl
thickness = 1;     % Dicke              
alphaT = 0;        % Waermeausdehnungskoeffizient

% ------------------------------------------------------------------------
% Volumenlast 
% z.B. Eigengewicht

fvol = [0;-0.5];
%fvol = [0;0];

% ------------------------------------------------------------------------
% Temperaturaenderung

DeltaT = [];

% ------------------------------------------------------------------------
% Randbedinungen 

% Definition of Sides for Boundary Conditions ----------------------------    
%       
%            3                 
%       ----------           1: bottom
%       |        |           2: right
%    4  |        | 2         3: top
%       |        |           4: left
%       ----------
%           1
%
%       e.g. DirichletBcond = [  0  1  0  1 ];  
% 
%                  -->  only fixed on left and right side
%
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% feste Lagerung an Rand 4
DirichletBcond = [ 0 0 0 1 ];     % Dirichlet BC 


% ------------------------------------------------------------------------
% konstante Linenlast q auf Kante 2 in x-Richtung
% q2 = 1;
%q2 = 0;
% NeumannBcond   = [ 0 5 0 0 ];     % Neumann BC


% konstante Linenlast q auf Kante 3 in y-Richtung
q3 = -1;
%q3 = 0;
NeumannBcond   = [ 0 0 q3 0 ];     % Neumann BC

% ------------------------------------------------------------------------
% FE-Modell erzeugen 

% Materialdaten: [E, nu, section, alphaT]
% 2D: section = thickness
matcard = [Emod  nue  thickness  alphaT];

[ coord , elem , bcond , mat , fnode ] = create_model_data_rectangle( Lx , Ly , numELx , numELy , ...
                                                       DirichletBcond , NeumannBcond , matcard);
                                                    
% ------------------------------------------------------------------------
% Zusätzliche Knotenlasten hinzufügen 

% z.B. Knotenlast oben rechts
% kn = size(coord,1);
% fnode = [fnode
%          kn 2 -10];
     

% ------------------------------------------------------------------------
% FE-Model Datenstruktur erzeugen
model = init_model(coord, elem, mat, bcond, fnode, fvol, DeltaT, setup); 

         

end
