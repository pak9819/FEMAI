function model = data_ex_quad4_01_two_elements(setup)
%DATA_EX_QUAD4_01_TWO_ELEMENTS Creates model input data for an FEM-Solid Edu example.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Creates model input data for an FEM-Solid Edu example.
%
% INPUT
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

if nargin < 1 || isempty(setup)
    setup = init_setup;
end

setup.element.type = 'quad4';


% ------------------------------------------------------------------------
% Netz
coord = [ 0  0
          1  0
          2  0
          0  1
          1  1
          2  1 ];

elem  = [ 1  2  5  4
          2  3  6  5 ];

% ------------------------------------------------------------------------
% Materialdaten: [E, nu, section, alphaT]
% 2D: section = thickness
mat = [ 400  0.3  1  0
        400  0.3  1  0 ];

% ------------------------------------------------------------------------
% Knotenlasten
fnode = [ 6  2  -5 ];

% ------------------------------------------------------------------------
% Volumenlast, z.B. aus Eigengewicht
fvol = [0; -2];

% ------------------------------------------------------------------------
% Temperaturaenderung
DeltaT = [];

% ------------------------------------------------------------------------
% Randbedingungen am Dirichlet-Rand
bcond = [ 1  1 0
          1  2 0
          4  1 0
          4  2 0];

% z.B. mit vorgegebener Verschiebung am Knoten 3
bcond = [ 1  1 0
          1  2 0
          4  1 0
          4  2 0
          3  1 0.5];

% ------------------------------------------------------------------------
model = init_model(coord, elem, mat, bcond, fnode, fvol, DeltaT, setup);

end
