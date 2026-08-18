function [coord,elem,bcond,mat,fnode] = create_model_data_rectangle(Lx,Ly,numELx,numELy,DirichletBcond,NeumannBcond,matcard)
%CREATE_MODEL_DATA_RECTANGLE Creates mesh and boundary data for an FEM-Solid Edu model.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Creates mesh and boundary data for an FEM-Solid Edu model.
%
% INPUT
%   Lx       Model length in x-direction
%   Ly       Model length in y-direction
%   numELx   Number of elements in x-direction
%   numELy   Number of elements in y-direction
%   DirichletBcond Boundary flags for Dirichlet conditions
%   NeumannBcond Boundary loads for Neumann conditions
%   matcard  Material parameter card [E, nu, section, alphaT]
%
% OUTPUT
%   coord    Nodal coordinates
%   elem     Element connectivity
%   bcond    Dirichlet boundary conditions
%   mat      Element material table
%   fnode    Nodal loads
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

nnodes = (numELx+1)*(numELy+1);
nelements = numELx*numELy;

x = linspace(0, Lx, numELx+1);
y = linspace(0, Ly, numELy+1);

coord = [repmat(x(:), numELy+1, 1), ...
         repelem(y(:), numELx+1, 1)];

n1 = (1:numELx).' + (0:numELy-1)*(numELx+1);
n1 = n1(:);
elem = [n1, n1+1, n1+numELx+2, n1+numELx+1];

% ------------------------------------------------------------------------
% Randknoten
% ------------------------------------------------------------------------
sideNodes = cell(4,1);
sideNodes{1} = (1:numELx+1).';                         % bottom
sideNodes{2} = (numELx+1:numELx+1:nnodes).';           % right
sideNodes{3} = (numELy*(numELx+1)+1:nnodes).';         % top
sideNodes{4} = (1:numELx+1:nnodes).';                  % left

% ------------------------------------------------------------------------
% Dirichlet-Randbedingungen
% ------------------------------------------------------------------------
bcond = [];
for side = 1:4
    if DirichletBcond(side) == 1
        nodes = sideNodes{side};
        bcondSide = [nodes, ones(numel(nodes),1), zeros(numel(nodes),1)
                     nodes, 2*ones(numel(nodes),1), zeros(numel(nodes),1)];
        bcond = [bcond; bcondSide]; 
    end
end

if ~isempty(bcond)
    bcond = unique(bcond,'rows');
end

% ------------------------------------------------------------------------
% Neumann-Randbedingungen als aequivalente Knotenlasten
% ------------------------------------------------------------------------
fnode = [];
edgeLength = [Lx/numELx, Ly/numELy, Lx/numELx, Ly/numELy];
loadDirection = [2, 1, 2, 1];

for side = 1:4
    lineload = NeumannBcond(side);
    if lineload ~= 0
        nodes = sideNodes{side};
        nload = numel(nodes);
        values = lineload * edgeLength(side) * ones(nload,1);
        values([1,end]) = 0.5 * values([1,end]);

        fnodeSide = [nodes, ...
                     loadDirection(side)*ones(nload,1), ...
                     values];
        fnode = [fnode; fnodeSide]; 
    end
end

if ~isempty(fnode)
    [nodeDir,~,idx] = unique(fnode(:,1:2),'rows');
    values = accumarray(idx, fnode(:,3));
    fnode = [nodeDir, values];
end

% ------------------------------------------------------------------------
% Material
% ------------------------------------------------------------------------
mat = repmat(matcard, nelements, 1);

end
