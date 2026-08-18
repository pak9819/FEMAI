function model = init_model(coord, elem, mat, bcond, fnode, fvol, DeltaT, setup)
%INIT_MODEL Initializes FEM-Solid Edu data structures.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Initializes FEM-Solid Edu data structures.
%
% INPUT
%   coord    Nodal coordinates
%   elem     Element connectivity
%   mat      Element material table [E, nu, section, alphaT]
%   bcond    Dirichlet boundary conditions
%   fnode    Nodal loads
%   fvol     Volume load data
%   DeltaT   Element temperature changes
%   setup    Setup structure with analysis, material and solver options
%
% OUTPUT
%   model    FE model structure
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

if nargin < 8
    setup = DeltaT;
    DeltaT = [];
end

model.coord = coord;
model.elem  = elem;
model.mat   = mat;
model.bcond = bcond;
model.fnode = fnode;

info.DIM   = size(coord,2);
info.NNODE = size(coord,1);
info.NEL   = size(elem,1);
info.NNEL  = size(elem,2);
info.DOF   = info.DIM;
info.NDOF  = info.DOF * info.NNODE;
info.gdof  = reshape(1:info.NDOF, info.DOF, info.NNODE)';
model.info = info;

if isempty(fvol)
    model.fvol = zeros(info.NEL, info.DOF);
elseif isvector(fvol) && numel(fvol) == info.DOF
    model.fvol = repmat(fvol(:)', info.NEL, 1);
elseif size(fvol,1) == info.NEL && size(fvol,2) == info.DOF
    model.fvol = fvol;
else
    error('fvol muss leer, ein %ix1/1x%i Vektor oder eine %ix%i Tabelle sein.', ...
          info.DOF, info.DOF, info.NEL, info.DOF);
end

if isempty(DeltaT)
    model.DeltaT = zeros(info.NEL,1);
elseif isscalar(DeltaT)
    model.DeltaT = repmat(DeltaT, info.NEL, 1);
elseif isvector(DeltaT) && numel(DeltaT) == info.NEL
    model.DeltaT = DeltaT(:);
elseif size(DeltaT,1) == info.NEL
    model.DeltaT = DeltaT;
else
    error('DeltaT muss leer, skalar oder eine Tabelle mit %i Zeilen sein.', info.NEL);
end

if info.DIM == 1
    if size(model.mat,2) == 2
        model.mat = [model.mat(:,1), NaN(info.NEL,1), model.mat(:,2), zeros(info.NEL,1)];
    elseif size(model.mat,2) == 3
        model.mat = [model.mat(:,1), NaN(info.NEL,1), model.mat(:,2), model.mat(:,3)];
    elseif size(model.mat,2) < 4
        error('1D-Materialdaten muessen [E, A] oder [E, A, alphaT] oder [E, NaN, A, alphaT] sein.');
    else
        model.mat(:,2) = NaN;
    end
elseif info.DIM == 2
    if size(model.mat,2) < 4
        model.mat(:,4) = 0;
    end
elseif info.DIM == 3
    if size(model.mat,2) == 2
        model.mat = [model.mat(:,1), model.mat(:,2), NaN(info.NEL,1), zeros(info.NEL,1)];
    elseif size(model.mat,2) == 3
        model.mat = [model.mat(:,1), model.mat(:,2), NaN(info.NEL,1), model.mat(:,3)];
    elseif size(model.mat,2) < 4
        error('3D-Materialdaten muessen [E, nu] oder [E, nu, alphaT] oder [E, nu, NaN, alphaT] sein.');
    else
        model.mat(:,3) = NaN;
    end
end

if ~setup.analysis.nl                         % Falls bei linearer Analyse ein nichtlineares Material gewählt wurde
    setup.material.name = 'Hooke';
    setup.analysis.numSteps = 1;
    setup.analysis.loadFactor = [];
elseif strcmpi(setup.material.name, 'Hooke')  % Falls bei nichtlineare Analyse 'Hooke' gewählt wurde
    setup.material.name = 'StVenant';
end

if setup.analysis.nl && any(abs(model.DeltaT(:)) > 0)
    warning('init_model:NonlinearTemperatureNotImplemented', ...
        ['Temperaturaenderungen sind derzeit nur fuer lineare Analysen ', ...
         'konsistent implementiert. In nichtlinearen Elementen wird Fte = 0 gesetzt.']);
end

if strcmpi(setup.material.condition, 'plainStress')
    setup.material.condition = 'planeStress';
end

if info.DIM == 1
    setup.material.condition = '1D';
elseif info.DIM == 3
    setup.material.condition = '3D';
end

if info.DIM == 2 && strcmpi(setup.material.condition, 'planeStress') && ...
        ~any(strcmpi(setup.material.name, {'Hooke','StVenant'}))
    warning('init_model:UnsupportedPlaneStress', ...
        ['planeStress ist nur fuer Hooke und StVenant verfuegbar. ', ...
         'Fuer %s wird automatisch planeStrain verwendet.'], setup.material.name);
    setup.material.condition = 'planeStrain';
end

model.physics.type = setup.physics.type;

model.analysis.type = setup.analysis.type;
model.analysis.nl = setup.analysis.nl;
model.analysis.numSteps = setup.analysis.numSteps;
if isempty(setup.analysis.loadFactor)
    model.analysis.loadFactor = linspace(1/model.analysis.numSteps, 1, model.analysis.numSteps);
else
    model.analysis.loadFactor = setup.analysis.loadFactor(:)';
    model.analysis.numSteps = numel(model.analysis.loadFactor);
end

etype = setup.element.type;
backend     = setup.element.backend;
integration = setup.element.integration;
model.element = element_library(etype, model.analysis.nl, backend, integration);

if model.info.NNEL ~= model.element.nnel
    error('elem besitzt %i Knoten pro Element, aber %s erwartet %i.', ...
          model.info.NNEL, model.element.type, model.element.nnel);
end
if model.info.DIM ~= model.element.dim
    error('coord hat Dimension %i, aber %s erwartet Dimension %i.', ...
          model.info.DIM, model.element.type, model.element.dim);
end
if model.info.DOF ~= model.element.ndofNode
    error('DOF je Knoten ist %i, aber %s erwartet %i.', ...
          model.info.DOF, model.element.type, model.element.ndofNode);
end

model.material.name = setup.material.name;
model.material.condition = setup.material.condition;
model.material.columns.E      = 1;
model.material.columns.nu     = 2;
model.material.columns.section = 3;
model.material.columns.A      = 3;
model.material.columns.t      = 3;
model.material.columns.alphaT = 4;

model.solver.type = setup.solver.type;
model.solver.maxIter = setup.solver.maxIter;
model.solver.tolR = setup.solver.tolR;
model.solver.tolU = setup.solver.tolU;
model.solver.verbose = setup.solver.verbose;
model.solver.linear.type = setup.solver.linear.type;
model.solver.linear.tol = setup.solver.linear.tol;
model.solver.linear.maxIter = setup.solver.linear.maxIter;

model.dofs = init_dofs(model);

end
