function element = element_library(etype, isNonlinear, backend, integration)
%ELEMENT_LIBRARY Returns element metadata and integration data.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Returns element metadata, Gauss integration data and the selected
%   element routine for a given element type.
%
% INPUT
%   etype        Element type name
%   isNonlinear  Flag for nonlinear element formulation
%   backend      Element routine backend name
%   integration  Integration rule name
%
% OUTPUT
%   element      Element data structure
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

if nargin < 2 || isempty(isNonlinear)
    isNonlinear = false;
end
if nargin < 3 || isempty(backend)
    backend = 'matlab';
end
if nargin < 4 || isempty(integration)
    integration = 'default';
end

etype = lower(etype);
etypeOriginal = etype;
hasFormulationSuffix = false;
suffixIsNonlinear = false;
if endsWith(etype, 'lin')
    etype = extractBefore(etype, 'lin');
    hasFormulationSuffix = true;
    suffixIsNonlinear = false;
elseif endsWith(etype, 'nl')
    etype = extractBefore(etype, 'nl');
    hasFormulationSuffix = true;
    suffixIsNonlinear = true;
end
etype = char(etype);

if hasFormulationSuffix && suffixIsNonlinear ~= isNonlinear
    error(['Element type "%s" conflicts with the selected analysis. ', ...
           'Use setup.element.type without lin/nl suffix and control ', ...
           'the formulation with setup.analysis.nl.'], ...
          etypeOriginal);
end

% ------------------------------------------------------------------------
switch etype
    case 'bar2'
        element.name = '2-node bar'; element.dim = 1; element.nnel = 2; element.ndofNode = 1;
    case 'truss2d'
        element.name = '2D truss'; element.dim = 2; element.nnel = 2; element.ndofNode = 2;
    case 'tria3'
        element.name = '3-node triangle'; element.dim = 2; element.nnel = 3; element.ndofNode = 2;
    case 'quad4'
        element.name = '4-node quadrilateral'; element.dim = 2; element.nnel = 4; element.ndofNode = 2;
    case 'truss3d'
        element.name = '3D truss'; element.dim = 3; element.nnel = 2; element.ndofNode = 3;
    case 'tetra4'
        element.name = '4-node tetrahedron'; element.dim = 3; element.nnel = 4; element.ndofNode = 3;
    case 'brick8'
        element.name = '8-node brick'; element.dim = 3; element.nnel = 8; element.ndofNode = 3;
    otherwise
        error('Unknown element type: %s', etype);
end


% ------------------------------------------------------------------------
element.type = etype;
element.nl = isNonlinear;
element.backend = backend;
element.ndofElement = element.nnel * element.ndofNode;
element.gauss = gauss_library(element.type, integration);
element.gp  = element.gauss.gp;
element.w   = element.gauss.w;
element.ngp = element.gauss.ngp;
element.routine = element_routine(element);


% Platzhalter fuer spaetere elementbezogene Optionen, z.B.
% Stabilisierung, Hourglass-Control oder spezielle Materialparameter,
% die dann an die Elementroutine übergeben werden.
element.opts = struct();



end
