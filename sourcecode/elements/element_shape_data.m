function [h, dh_dx, detJ] = element_shape_data(element, coord_e, rst_loc)
%ELEMENT_SHAPE_DATA Evaluates shape functions and global derivatives.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Evaluates shape functions, derivatives with respect to global
%   coordinates and the integration mapping determinant for element types in
%   the element library.
%
% INPUT
%   element  Element metadata structure from element_library
%   coord_e  Element nodal coordinates
%   rst_loc  Local integration point coordinates
%
% OUTPUT
%   h        Shape function vector
%   dh_dx    Shape function derivatives with respect to global coordinates
%   detJ     Integration mapping determinant
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

switch lower(element.type)
    case 'bar2'
        [h, dh_dx, detJ] = shape_bar2(coord_e, rst_loc);

    case {'truss2d','truss3d'}
        error(['Shape data for "%s" is not implemented yet. ', ...
               'Implement a matching shape function first.'], ...
              element.type);

    case 'tria3'
        error(['Shape data for "tria3" is not implemented yet. ', ...
               'Implement shape_tria3 first.']);

    case 'quad4'
        [h, dh_dx, detJ] = shape_quad4(coord_e, rst_loc);

    case 'tetra4'
        error(['Shape data for "tetra4" is not implemented yet. ', ...
               'Implement shape_tetra4 first.']);

    case 'brick8'
        [h, dh_dx, detJ] = shape_brick8(coord_e, rst_loc);

    otherwise
        error('Shape data is not implemented for element type "%s".', ...
              element.type);
end

end
