function [ h , dh_dx, detJ ] = shape_quad4(coord_e,rst_loc)
%SHAPE_QUAD4 Evaluates shape functions and derivatives.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Shape functions and derivatives of the shape functions for a 4-node
%   bilinear isoparametric quadrilateral element.
%
% INPUT
%   coord_e  Global x/y coordinates of the element nodes
%   rst_loc  Local coordinates in the reference element
%            2D: rst_loc = [r, s]
%
% OUTPUT
%   h        Vector of shape functions (4x1)
%            h = [h1 h2 h3 h4]^T
%
%   dh_dx    Matrix with derivatives of the shape functions with respect
%            to x = [x, y]
%            2D: dh_dx = [h_x, h_y]  (4x2)
%
%            h_x = dh/dx = [dh1/dx dh2/dx ...]^T
%            h_y = dh/dy = [dh1/dy dh2/dy ...]^T
%
%   detJ     Jacobian determinant
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

r = rst_loc(1);
s = rst_loc(2);

% ------------------------------------------------------------------------
% globale Knotenkoordinaten x,y 
x = coord_e(:,1);
y = coord_e(:,2);

% ------------------------------------------------------------------------
% Formfunktionen 
h = zeros(4,1);
h(1) = 1/4*(1-r)*(1-s);
h(2) = 1/4*(1+r)*(1-s);
h(3) = 1/4*(1+r)*(1+s);
h(4) = 1/4*(1-r)*(1+s);
  
% ------------------------------------------------------------------------
% Ableitungen der Formfunktionen nach r und s 
h_r = zeros(4,1);
h_s = zeros(4,1);

h_r(1) =  (s - 1)/4;
h_r(2) = -(s - 1)/4;
h_r(3) =  (s + 1)/4; 
h_r(4) = -(s + 1)/4;

h_s(1) =  (r - 1)/4;
h_s(2) = -(r + 1)/4;
h_s(3) =  (r + 1)/4;
h_s(4) = -(r - 1)/4;

dh_dr = [h_r , h_s];

% ------------------------------------------------------------------------
% Ableitungen der Koordinaten nach r und s (Jacobi-Matrix)
% J = [ x_r  x_s
%       y_r  y_s ]

x_r = x'*h_r;
x_s = x'*h_s;
y_r = y'*h_r;
y_s = y'*h_s;
  
% ------------------------------------------------------------------------
% Jacobideterminante 
detJ = x_r* y_s - y_r* x_s;

% ------------------------------------------------------------------------
% Ableitungen der Formfunktionen nach x und y 
h_x = ( y_s*h_r - y_r*h_s)/ detJ; 
h_y = (-x_s*h_r + x_r*h_s)/ detJ;
      
dh_dx = [h_x , h_y];

% ------------------------------------------------------------------------
end      
      
      

