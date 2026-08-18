function [Ke, Fbe, Fte, Finte, history] = element_quad4_lin(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_QUAD4_LIN Computes element stiffness, internal forces and loads.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Computes element stiffness, internal forces and loads.
%
% INPUT
%   coord_e  Element nodal coordinates
%   mat_e    Element material parameters
%   b_e      Element volume load vector
%   DeltaT_e Element temperature change
%   Ue       Element displacement vector
%   history  Element history variables
%   gp       Gauss point coordinates
%   w        Gauss weights
%   MATNAME  Material model name
%   MATCOND  Material condition
%   opts     Option structure
%
% OUTPUT
%   Ke       Element stiffness matrix
%   Fbe      Element load vector from volume loads
%   Fte      Element load vector from temperature changes
%   Finte    Element internal force vector
%   history  Element history variables
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

DIM   = size(coord_e,2);     % DIM
DOF   = DIM;                 % Anzahl Freiheihsgrade je Knoten
NNEL  = size(coord_e,1);     % Anzahl Knoten pro Element
NDOFEL = NNEL*DOF;           % Anzahl FHG pro Element

Ke = zeros(NDOFEL,NDOFEL);   % Elementsteifigkeitsmatrix
Fbe = zeros(NDOFEL,1);       % Elementlastvektor infolge Volumenlast b
Fte = zeros(NDOFEL,1);       % Elementlastvektor infolge Temperaturaenderung
Finte = zeros(NDOFEL,1);     % Elementvektor der inneren Kräfte

d = mat_e(3);                % Elementdicke
alphaT = mat_e(4);           % Waermeausdehnungskoeffizient

if isempty(DeltaT_e)
    DeltaT = 0;
else
    DeltaT = DeltaT_e(1);
end
epsT = alphaT*DeltaT*[1; 1; 0];

% ------------------------------------------------------------------------
% Elatizitätsmatrix 
GradU = zeros(2); % wird beim linearen Element nicht benötigt
[~, C] = material_elasticity(mat_e,GradU,MATNAME,MATCOND);

% ------------------------------------------------------------------------
% Summation über Gausspunkte 
numgp = size(gp,1);
     
for i=1:numgp
 
   % Koordinaten der Stützstellen im Referenzelement 
   % r = gp(i,1); s = gp(i,2);
   rst_loc = gp(i,:);

   % Formfunktionen und Ableitungen nach x und y 
   [h , dh_dx, detJ] = shape_quad4(coord_e, rst_loc);
 
   % H-Matrix: Verschiebungsinterpolationsmatrix 
   % H = zeros(2,8);
   % H(1,1:2:7) = h;   
   % H(2,2:2:8) = h;
      
   H = [ h(1)   0        h(2)   0         h(3)   0        h(4)   0
         0        h(1)   0        h(2)    0        h(3)   0        h(4) ];

   % B-Matrix 
   h_x = dh_dx(:,1);
   h_y = dh_dx(:,2);

   B = [ h_x(1)   0        h_x(2)   0         h_x(3)   0        h_x(4)   0
         0        h_y(1)   0        h_y(2)    0        h_y(3)   0        h_y(4)
         h_y(1)   h_x(1)   h_y(2)   h_x(2)    h_y(3)   h_x(3)   h_y(4)   h_x(4) ];
  
   % "gewichtetes Volumenelement" am Integrationspunkt i
   dV = detJ*d*w(i); 

   % Addition des Anteils des aktuellen Gausspunktes zur
   %     Steifigkeitsmatrix
   % Ke = int_Ve = B'CB dV 
   %    = d* int_s int_r B'CB det J dr ds
   %    approx sum_i B'CB dVi    mit dVi = detJ*d*wp
   Ke = Ke +  B'*C*B*dV;
    
   % Addition des Anteils des aktuellen Gausspunktes zum Lastvektor
   Fbe = Fbe + H'*b_e*dV;
   Fte = Fte + B'*C*epsT*dV;

end

% Innere Kräfte 
Finte = Ke*Ue;

% ------------------------------------------------------------------------
end
