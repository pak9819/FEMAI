function [Ke,Fbe,Fte,Finte,history] = element_quad4_nl(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_QUAD4_NL Computes element stiffness, internal forces and loads.
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
%   2026-05-06
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

% ------------------------------------------------------------------------
% Summation über Gausspunkte 
numgp = size(gp,1);

for i=1:numgp

   % Koordinaten der Stützstellen im Referenzelement 
   % r = gp(i,1); s = gp(i,2);
   rst_loc = gp(i,:);

   % Formfunktionen und Ableitungen nach x und y 
   [h , dh_dx, detJ] = shape_quad4(coord_e, rst_loc);

   % Verschiebungsinterpolationsmatrix 
   H = zeros(2,8);
   H(1,1:2:7) = h;   
   H(2,2:2:8) = h;

   % Ableitungen der Formfunktionen  
   h_x = dh_dx(:,1);
   h_y = dh_dx(:,2);

   % "gewichtetes Volumenelement" am Integrationspunkt i
   dV = detJ*w(i)*d;

   % Verschiebungsgradient 
   gradU = compute_gradU(dh_dx, Ue, DIM, DOF);     
   
   % Deformationsgradient 
   F = eye(DIM) + gradU;
   
   % GREEN-LAGRANGEscher Verzerrungstensor 
   E    = 0.5*(F'*F - eye(DIM));
   %Evec = [ E(1,1) ; E(2,2) ; 2*E(1,2) ]; % Vektor für Voigt-Notation (3x1)
      
   % 2. PK Spannung und Elatizitätsmatrix 
   [S, C] = material_elasticity(mat_e,gradU,MATNAME,MATCOND);

   Svec = [S(1,1);S(2,2);S(1,2)]; % Vektor für Voigt-Notation (3x1)
      
   % Volumenkräfte 
   Fbe = Fbe + H'*b_e*dV;       

   % ---------------------------------------------------------------------
   % Schleifen über die Knoten 
   for ni=1:4
      % aktuelle Position
      posni = 2*ni-1:2*ni;

      % Vektor der Formfunktionen am Knoten ni 
      Li = [h_x(ni); h_y(ni)]; 
      
      % Bu-Matrix am Knoten ni 
      % Eu = sym(F'*Grad v) = Bu*v
      Bui = zeros(3,2);
      Bui(1,1) = F(1,1)*h_x(ni);
      Bui(2,1) = F(1,2)*h_y(ni);
      Bui(3,1) = F(1,1)*h_y(ni) + F(1,2)*h_x(ni);
      Bui(1,2) = F(2,1)*h_x(ni);
      Bui(2,2) = F(2,2)*h_y(ni);
      Bui(3,2) = F(2,1)*h_y(ni) + F(2,2)*h_x(ni);

      % Innere Kräfte 
      Finte(posni) = Finte(posni) + Bui'*Svec*dV; 

      % ------------------------------------------------------------------
      for nj=1:4
         % aktuelle Position 
         posnj = 2*nj-1:2*nj;
         
         % Vektor der Formfunktionen am Knoten nj 
         Lj = [h_x(nj); h_y(nj)];
         
         % Bu-Matrix am Knoten nj 
         % Eu = sym(F'*Grad v) = Bu*v
         Buj = zeros(3,2);
         Buj(1,1) = F(1,1)*h_x(nj);
         Buj(2,1) = F(1,2)*h_y(nj);
         Buj(3,1) = F(1,1)*h_y(nj) + F(1,2)*h_x(nj);
         Buj(1,2) = F(2,1)*h_x(nj);
         Buj(2,2) = F(2,2)*h_y(nj);
         Buj(3,2) = F(2,1)*h_y(nj) + F(2,2)*h_x(nj);
                  
         % Tangentiale Steifigkeitsmatrix 
         Kij = ( Bui'*C*Buj + Li'*S*Lj*eye(DIM) )*dV;

         Ke(posni,posnj) = Ke(posni,posnj) + Kij;

      end %nj
   end % ni     
end % ip
        
         

end
      
      
      
      
