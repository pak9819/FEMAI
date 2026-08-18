function [S, C2, C4] = material_elasticity(mat_e,GradU,MATNAME,MATCOND)
%MATERIAL_ELASTICITY Evaluates material response and tangent stiffness.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Evaluates material response and tangent stiffness.
%
% INPUT
%   mat_e    Element material parameters
%   GradU    Displacement gradient
%   MATNAME  Material model name
%   MATCOND  Material condition
%
% OUTPUT
%   S        Stress tensor
%   C2       Material tangent matrix
%   C4       Fourth-order material tangent tensor
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

coder.varsize('C2');
% S and C tensors

if nargout == 1
   calcC2 = 0;
   calcC4 = 0;
elseif nargout == 2
   calcC2 = 1;
   calcC4 = 0;
elseif nargout == 3
   calcC2 = 1;
   calcC4 = 1;   
end

Emod = mat_e(1);
nu = mat_e(2);

DIM = size(GradU,1);

if DIM > 1
   lambda = Emod*nu/((1+nu)*(1-2*nu));
   mu	 = Emod/(2*(1+nu));
end

% def. nur für Matlab Coder erforderlich
if  DIM == 1
   C2 = 0;
elseif  DIM == 2
   C2 = zeros(3,3); 
elseif DIM == 3
   C2 = zeros(6,6);
end




%% Hooke 
if strcmpi( MATNAME ,'Hooke') 
         
   if DIM == 1
      C2 = Emod; 
      C4 = C2;
   elseif DIM == 2
      C2 = zeros(3,3);
      % ebener Spannungszustand
      if strcmpi(MATCOND ,'planeStress')
         C2(1,1) = lambda/(nu-1) + 2*mu + 2*lambda;
         C2(1,2) = lambda/(nu-1) + 2*lambda;
         C2(1,3) = 0;
         C2(2,1) = lambda/(nu-1) + 2*lambda;
         C2(2,2) = lambda/(nu-1) + 2*mu + 2*lambda;
         C2(2,3) = 0;
         C2(3,1) = 0;
         C2(3,2) = 0;
         C2(3,3) = mu;

      % ebener Verzerrungszustand  
      elseif strcmpi(MATCOND ,'planeStrain')
         C2(1,1) = lambda + 2*mu;
         C2(1,2) = lambda;
         C2(1,3) = 0;
         C2(2,1) = lambda;
         C2(2,2) = lambda + 2*mu;
         C2(2,3) = 0;
         C2(3,1) = 0;
         C2(3,2) = 0;
         C2(3,3) = mu;
      end

   % 3D    
   elseif DIM == 3
      C2 = zeros(6,6); 
      C2(1:3,1:3) = lambda*ones(3,3)+2*mu*eye(3);
      C2(4:6,4:6) = mu*eye(3);
   end
      
  
   if calcC4 == 1
      I2 = eye(DIM);
      I4 = zeros(DIM,DIM,DIM,DIM);
      C4 = zeros(DIM,DIM,DIM,DIM);
      
      for i=1:DIM
         for j=1:DIM
            for k=1:DIM
               for l=1:DIM
                  I4(i,j,k,l) = 0.5*(I2(i,k)*I2(j,l) + I2(i,l)*I2(j,k));
                  C4(i,j,k,l) =  lambda*I2(i,j)*I2(k,l) + 2*(mu)*I4(i,j,k,l);
               end
            end
         end
      end
   
   end

   E = 0.5*(GradU'+GradU);
   Evec = Emat2Evec(E);
   Svec = C2*Evec;
   S = Svec2Smat(Svec);
   
   
%% St. Venant-Kirchhoff-Material    
elseif strcmpi( MATNAME ,'StVenant')
    
   
   if DIM == 1
      C2 = Emod;
      C4 = C2;
   elseif DIM == 2
      C2 = zeros(3,3);
      % ebener Spannungszustand
      if strcmpi(MATCOND ,'planeStress')

         C2(1,1) = lambda/(nu-1) + 2*mu + 2*lambda;
         C2(1,2) = lambda/(nu-1) + 2*lambda;
         C2(1,3) = 0;
         C2(2,1) = lambda/(nu-1) + 2*lambda;
         C2(2,2) = lambda/(nu-1) + 2*mu + 2*lambda;
         C2(2,3) = 0;
         C2(3,1) = 0;
         C2(3,2) = 0;
         C2(3,3) = mu;
         
      % ebener Verzerrungszustand  
      elseif strcmpi(MATCOND ,'planeStrain')
         C2(1,1) = lambda + 2*mu;
         C2(1,2) = lambda;
         C2(1,3) = 0;
         C2(2,1) = lambda;
         C2(2,2) = lambda + 2*mu;
         C2(2,3) = 0;
         C2(3,1) = 0;
         C2(3,2) = 0;
         C2(3,3) = mu;
      end
   % 3D    
   elseif DIM == 3
      C2 = zeros(6,6); 
      C2(1:3,1:3) = lambda*ones(3,3)+2*mu*eye(3);
      C2(4:6,4:6) = mu*eye(3);
   end
    
   % Deformation gradient
   F = GradU + eye(DIM);

   if calcC4 == 1
      % right Cauchy-Green tensor, Cf=F'*F
      %Cf = F'*F;
      Cf_inv = inv(F'*F);
      I2 = eye(DIM);
      I4 = zeros(DIM,DIM,DIM,DIM);
      C4 = zeros(DIM,DIM,DIM,DIM);
      
      for i=1:DIM
         for j=1:DIM
            for k=1:DIM
               for l=1:DIM
                  I4(i,j,k,l) = 0.5*(I2(i,k)*I2(j,l) + I2(i,l)*I2(j,k));
                  C4(i,j,k,l) =  lambda*I2(i,j)*I2(k,l) + 2*(mu)*I4(i,j,k,l);
               end
            end
         end
      end
   
   end
   
   E = 0.5*(F'*F - eye(DIM));
   Evec = Emat2Evec(E);
   Svec = C2*Evec;
   S = Svec2Smat(Svec);
   

%% Neo-Hookean 1
elseif strcmpi(MATNAME ,'NeoHookean1') || strcmpi(MATNAME ,'NeoHooke') 
   
   if DIM == 1 
       F = GradU + 1;
       
       % Im 1D-Fall mit nu = 0 folgt lambda = 0.
       % Die die quadratische Volumendehnenergie 
       % fällt weg, wodurch das Modell zu einem reinen Neo-Hooke-Modell wird.
       S  = (Emod/2) * (1 - 1/F^2);
       C2 = Emod / F^4;

       % Im 1D-Fall entspricht der Materialtensor 4. Ordnung dem skalaren C2
       C4 = C2;
      
       % Physikalische Besonderheit im 1D-Fall: 
       % Aufgrund der der 1D-Reduktion nimmt C2 mit zunehmender Streckung F ab (C2 ~ 1/F^4).
       % Dies führt zu einem "strukturweichenden" Verhalten bei Zug.
       % Im Gegensatz zum linearen Hooke-Gesetz (E = const.) wird dieses hyperelastische 
       % Modell bei großen Dehnungen weicher.


   elseif DIM == 2 || DIM == 3    
      % Deformation gradient
      F = GradU + eye(DIM);
      J = det(F);
      %F_inv = inv(F);
      % right Cauchy-Green tensor, Cf=F'*F
      %Cf = F'*F;
      Cf_inv = inv(F'*F);
      S = mu*eye(DIM) - (mu - lambda*(J^2-J))*Cf_inv;
      

      if calcC2 == 1 || calcC4 == 1
      
         C4 = zeros(DIM,DIM,DIM,DIM);
         V4 = zeros(DIM,DIM,DIM,DIM);
      
         for i=1:DIM
            for j=1:DIM
               for k=1:DIM
                  for l=1:DIM
                     V4(i,j,k,l) = -0.5*(Cf_inv(i,k)*Cf_inv(j,l) + Cf_inv(i,l)*Cf_inv(j,k));
                     C4(i,j,k,l) =  lambda*(2*J^2-J)*Cf_inv(i,j)*Cf_inv(k,l) - 2*(mu - lambda*(J^2-J))*V4(i,j,k,l);
                  end
               end
            end
         end

         C2 = tensor4tomatrix( C4 );

      end
  else
     error('DIM')
  end    
 
   
   
%% Neo-Hookean 2  
elseif strcmpi(MATNAME ,'NeoHookean2')

   if DIM == 1 
       F = GradU + 1;
        
       % Im 1D-Fall mit nu = 0 folgt lambda = 0.
       % Die logarithmische Volumendehnenergie 
       % fällt weg, wodurch das Modell zu einem reinen Neo-Hooke-Modell wird.
       S  = (Emod/2) * (1 - 1/F^2);
       C2 = Emod / F^4;

       % Im 1D-Fall entspricht der Materialtensor 4. Ordnung dem skalaren C2
       C4 = C2;

       % Hinweis: Unter diesen Annahmen ist die Formulierung identisch mit 'NeoHookean1'.

   elseif DIM == 2 || DIM == 3  
      % Deformation gradient
      F = GradU + eye(DIM);
      J = det(F);
      %F_inv = inv(F);
      % right Cauchy-Green tensor, Cf=F'*F
      %Cf = F'*F;
      Cf_inv = inv(F'*F);
      S = mu*(eye(DIM) - Cf_inv') + lambda*log(J)*Cf_inv';
   
   
      if calcC2 == 1 || calcC4 == 1
      
         C4 = zeros(DIM,DIM,DIM,DIM);
         V4 = zeros(DIM,DIM,DIM,DIM);
      
         for i=1:DIM
            for j=1:DIM
               for k=1:DIM
                  for l=1:DIM
                  % ------------------------------------------------------
                  % V4 = dC^-1/dC
                  % ------------------------------------------------------
                  V4(i,j,k,l) = -1/2*( Cf_inv(i,k)*Cf_inv(j,l) + Cf_inv(i,l)*Cf_inv(j,k) );
                  C4(i,j,k,l) = lambda*Cf_inv(j,i)*Cf_inv(l,k) - 2*(mu - lambda*log(J))*V4(i,j,k,l);

                  end
               end
            end
         end

         C2 = tensor4tomatrix( C4 );
      end
      
      
   end
   
     
      
%%
else 
   error('MATNAME');
end


