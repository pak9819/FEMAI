function C2 = tensor4tomatrix( C4 )
%TENSOR4TOMATRIX Converts a fourth-order tensor to a Voigt matrix.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Converts a fourth-order tensor to its matrix representation 
%   in Voigt notation.
%
% INPUT
%   C4       Fourth-order material tangent tensor
%
% OUTPUT
%   C2       Material tangent matrix in Voigt notation
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

a = size(C4);
DIM=a(1);

% ------------------------------------------------------------------------
% 3D
if DIM==3
   C2 = [ C4(1,1,1,1)  C4(1,1,2,2)  C4(1,1,3,3)  C4(1,1,1,2)  C4(1,1,2,3)  C4(1,1,1,3)
          C4(2,2,1,1)  C4(2,2,2,2)  C4(2,2,3,3)  C4(2,2,1,2)  C4(2,2,2,3)  C4(2,2,1,3) 
          C4(3,3,1,1)  C4(3,3,2,2)  C4(3,3,3,3)  C4(3,3,1,2)  C4(3,3,2,3)  C4(3,3,1,3) 
          C4(1,2,1,1)  C4(1,2,2,2)  C4(1,2,3,3)  C4(1,2,1,2)  C4(1,2,2,3)  C4(1,2,1,3)
          C4(2,3,1,1)  C4(2,3,2,2)  C4(2,3,3,3)  C4(2,3,1,2)  C4(2,3,2,3)  C4(2,3,1,3)
          C4(1,3,1,1)  C4(1,3,2,2)  C4(1,3,3,3)  C4(1,3,1,2)  C4(1,3,2,3)  C4(1,3,1,3) ];
% 2D
elseif DIM == 2
   C2 = [ C4(1,1,1,1)  C4(1,1,2,2)    C4(1,1,1,2)      
          C4(2,2,1,1)  C4(2,2,2,2)    C4(2,2,1,2)      
          C4(1,2,1,1)  C4(1,2,2,2)    C4(1,2,1,2)  ]; 

% 1D       
elseif DIM == 1       
   C2 =  C4(1,1,1,1); 

else
   error('error in tensor4tomatrix.m! unknown ndim')
   
end


% --- end ----------------------------------------------------------------
end
