function Evec = Emat2Evec(Emat)
%EMAT2EVEC Converts a strain tensor matrix to a Voigt strain vector.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Converts a strain tensor matrix to its vector representation in Voigt
%   notation.
%
%   The shear components are stored as gamma_ij = 2*E_ij. This makes the
%   strain vector energetically conjugate to the stress vector, i.e.
%
%       S:E = Svec' * Evec
%
% INPUT
%   Emat     Strain tensor matrix
%
% OUTPUT
%   Evec     Voigt strain vector
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

if size(Emat,1) == 1
   Evec = Emat(1,1);
elseif size(Emat,1) == 2
   Evec = [ Emat(1,1); Emat(2,2); 2*Emat(1,2) ];
elseif size(Emat,1) == 3
   Evec = [ Emat(1,1); Emat(2,2); Emat(3,3) ; 2*Emat(1,2); 2*Emat(2,3); 2*Emat(3,1) ];
else
   error('Unknown size');
end


