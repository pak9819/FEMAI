function Svec = Smat2Svec(Smat)
%SMAT2SVEC Converts a stress tensor matrix to a stress vector.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Converts a stress tensor matrix to its vector representation in Voigt
%   notation.
%
% INPUT
%   Smat     Stress tensor matrix
%
% OUTPUT
%   Svec     Stress vector in Voigt notation
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

if size(Smat,1) == 1
   Svec = Smat(1,1);
elseif size(Smat,1) == 2
   Svec = [ Smat(1,1); Smat(2,2); Smat(1,2) ];
elseif size(Smat,1) == 3
   Svec = [ Smat(1,1); Smat(2,2); Smat(3,3) ; Smat(1,2); Smat(2,3); Smat(3,1) ];
else
   error('Unknown size');
end


