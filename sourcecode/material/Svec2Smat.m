function Smat = Svec2Smat(Svec)
%SVEC2SMAT Converts a stress vector to a stress tensor matrix.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Converts a stress vector in Voigt notation to its tensor matrix
%   representation.
%
% INPUT
%   Svec     Stress vector in Voigt notation
%
% OUTPUT
%   Smat     Stress tensor matrix
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

if isscalar(Svec)
    Smat = Svec(1,1);
elseif numel(Svec) == 3
    Smat = [Svec(1) Svec(3)
            Svec(3) Svec(2)];
elseif numel(Svec) == 6
    Smat = [Svec(1) Svec(4) Svec(6)
            Svec(4) Svec(2) Svec(5)
            Svec(6) Svec(5) Svec(3)];
else
    error('Unknown size');
end

end
