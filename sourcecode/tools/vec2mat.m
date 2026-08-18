function Umat = vec2mat(Uvec,DOF)
%VEC2MAT Converts a nodal vector into a nodal value matrix.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Converts a nodal vector into a nodal value matrix.
%
% INPUT
%   Uvec     Global nodal vector
%   DOF      Degrees of freedom per node
%
% OUTPUT
%   Umat     Nodal value matrix
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

if isempty(Uvec)
    Umat = [];
    return;
end

% Konsistenzcheck
n = numel(Uvec);
if mod(n, DOF) ~= 0
    error('Laenge von Uvec (%d) ist kein Vielfaches von DOF (%d).', n, DOF);
end

NNODE = n / DOF;

% Umformung des Vektors Uvec in eine Matrix Umat mit DOF Spalten:
% reshape erzeugt DOF Zeilen, dann transponieren => NNODE x DOF
Umat = reshape(Uvec, DOF, NNODE).';


end

