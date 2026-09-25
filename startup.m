%STARTUP Sets the MATLAB search path for FEMAI.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Sets the MATLAB search path for FEMAI (Deep Learned FEM subset of
%   FEM-Solid Edu: quad4 linear + nonlinear, classic and AI backend).
%
% ------------------------------------------------------------------------

root = fileparts(mfilename('fullpath'));
sourceRoot = fullfile(root, 'sourcecode');

addpath(fullfile(root, 'examples'));
addpath(genpath(sourceRoot));

% Alle Figuren im hellen Design (MATLAB >= R2025a folgt sonst dem System-
% bzw. Batch-Design und zeichnet dunkle Achsen). Aeltere Versionen: ignoriert.
try
    set(groot, 'defaultFigureCreateFcn', @(f, ~) theme(f, 'light'));
catch
end
