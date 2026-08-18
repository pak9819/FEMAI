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
