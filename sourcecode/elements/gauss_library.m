function gauss = gauss_library(etype, integration)
%GAUSS_LIBRARY Returns library data for FEM-Solid Edu.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Returns library data for FEM-Solid Edu.
%
% INPUT
%   etype    Element type name
%   integration Integration rule name
%
% OUTPUT
%   gauss    Gauss integration data structure
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

if nargin < 2 || isempty(integration)
    integration = 'default';
end

etype = lower(etype);
integration = lower(integration);

% ------------------------------------------------------------------------
switch etype

    % --------------------------------------------------------------------
    case {'bar2','truss2d','truss3d'}
        if strcmp(integration,'default')
            integration = '2point';
        end

        switch integration
            case '1point'
                gp = 0;
                w  = 2;

            case '2point'
                a = 1/sqrt(3);
                gp = [-a; a];
                w  = [1; 1];

            otherwise
                error('Unknown integration for %s: %s', etype, integration);
        end

    % --------------------------------------------------------------------
    case 'tria3'
        if strcmp(integration,'default')
            integration = '1point';
        end

        switch integration
            case '1point'
                gp = [1/3  1/3];
                w  = 1/2;

            otherwise
                error('Unknown integration for tria3: %s', integration);
        end

    % --------------------------------------------------------------------
    case 'quad4'
        if strcmp(integration,'default')
            integration = '2x2';
        end

        switch integration
            case {'1point','1x1','reduced'}
                gp = [0 0];
                w  = 4;

            case {'2x2','full'}
                a = 1/sqrt(3);
                gp = [ -a -a
                        a -a
                        a  a
                       -a  a ];
                w = [1; 1; 1; 1];

            otherwise
                error('Unknown integration for quad4: %s', integration);
        end

    % --------------------------------------------------------------------
    case 'tetra4'
        if strcmp(integration,'default')
            integration = '1point';
        end

        switch integration
            case '1point'
                gp = [1/4  1/4  1/4];
                w  = 1/6;

            otherwise
                error('Unknown integration for tetra4: %s', integration);
        end

    % --------------------------------------------------------------------
    case 'brick8'
        if strcmp(integration,'default')
            integration = '2x2x2';
        end

        switch integration
            case {'1point','1x1x1','reduced'}
                gp = [0 0 0];
                w  = 8;

            case {'2x2x2','full'}
                a = 1/sqrt(3);
                gp = [ -a -a -a
                        a -a -a
                        a  a -a
                       -a  a -a
                       -a -a  a
                        a -a  a
                        a  a  a
                       -a  a  a ];
                w = ones(8,1);

            otherwise
                error('Unknown integration for brick8: %s', integration);
        end

    % --------------------------------------------------------------------
    otherwise
        error('Unknown element type: %s', etype);
end


% ------------------------------------------------------------------------
gauss.gp = gp;
gauss.w  = w;
gauss.ngp = size(gp,1);
gauss.rule = integration;

end
