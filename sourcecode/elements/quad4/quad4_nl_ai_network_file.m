function [netFile, matKey] = quad4_nl_ai_network_file(MATNAME)
%QUAD4_NL_AI_NETWORK_FILE Netzdatei des nichtlinearen KI-quad4 je Material.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Jedes Materialgesetz hat ein eigenes, fest eintrainiertes Energienetz:
%
%     StVenant             -> quad4_nl_W_network.mat
%     NeoHooke/NeoHookean1 -> quad4_nl_W_network_NeoHookean1.mat
%
%   'NeoHooke' ist in material_elasticity.m ein Alias fuer 'NeoHookean1'
%   und wird deshalb auf denselben Schluessel abgebildet.
%
% INPUT
%   MATNAME  Materialname aus dem Modell
%
% OUTPUT
%   netFile  voller Pfad der Netzdatei (muss nicht existieren)
%   matKey   normierter Materialname, wie er im Netz als 'material' steht
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-17
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

switch lower(strtrim(char(MATNAME)))
    case 'stvenant'
        matKey = 'StVenant';
        name   = 'quad4_nl_W_network.mat';
    case {'neohooke', 'neohookean1'}
        matKey = 'NeoHookean1';
        name   = 'quad4_nl_W_network_NeoHookean1.mat';
    otherwise
        error('quad4_nl_ai_network_file:Material', ...
            'Fuer Material "%s" gibt es kein nichtlineares KI-Netz.', MATNAME);
end

netFile = fullfile(fileparts(mfilename('fullpath')), name);
end
