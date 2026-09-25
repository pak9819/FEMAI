function [netFile, matKey] = brick8_nl_ai_network_file(MATNAME)
%BRICK8_NL_AI_NETWORK_FILE Netzdatei des nichtlinearen KI-brick8 je Material.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Jedes Materialgesetz hat ein eigenes, fest eintrainiertes Energienetz:
%
%     StVenant             -> brick8_nl_W_network.mat
%     NeoHooke/NeoHookean1 -> brick8_nl_W_network_NeoHookean1.mat (geplant)
%
%   BRICK8_NET_FILE=<name> ueberschreibt den Dateinamen (relativ zum
%   Elementordner oder absolut), z.B. fuer Architekturvergleiche. Das Netz
%   wird persistent gecacht -- nach einem Wechsel "clear all".
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

switch lower(strtrim(char(MATNAME)))
    case 'stvenant'
        matKey = 'StVenant';
        name   = 'brick8_nl_W_network.mat';
    case {'neohooke', 'neohookean1'}
        matKey = 'NeoHookean1';
        name   = 'brick8_nl_W_network_NeoHookean1.mat';
    otherwise
        error('brick8_nl_ai_network_file:Material', ...
            'Fuer Material "%s" gibt es kein nichtlineares KI-brick8-Netz.', MATNAME);
end

override = strtrim(getenv('BRICK8_NET_FILE'));
if ~isempty(override)
    name = override;
end
if isempty(fileparts(name))
    netFile = fullfile(fileparts(mfilename('fullpath')), name);
else
    netFile = name;
end
end
