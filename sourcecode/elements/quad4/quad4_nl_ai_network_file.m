function [netFile, matKey, stateForm] = quad4_nl_ai_network_file(MATNAME)
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
%   Zustandsform (Umgebungsvariablen, Standard = Ko-Rotation, unveraendert):
%     QUAD4_STATE_FORM=gram  -> Gram-/Metrik-Netz quad4_nl_W_network_gram.mat
%                               (StVenant; Kette quad4_nl_ai_energy_gram.m)
%     QUAD4_NET_FILE=<name>  -> ueberschreibt den Dateinamen (relativ zum
%                               Elementordner oder absolut), z.B. fuer
%                               Architekturvergleiche.
%   Die Wahl wird in quad4_nl_ai_energy persistent gecacht -- nach einem
%   Wechsel "clear all" ausfuehren.
%
% INPUT
%   MATNAME  Materialname aus dem Modell
%
% OUTPUT
%   netFile   voller Pfad der Netzdatei (muss nicht existieren)
%   matKey    normierter Materialname, wie er im Netz als 'material' steht
%   stateForm 'corotation' oder 'gram'
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

stateForm = 'corotation';
if strcmpi(strtrim(getenv('QUAD4_STATE_FORM')), 'gram')
    stateForm = 'gram';
end

switch lower(strtrim(char(MATNAME)))
    case 'stvenant'
        matKey = 'StVenant';
        if strcmp(stateForm, 'gram')
            name = 'quad4_nl_W_network_gram.mat';
        else
            name = 'quad4_nl_W_network.mat';
        end
    case {'neohooke', 'neohookean1'}
        matKey = 'NeoHookean1';
        if strcmp(stateForm, 'gram')
            name = 'quad4_nl_W_network_gram_NeoHookean1.mat';
        else
            name = 'quad4_nl_W_network_NeoHookean1.mat';
        end
    otherwise
        error('quad4_nl_ai_network_file:Material', ...
            'Fuer Material "%s" gibt es kein nichtlineares KI-Netz.', MATNAME);
end

override = strtrim(getenv('QUAD4_NET_FILE'));
if ~isempty(override)
    name = override;
end

if isempty(fileparts(name))
    netFile = fullfile(fileparts(mfilename('fullpath')), name);
else
    netFile = name;
end
end
