function routine = element_routine(element)
%ELEMENT_ROUTINE Selects the element routine function handle.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Selects the element routine function handle from element type,
%   analysis type and backend name.
%
% INPUT
%   element  Element data structure
%
% OUTPUT
%   routine  Function handle of the selected element routine
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-05-08
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

matlabRoutineName = get_matlab_routine_name(element);
backend = lower(element.backend);

switch backend

    case 'matlab'
        routineName = matlabRoutineName;

    case 'vectorized'
        routineName = [matlabRoutineName '_vectorized'];

    case 'mex'
        routineName = [matlabRoutineName '_mex'];

    case 'ai'
        routineName = [matlabRoutineName '_ai'];

    otherwise
        error('Unknown element backend: %s', element.backend);

end

if exist(routineName, 'file') ~= 2 && exist(routineName, 'file') ~= 3
    error(['Element routine "%s" for element type "%s" and backend "%s" ', ...
           'was not found.'], ...
          routineName, element.type, element.backend);
end

routine = str2func(routineName);

end

% ------------------------------------------------------------------------
function routineName = get_matlab_routine_name(element)
%GET_MATLAB_ROUTINE_NAME Name der Referenz-MATLAB-Routine.

if element.nl
    suffix = 'nl';
else
    suffix = 'lin';
end

routineName = sprintf('element_%s_%s', element.type, suffix);

end
