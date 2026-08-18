function edofs = get_element_dofs(e,elem,DOF)
%GET_ELEMENT_DOFS Returns global degree-of-freedom indices of one element.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Returns global degree-of-freedom indices of one element.
%
% INPUT
%   e        Element number
%   elem     Element connectivity
%   DOF      Degrees of freedom per node
%
% OUTPUT
%   edofs    Element degree-of-freedom indices
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

dofnode = DOF*elem(e,:);

% --- 1 DOF per node ----------------------------------------------------- 
if DOF == 1 
   edofs = dofnode;
% --- 2 DOF per node -----------------------------------------------------      
elseif DOF == 2 
   edofs = [dofnode - 1; dofnode];
% --- 3 DOF per node -----------------------------------------------------      
elseif DOF == 3 
   edofs = [dofnode - 2; dofnode - 1; dofnode]; 
else
    error('error DOF');
end   
      
% ------------------------------------------------------------------------
% sortieren
edofs =  edofs(:)';

