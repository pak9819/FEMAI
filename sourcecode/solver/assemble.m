function [K,Fext,Fint] = assemble(model,U)
%ASSEMBLE Assembles global stiffness matrix and force vectors.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Assembles global stiffness matrix and force vectors.
%
% INPUT
%   model    FE model structure
%   U        Global displacement vector
%
% OUTPUT
%   K        Global stiffness matrix
%   Fext     Global external force vector
%   Fint     Global internal force vector
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

DOF  = model.info.DOF;
NDOF = model.info.NDOF;
NEL  = model.info.NEL;

if nargin == 1
    U = zeros(NDOF,1);
end

Fb   = zeros(NDOF,1);  % Globaler Lastvektor infolge Volumenlast im Gebiet
Fk   = zeros(NDOF,1);  % Globaler Lastvektor infolge zusätzlicher Knotenlasten
Ft   = zeros(NDOF,1);  % Globaler Lastvektor infolge Temperaturänderung im Gebiet

Fint = zeros(NDOF,1);  % Globaler Vektor der inneren Kraefte

MATNAME = model.material.name;
MATCOND = model.material.condition;

% Anzahl lokaler Freiheitsgrade
NNEL   = size(model.elem,2);
NDOFEL = DOF * NNEL;

% Speicher fuer sparse Triplets
nEntries = NEL * NDOFEL * NDOFEL;
I = zeros(nEntries,1);
J = zeros(nEntries,1);
Kvec = zeros(nEntries,1);

idx = 0;
% ------------------------------------------------------------------------
for e = 1:NEL

    dofs_e  = get_element_dofs(e, model.elem, DOF);
    coord_e = model.coord(model.elem(e,:),:);
    mat_e   = model.mat(e,:);
    Ue      = U(dofs_e);
    b_e     = model.fvol(e,:)';
    DeltaT_e = model.DeltaT(e,:);

    history_e = [];  % wird bei Elastizitaet nicht benoetigt

    [Ke,Fpe,Fte,Finte,history_e] = model.element.routine( ...
        coord_e, mat_e, b_e, DeltaT_e, Ue, history_e, ...
        model.element.gp, model.element.w, ...
        MATNAME, MATCOND, model.element.opts);

    % Sparse-Koordinaten fuer dieses Element
    [II,JJ] = ndgrid(dofs_e,dofs_e);
    n = numel(Ke);

    I(idx+1:idx+n) = II(:);
    J(idx+1:idx+n) = JJ(:);
    Kvec(idx+1:idx+n) = Ke(:);

    idx = idx + n;

    Fb(dofs_e)   = Fb(dofs_e) + Fpe;
    Ft(dofs_e)   = Ft(dofs_e) + Fte;
    Fint(dofs_e) = Fint(dofs_e) + Finte;
end
% ------------------------------------------------------------------------


% Nicht benutzte Eintraege abschneiden
I = I(1:idx);
J = J(1:idx);
Kvec = Kvec(1:idx);

% ------------------------------------------------------------------------
% Globale sparse Steifigkeitsmatrix
K = sparse(I,J,Kvec,NDOF,NDOF);

% ------------------------------------------------------------------------
% Knotenlasten
if ~isempty(model.fnode)
    for i = 1:size(model.fnode,1)
        node  = model.fnode(i,1);
        dir   = model.fnode(i,2);
        value = model.fnode(i,3);

        gdof = model.info.gdof(node,dir);
        Fk(gdof) = Fk(gdof) + value;
    end
end

% ------------------------------------------------------------------------
Fext = Fb + Fk + Ft;

end
