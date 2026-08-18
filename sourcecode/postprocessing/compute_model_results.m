function results = compute_model_results(model, results)
%COMPUTE_MODEL_RESULTS Computes stresses and strains for FE results.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Computes stress, strain and von-Mises values at Gauss points and
%   averaged nodal values for one or more load-step result structures.
%   Thermal strains are subtracted before stress evaluation, i.e. stresses
%   are calculated from the elastic strain.
%
% INPUT
%   model    FE model structure
%   results  Result structure or result structure array with field U
%
% OUTPUT
%   results  Result structure extended by stress, strain and vonMises:
%
%            results(k).stress.gp{e}
%                     Stress values at the Gauss points of element e in
%                     Voigt notation.
%
%            results(k).stress.node
%                     Nodal stress values averaged from the element
%                     Gauss-point values.
%
%            results(k).strain.gp{e}
%                     Total strain values at the Gauss points of element e
%                     in Voigt notation.
%
%            results(k).strain.node
%                     Nodal total strain values averaged from the element
%                     Gauss-point values.
%
%            results(k).strain.total
%                     Same total strain values as results(k).strain.
%
%            results(k).strain.thermal
%                     Thermal strain values alphaT*DeltaT.
%
%            results(k).strain.elastic
%                     Elastic strain values used for the stress
%                     calculation.
%
%            results(k).vonMises.gp{e}
%                     von-Mises stress values at the Gauss points of
%                     element e.
%
%            results(k).vonMises.node
%                     Nodal von-Mises stress values averaged from the
%                     element Gauss-point values.
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

timer = tic;
fprintf('\nStart postprocessing ...\n');

for k = 1:numel(results)
    if ~isfield(results(k), 'U') || isempty(results(k).U)
        error('compute_model_results needs results(%i).U.', k);
    end

    post = compute_post_results(model, results(k).U);
    results(k).stress = post.stress;                 
    results(k).strain = post.strain;
    results(k).vonMises = post.vonMises;
end

elapsedTime = toc(timer);
fprintf('End postprocessing. Calculation time: %.3f s.\n', elapsedTime);

end

% -------------------------------------------------------------------------
function post = compute_post_results(model, U)

NEL = model.info.NEL;
nVoigt = compute_num_voigt(model.info.DIM);

if model.analysis.nl && ~any(strcmpi(model.material.name, {'Hooke','StVenant'})) && ...
        any(abs(model.DeltaT(:)) > 0)
    warning('compute_model_results:ThermalHyperelasticNotImplemented', ...
        ['Thermal stress correction is implemented for Hooke and StVenant. ', ...
         'For %s, stress is computed without thermal strain correction.'], ...
        model.material.name);
end

post.stress.gp = cell(NEL, 1);
post.strain.gp = cell(NEL, 1);
post.strain.total.gp = cell(NEL, 1);
post.strain.thermal.gp = cell(NEL, 1);
post.strain.elastic.gp = cell(NEL, 1);
post.vonMises.gp = cell(NEL, 1);

for e = 1:NEL
    dofs_e  = get_element_dofs(e, model.elem, model.info.DOF);
    coord_e = model.coord(model.elem(e,:), :);
    mat_e   = model.mat(e,:);
    DeltaT_e = model.DeltaT(e,:);
    Ue      = U(dofs_e);

    [sig_e, eps_total_e, eps_thermal_e, eps_elastic_e, vm_e] = ...
        compute_element_stress_strain(coord_e, mat_e, DeltaT_e, Ue, model);

    post.stress.gp{e} = sig_e;
    post.strain.gp{e} = eps_total_e;
    post.strain.total.gp{e} = eps_total_e;
    post.strain.thermal.gp{e} = eps_thermal_e;
    post.strain.elastic.gp{e} = eps_elastic_e;
    post.vonMises.gp{e} = vm_e;
end

[post.stress.node, post.strain.node, post.vonMises.node] = ...
    compute_nodal_average(model, post, nVoigt);
post.strain.total.node = post.strain.node;
post.strain.thermal.node = compute_nodal_average_field(model, post.strain.thermal.gp, nVoigt);
post.strain.elastic.node = compute_nodal_average_field(model, post.strain.elastic.gp, nVoigt);

end

% -------------------------------------------------------------------------
function [sig_voigt, eps_total_voigt, eps_thermal_voigt, eps_elastic_voigt, vm_gp] = ...
    compute_element_stress_strain(coord_e, mat_e, DeltaT_e, Ue, model)
% Verzerrungen und Spannungen an den Gauß-Punkten

DIM = size(coord_e, 2);
numgp = size(model.element.gp, 1);
nVoigt = compute_num_voigt(DIM);

sig_voigt = zeros(numgp, nVoigt);          % sigma in Voigt-Notation
eps_total_voigt = zeros(numgp, nVoigt);    % Gesamtverzerrung
eps_thermal_voigt = zeros(numgp, nVoigt);  % thermische Verzerrung
eps_elastic_voigt = zeros(numgp, nVoigt);  % elastische Verzerrung
vm_gp = zeros(numgp, 1);                   % von Mises-Vergleichsspannung

eps_thermal = compute_thermal_strain_vector(DIM, mat_e, DeltaT_e);

for i = 1:numgp
    rst_loc = model.element.gp(i,:);
    [~, dh_dx, ~] = element_shape_data(model.element, coord_e, rst_loc);

    % Verschiebungsgradient
    gradU = compute_gradU(dh_dx, Ue, DIM, model.info.DOF);

    if model.analysis.nl
        % Deformationsgradient
        F = eye(DIM) + gradU;
        % Green-Lagrangesche Verzerrung
        E = 0.5*(F'*F - eye(DIM));
        Evec = Emat2Evec(E);
        Eel_vec = Evec - eps_thermal;

        if any(strcmpi(model.material.name, {'Hooke','StVenant'}))
            [~, C] = material_elasticity(mat_e, gradU, ...
                model.material.name, model.material.condition);
            S = Svec2Smat(C*Eel_vec);
        else
            % Thermal strains for hyperelastic material models require a
            % multiplicative thermal split and are not part of this Edu
            % implementation yet.
            S = material_elasticity(mat_e, gradU, ...
                model.material.name, model.material.condition);
        end

        % Cauchy-Spannung
        if DIM == 1
            sigma = F*S;  % beim 1D-Dehnstab ist sigma = P = F*S
        else
            sigma = (1/det(F))*F*S*F';
        end
    else
        % Lineareer Verzerrungstensor
        E = 0.5*(gradU + gradU');
        Evec = Emat2Evec(E);
        Eel_vec = Evec - eps_thermal;
        [~, C] = material_elasticity(mat_e, zeros(DIM), ...
            model.material.name, model.material.condition);
        sigma = Svec2Smat(C*Eel_vec);
    end

    sig_voigt(i,:) = Smat2Svec(sigma)';       
    eps_total_voigt(i,:) = Evec';
    eps_thermal_voigt(i,:) = eps_thermal';
    eps_elastic_voigt(i,:) = Eel_vec';
    vm_gp(i) = compute_von_mises(sigma, DIM); % von Mises-Vergleichsspannung
end

end


% -------------------------------------------------------------------------
function [node_sig, node_eps, node_vm] = compute_nodal_average(model, post, nVoigt)
% Mittelwerte an den Knoten berechnen

NNODE = model.info.NNODE;
node_sig = zeros(NNODE, nVoigt);
node_eps = zeros(NNODE, nVoigt);
node_vm = zeros(NNODE, 1);
node_count = zeros(NNODE, 1);

for e = 1:model.info.NEL
    nodes_e = model.elem(e, :);

    sig_mean = mean(post.stress.gp{e}, 1);
    eps_mean = mean(post.strain.gp{e}, 1);
    vm_mean = mean(post.vonMises.gp{e}, 1);

    node_sig(nodes_e,:) = node_sig(nodes_e,:) + sig_mean;
    node_eps(nodes_e,:) = node_eps(nodes_e,:) + eps_mean;
    node_vm(nodes_e) = node_vm(nodes_e) + vm_mean;
    node_count(nodes_e) = node_count(nodes_e) + 1;
end

node_count(node_count == 0) = 1;
node_sig = node_sig ./ node_count;
node_eps = node_eps ./ node_count;
node_vm = node_vm ./ node_count;

end


% -------------------------------------------------------------------------
function node_values = compute_nodal_average_field(model, element_values, nVoigt)
% Mittelwerte eines Elementfelds an den Knoten berechnen

NNODE = model.info.NNODE;
node_values = zeros(NNODE, nVoigt);
node_count = zeros(NNODE, 1);

for e = 1:model.info.NEL
    nodes_e = model.elem(e, :);
    elem_mean = mean(element_values{e}, 1);

    node_values(nodes_e,:) = node_values(nodes_e,:) + elem_mean;
    node_count(nodes_e) = node_count(nodes_e) + 1;
end

node_count(node_count == 0) = 1;
node_values = node_values ./ node_count;

end


% -------------------------------------------------------------------------
function eps_thermal = compute_thermal_strain_vector(DIM, mat_e, DeltaT_e)
% Thermische Dehnung in Voigt-Notation

alphaT = mat_e(4);
if isempty(DeltaT_e)
    DeltaT = 0;
else
    DeltaT = DeltaT_e(1);
end

switch DIM
    case 1
        eps_thermal = alphaT*DeltaT;
    case 2
        eps_thermal = alphaT*DeltaT*[1; 1; 0];
    case 3
        eps_thermal = alphaT*DeltaT*[1; 1; 1; 0; 0; 0];
    otherwise
        error('Unsupported dimension: %i.', DIM);
end

end


% -------------------------------------------------------------------------
function vm = compute_von_mises(S, DIM)
% von Mises-Vergleichsspannung

if DIM == 1
    vm = abs(S);
elseif DIM == 2
    s11 = S(1,1);
    s22 = S(2,2);
    s12 = S(1,2);
    vm = sqrt(s11^2 - s11*s22 + s22^2 + 3*s12^2);
elseif DIM == 3
    s11 = S(1,1);
    s22 = S(2,2);
    s33 = S(3,3);
    s12 = S(1,2);
    s23 = S(2,3);
    s31 = S(3,1);
    vm = sqrt(0.5*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2) + ...
              3*(s12^2 + s23^2 + s31^2));
else
    error('Von-Mises stress is not defined for DIM = %i.', DIM);
end

end


% -------------------------------------------------------------------------
function n = compute_num_voigt(DIM)
% Hilfsfunktion für Voigt-Notation

switch DIM
    case 1
        n = 1;
    case 2
        n = 3;
    case 3
        n = 6;
    otherwise
        error('Unsupported dimension: %i.', DIM);
end

end
