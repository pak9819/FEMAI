%FEMSOLID_EX_QUAD4_03_AI Demonstrates the Deep Learned FEM approach on quad4.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Loest dasselbe 2D-Problem mit zwei Backends und vergleicht:
%
%     (1) Klassisch:  Ke aus analytischer Gauss-Integration (int B'CB dV)
%                     (backend = 'matlab')
%     (2) KI-Element: Ke als Ganzes aus einem neuronalen Netz
%                     (backend = 'ai')
%
%   Das KI-Element lernt nicht mehr die B-Matrix, sondern direkt die
%   gesamte Element-Steifigkeitsmatrix Ke (8x8). Ein einziger Forward-Pass
%   liefert die komplette Matrix -- ohne Gauss-Schleife und ohne B-Matrix.
%
%   Genutzte Physik: Ke ist in 2D groesseninvariant und skaliert linear mit
%   E*d. Das Netz lernt die dimensionslose Form-Steifigkeit; nu und der
%   ebene Zustand (planeStrain) sind fest eintrainiert.
%
%   Geometrie: Rechteck 2x1, zwei quad4-Elemente, linke Seite fest,
%              Knotenlast am rechten unteren Knoten, Volumenlast.
%
% PREREQUISITE
%   Run train_quad4_K_network.py first.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-06-10
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

close all; clear; clc;

fprintf('=== FEM-Solid: quad4 -- Klassisch vs. Deep Learned FEM (ganze Ke) ===\n\n');

% ------------------------------------------------------------------------
%% (1) Referenzloesung: klassisches MATLAB-Backend
% ------------------------------------------------------------------------

setup_ref = init_setup;
setup_ref.analysis.nl     = false;
setup_ref.element.type    = 'quad4';
setup_ref.element.backend = 'matlab';
setup_ref.material.name   = 'Hooke';
setup_ref.material.condition = 'planeStrain';

model_ref = data_ex_quad4_01_two_elements(setup_ref);
[U_ref, stepResults_ref] = solve_FE(model_ref);
results_ref = compute_model_results(model_ref, stepResults_ref(end));

% ------------------------------------------------------------------------
%% (2) KI-Element: backend = 'ai'
% ------------------------------------------------------------------------

setup_ai = init_setup;
setup_ai.analysis.nl     = false;
setup_ai.element.type    = 'quad4';
setup_ai.element.backend = 'ai';
setup_ai.material.name   = 'Hooke';
setup_ai.material.condition = 'planeStrain';

model_ai = data_ex_quad4_01_two_elements(setup_ai);
[U_ai, stepResults_ai] = solve_FE(model_ai);
results_ai = compute_model_results(model_ai, stepResults_ai(end));

% ------------------------------------------------------------------------
%% Direkter Vergleich der Element-Steifigkeitsmatrix (Element 1)
%
%   Zeigt unmittelbar, was das Netz gelernt hat: die gesamte 8x8-Matrix.
% ------------------------------------------------------------------------

e        = 1;
nodes    = model_ref.elem(e, :);
coord_e  = model_ref.coord(nodes, :);
mat_e    = model_ref.mat(e, :);
gpts     = model_ref.element.gp;
wts      = model_ref.element.w;
ndofEl   = model_ref.element.ndofElement;
Ue0      = zeros(ndofEl, 1);
b_e0     = zeros(model_ref.info.DOF, 1);

Ke_classic = element_quad4_lin(coord_e, mat_e, b_e0, 0, Ue0, [], gpts, wts, ...
                               model_ref.material.name, model_ref.material.condition, model_ref.element.opts);
Ke_ai      = element_quad4_lin_ai(coord_e, mat_e, b_e0, 0, Ue0, [], gpts, wts, ...
                               model_ai.material.name, model_ai.material.condition, model_ai.element.opts);

relKe = norm(Ke_ai - Ke_classic, 'fro') / norm(Ke_classic, 'fro');

fprintf('--- Element-Steifigkeitsmatrix Ke (Element %d) ---\n', e);
fprintf('  Klassisch (analytisch):\n');
disp(Ke_classic);
fprintf('  KI-Element (gelernt):\n');
disp(Ke_ai);
fprintf('  Relativer Fehler (Frobenius-Norm): %.4e\n', relKe);
fprintf('  Symmetriefehler KI-Ke:             %.2e\n', norm(Ke_ai - Ke_ai', 'fro'));

% ------------------------------------------------------------------------
%% Vergleich: Verschiebungen
% ------------------------------------------------------------------------

fprintf('\n--- Verschiebungsvergleich ---\n');
fprintf('  DOF | u_klassisch | u_KI       | rel. Fehler\n');
fprintf('  ----|-------------|------------|------------\n');

NDOF = length(U_ref);
for i = 1:NDOF
    if abs(U_ref(i)) > 1e-12
        rel_err = abs(U_ai(i) - U_ref(i)) / abs(U_ref(i));
    else
        rel_err = abs(U_ai(i) - U_ref(i));
    end
    fprintf('  %3d | %11.6f | %10.6f | %.2e\n', i, U_ref(i), U_ai(i), rel_err);
end

max_rel_err = max(abs(U_ai - U_ref) ./ max(abs(U_ref), 1e-12));
fprintf('\n  Max. relativer Fehler (Verschiebungen): %.2e\n', max_rel_err);

% ------------------------------------------------------------------------
%% Vergleich: von-Mises-Spannung
% ------------------------------------------------------------------------

vm_ref = results_ref.vonMises.node;
vm_ai  = results_ai.vonMises.node;

fprintf('\n--- von-Mises-Spannung (knotenweise) ---\n');
fprintf('  Knoten | vonMises_klass | vonMises_KI  | rel. Fehler\n');
fprintf('  -------|----------------|--------------|------------\n');
for i = 1:length(vm_ref)
    if abs(vm_ref(i)) > 1e-12
        rel_err = abs(vm_ai(i) - vm_ref(i)) / abs(vm_ref(i));
    else
        rel_err = abs(vm_ai(i) - vm_ref(i));
    end
    fprintf('  %6d | %14.4f | %12.4f | %.2e\n', i, vm_ref(i), vm_ai(i), rel_err);
end

% ------------------------------------------------------------------------
%% Plots
% ------------------------------------------------------------------------

figure('Name', 'Klassisch: Vernetzung', 'NumberTitle', 'off');
plot_results(model_ref, 'undeformed');
plot_results(model_ref, 'support', 1.0);
plot_results(model_ref, 'nodenumber');
title('Klassisch: Vernetzung');

figure('Name', 'Klassisch: Verformung + von-Mises', 'NumberTitle', 'off');
scale = 1.0;
plot_results(model_ref, 'undeformed');
plot_results(model_ref, 'deformed', results_ref, scale);
title('Klassisch: Verformung');

figure('Name', 'Klassisch: von-Mises-Spannung', 'NumberTitle', 'off');
plot_results(model_ref, 'vonMises', results_ref, 1.0);
title('Klassisch (matlab)');

figure('Name', 'KI-Element: von-Mises-Spannung', 'NumberTitle', 'off');
plot_results(model_ai, 'vonMises', results_ai, 1.0);
title('KI-Element (ai)');
