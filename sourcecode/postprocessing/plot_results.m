function plot_results(model, command, varargin)
%PLOT_RESULTS Plots FEM-Solid Edu model data and result fields.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Central plotting routine for meshes, boundary data, nodal forces,
%   deformed configurations and scalar result fields such as stresses,
%   strains and von-Mises stress. 3D meshes are shown as light gray
%   transparent surfaces with visible element edges by default. Use command
%   'wireframe' for a pure wireframe plot.
%
% INPUT
%   model    FE model structure
%   command  Plot command
%   varargin Optional results structure and/or scale factor
%
% OUTPUT
%   none
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

if isnumeric(command)
    [results, scale] = parse_results_scale(varargin, []);
    plot_result_field(model, results, command, scale);
    return
end

cmd = lower(char(command));

switch cmd
    case {'mesh','undeformed'}
        coord = model.coord;
        hold on;
        if model.info.DIM == 3
            plot_elements(model, coord, [0.2 0.2 0.2], [0.85 0.85 0.85], [], 0.35);
        else
            plot_elements(model, coord, 'k', 'none', []);
        end
        set_scale(model, coord);

    case 'wireframe'
        coord = model.coord;
        hold on;
        plot_elements(model, coord, [0.2 0.2 0.2], 'none', [], 1.0);
        set_scale(model, coord);

    case 'deformed'
        [results, scale] = parse_results_scale(varargin, 1.0);
        require_results(results, 'U', command);
        if model.info.DIM == 1
            plot_bar_deformation(model, results, scale);
            return
        end
        coordDef = deformed_coordinates(model, results.U, scale);
        hold on;
        if model.info.DIM == 3
            plot_elements(model, coordDef, [0.1 0.1 0.1], [0.35 0.55 0.9], [], 0.55);
        else
            plot_elements(model, coordDef, 'k', 'blue', []);
        end
        set_scale(model, [model.coord; coordDef]);

    case 'nodeforcesfext'
        [results, scale] = parse_results_scale(varargin, 1.0);
        require_results(results, 'Fext', command);
        hold on;
        plot_node_forces(model, model.coord, scale, results.Fext);
        set_scale(model, model.coord);

    case 'support'
        [~, scale] = parse_results_scale(varargin, 1.0);
        hold on;
        plot_supports(model, model.coord, scale);
        set_scale(model, model.coord);

    case 'nodenumber'
        hold on;
        plot_node_numbers(model);

    case 'elementnumber'
        hold on;
        plot_element_numbers(model);

    case 'setscale'
        set_scale(model, model.coord);

    otherwise
        [results, scale] = parse_results_scale(varargin, []);
        plot_result_field(model, results, command, scale);
end

end

% -------------------------------------------------------------------------
function [results, scale] = parse_results_scale(args, defaultScale)

results = [];
scale = defaultScale;

for i = 1:numel(args)
    arg = args{i};
    if isstruct(arg)
        results = arg;
    elseif isnumeric(arg) && isscalar(arg)
        scale = arg;
    elseif isempty(arg)
        continue
    else
        error('Unexpected plotting argument.');
    end
end

end

% -------------------------------------------------------------------------
function require_results(results, fieldName, command)

if isempty(results) || ~isfield(results, fieldName)
    error('plot_results needs results.%s for command "%s".', ...
          fieldName, command);
end

end

% -------------------------------------------------------------------------
function plot_bar_deformation(model, results, scale)

x = model.coord(:,1);
u = results.U(:);

hold on;
plot(x, zeros(model.info.NNODE,1), '-ko', ...
     'LineWidth', 2, ...
     'MarkerSize', 4);
plot(x, scale*u, '-bo', ...
     'LineWidth', 1, ...
     'MarkerSize', 3);

if ~isempty(model.bcond)
    nodesBC = model.bcond(:,1);
    plot(x(nodesBC), zeros(numel(nodesBC),1), '>r', ...
         'MarkerSize', 10);
end

title(sprintf('Displacement u_h(x) (scale x %.3g)', scale));
xlabel('x');
ylabel('u_h');
grid on;
set_scale(model, [x, zeros(size(x)); x, scale*u]);

end

% -------------------------------------------------------------------------
function plot_result_field(model, results, field, scale)

require_results(results, 'U', field);

if isempty(scale)
    scale = auto_deformation_scale(model, results.U);
end

[fieldValues, labelText] = resolve_field(results, field);
fieldValues = fieldValues(:);

if numel(fieldValues) ~= model.info.NNODE
    error('Result field must contain one value per node.');
end

if model.info.DIM == 1
    coordPlot = model.coord;
else
    coordPlot = deformed_coordinates(model, results.U, scale);
end

%figure('Color', 'w', 'Name', labelText);
hold on;
grid on;

plot_elements(model, coordPlot, [0.2 0.2 0.2], 'interp', fieldValues);

if model.info.DIM > 1
    colormap(jet(256));
    cb = colorbar;
    ylabel(cb, labelText);
end
if model.info.DIM == 1
    title(labelText);
else
    %title(sprintf('%s (deformation x %.3g)', labelText, scale));
    title(sprintf('%s ', labelText, scale));
end
xlabel('X');
if model.info.DIM >= 2
    ylabel('Y');
else
    ylabel('value');
end
if model.info.DIM == 3
    zlabel('Z');
end

end

% -------------------------------------------------------------------------
function [fieldValues, labelText] = resolve_field(results, field)

if isnumeric(field)
    fieldValues = field;
    labelText = 'FE result';
    return
end

name = lower(char(field));

switch name
    case {'vonmises','vm'}
        fieldValues = results.vonMises.node;
        labelText = 'Von-Mises stress';
    case {'stress_xx','sig_xx','sigma_xx','sxx'}
        fieldValues = get_node_component(results.stress.node, 1, field);
        labelText = 'Stress xx';
    case {'stress_yy','sig_yy','sigma_yy','syy'}
        fieldValues = get_node_component(results.stress.node, 2, field);
        labelText = 'Stress yy';
    case {'stress_zz','sig_zz','sigma_zz','szz'}
        fieldValues = get_node_component(results.stress.node, 3, field);
        labelText = 'Stress zz';
    case {'stress_xy','sig_xy','sigma_xy','sxy'}
        fieldValues = get_node_shear_component(results.stress.node, 'xy', field);
        labelText = 'Stress xy';
    case {'stress_yz','sig_yz','sigma_yz','syz'}
        fieldValues = get_node_component(results.stress.node, 5, field);
        labelText = 'Stress yz';
    case {'stress_zx','stress_xz','sig_zx','sigma_zx','szx','sxz'}
        fieldValues = get_node_component(results.stress.node, 6, field);
        labelText = 'Stress zx';
    case {'strain_xx','eps_xx','exx'}
        fieldValues = get_node_component(results.strain.node, 1, field);
        labelText = 'Strain xx';
    case {'strain_yy','eps_yy','eyy'}
        fieldValues = get_node_component(results.strain.node, 2, field);
        labelText = 'Strain yy';
    case {'strain_zz','eps_zz','ezz'}
        fieldValues = get_node_component(results.strain.node, 3, field);
        labelText = 'Strain zz';
    case {'strain_xy','eps_xy','exy'}
        fieldValues = get_node_shear_component(results.strain.node, 'xy', field);
        labelText = 'Strain xy';
    case {'strain_yz','eps_yz','eyz'}
        fieldValues = get_node_component(results.strain.node, 5, field);
        labelText = 'Strain yz';
    case {'strain_zx','strain_xz','eps_zx','ezx','exz'}
        fieldValues = get_node_component(results.strain.node, 6, field);
        labelText = 'Strain zx';
    otherwise
        error('Unknown result field "%s".', field);
end

end

% -------------------------------------------------------------------------
function values = get_node_component(nodeValues, component, field)

if isempty(nodeValues) || size(nodeValues,2) < component
    error('Result field "%s" is not available. Run compute_model_results first and check the model dimension.', field);
end

values = nodeValues(:,component);

end

% -------------------------------------------------------------------------
function values = get_node_shear_component(nodeValues, componentName, field)

if isempty(nodeValues)
    error('Result field "%s" is not available. Run compute_model_results first.', field);
end

switch componentName
    case 'xy'
        if size(nodeValues,2) == 3
            component = 3;
        else
            component = 4;
        end
    otherwise
        error('Unknown shear component "%s".', componentName);
end

values = get_node_component(nodeValues, component, field);

end

% -------------------------------------------------------------------------
function plot_elements(model, coord, edgeColor, faceColor, fieldValues, faceAlpha)

if nargin < 6 || isempty(faceAlpha)
    faceAlpha = 1.0;
end

switch model.info.DIM
    case 1
        c = coord_to_3d(coord);
        if isempty(fieldValues)
            for e = 1:model.info.NEL
                nodes = model.elem(e,:);
                plot3(c(nodes,1), c(nodes,2), c(nodes,3), '-', ...
                      'Color', edgeColor, 'LineWidth', 1.0);
            end
            plot3(c(:,1), c(:,2), c(:,3), 'ko', ...
                  'MarkerSize', 4, ...
                  'MarkerFaceColor', 'w');
        else
            plot_bar_field(model, coord, fieldValues);
        end
        view(2);

    case 2
        if isempty(fieldValues)
            patch('Faces', model.elem, ...
                  'Vertices', coord, ...
                  'EdgeColor', edgeColor, ...
                  'FaceColor', faceColor);
        else
            patch('Faces', model.elem, ...
                  'Vertices', coord, ...
                  'FaceVertexCData', fieldValues, ...
                  'FaceColor', faceColor, ...
                  'EdgeColor', edgeColor);
        end
        view(2);

    case 3
        if isempty(fieldValues)
            patch('Faces', element_faces(model), ...
                  'Vertices', coord, ...
                  'EdgeColor', edgeColor, ...
                  'FaceColor', faceColor, ...
                  'FaceAlpha', faceAlpha);
        else
            patch('Faces', element_faces(model), ...
                  'Vertices', coord, ...
                  'FaceVertexCData', fieldValues, ...
                  'FaceColor', faceColor, ...
                  'EdgeColor', 'none');
            camlight;
            lighting gouraud;
        end
        view(3);

    otherwise
        error('plot_results supports only DIM = 1, 2, 3.');
end

if ~(model.info.DIM == 1 && ~isempty(fieldValues))
    axis equal;
end

end

% -------------------------------------------------------------------------
function faces = element_faces(model)

switch model.info.NNEL
    case 2
        faces = model.elem;
    case 3
        faces = model.elem;
    case 4
        if model.info.DIM == 2
            faces = model.elem;
        else
            localFaces = [1 2 3
                          1 2 4
                          2 3 4
                          3 1 4];
            faces = expand_faces(model.elem, localFaces);
        end
    case 8
        localFaces = [1 2 3 4
                      5 6 7 8
                      1 2 6 5
                      2 3 7 6
                      3 4 8 7
                      4 1 5 8];
        faces = expand_faces(model.elem, localFaces);
    otherwise
        error('No face definition for elements with %i nodes.', model.info.NNEL);
end

end

% -------------------------------------------------------------------------
function faces = expand_faces(elem, localFaces)

ne = size(elem,1);
nf = size(localFaces,1);
faces = zeros(ne*nf, size(localFaces,2));

for e = 1:ne
    rows = (e-1)*nf + (1:nf);
    faces(rows,:) = reshape(elem(e,localFaces(:)), size(localFaces));
end

end

% -------------------------------------------------------------------------
function plot_node_forces(model, coord, scale, Fext)

Fmat = vec2mat(Fext, model.info.DOF);
c = coord_to_3d(coord);

for i = 1:model.info.NNODE
    f = zeros(1,3);
    f(1:min(3,model.info.DOF)) = Fmat(i,1:min(3,model.info.DOF));
    f = scale*f;

    if norm(f) == 0
        continue
    end

    if model.info.DIM <= 2
        quiver(c(i,1), c(i,2), f(1), f(2), 0, 'r-');
    else
        quiver3(c(i,1), c(i,2), c(i,3), f(1), f(2), f(3), 0, 'r-');
    end
end

end

% -------------------------------------------------------------------------
function plot_bar_field(model, coord, fieldValues)

x = coord(:,1);

plot(x, zeros(model.info.NNODE,1), '-ko', ...
     'LineWidth', 2, ...
     'MarkerSize', 4);

for e = 1:model.info.NEL
    nodes = model.elem(e,:);
    x1 = x(nodes(1));
    x2 = x(nodes(2));
    value = mean(fieldValues(nodes));

    fill([x1 x2 x2 x1], [0 0 value value], [0.65 0.65 0.65], ...
         'FaceAlpha', 0.35, ...
         'EdgeColor', 'k');
end

plot(x, zeros(model.info.NNODE,1), 'ko', ...
     'MarkerSize', 4, ...
     'MarkerFaceColor', 'w');

set_bar_field_limits(x, fieldValues);

end

% -------------------------------------------------------------------------
function set_bar_field_limits(x, fieldValues)

xSpan = max(x) - min(x);
if xSpan == 0
    xSpan = 1.0;
end

yMin = min([0; fieldValues(:)]);
yMax = max([0; fieldValues(:)]);
ySpan = yMax - yMin;
if ySpan == 0
    ySpan = max(abs(yMax), 1.0);
end

xlim([min(x)-0.05*xSpan, max(x)+0.05*xSpan]);
ylim([yMin-0.15*ySpan, yMax+0.15*ySpan]);

end

% -------------------------------------------------------------------------
function plot_supports(model, coord, scale)

if isempty(model.bcond)
    return
end

c = coord_to_3d(coord);
markerSize = max(4, 6*scale);
color = [1 0 0];

for i = 1:size(model.bcond,1)
    node = model.bcond(i,1);
    dir = model.bcond(i,2);

    switch dir
        case 1
            marker = '<';
        case 2
            marker = 'v';
        case 3
            marker = 's';
        otherwise
            marker = 'o';
    end

    if model.info.DIM <= 2
        plot(c(node,1), c(node,2), marker, ...
             'Color', color, ...
             'MarkerFaceColor', color, ...
             'MarkerSize', markerSize);
    else
        plot3(c(node,1), c(node,2), c(node,3), marker, ...
              'Color', color, ...
              'MarkerFaceColor', color, ...
              'MarkerSize', markerSize);
    end
end

end

% -------------------------------------------------------------------------
function plot_node_numbers(model)

if model.info.DIM == 1
    plot_bar_node_numbers(model);
    return
end

offset = label_offset(model.coord, model.info.DIM);
labelPoints = zeros(model.info.NNODE, 3);

for i = 1:model.info.NNODE
    labelPoints(i,:) = plot_text(model.coord(i,:), offset, num2str(i), 'k', model.info.DIM);
end

expand_limits(labelPoints, model.info.DIM);

end

% -------------------------------------------------------------------------
function plot_element_numbers(model)

if model.info.DIM == 1
    plot_bar_element_numbers(model);
    return
end

labelPoints = zeros(model.info.NEL, 3);

for e = 1:model.info.NEL
    center = mean(model.coord(model.elem(e,:),:),1);
    labelPoints(e,:) = plot_text(center, zeros(1,3), num2str(e), 'b', model.info.DIM, false);
end

expand_limits(labelPoints, model.info.DIM);

end

% -------------------------------------------------------------------------
function plot_bar_node_numbers(model)

x = model.coord(:,1);
y = bar_label_height();
labelPoints = zeros(model.info.NNODE, 3);

for i = 1:model.info.NNODE
    text(x(i), y, num2str(i), ...
         'FontSize', 8, ...
         'Color', 'k', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', ...
         'Clipping', 'off');
    labelPoints(i,:) = [x(i), y, 0];
end

expand_limits(labelPoints, 1);

end

% -------------------------------------------------------------------------
function plot_bar_element_numbers(model)

y = bar_label_height();
labelPoints = zeros(model.info.NEL, 3);

for e = 1:model.info.NEL
    xCenter = mean(model.coord(model.elem(e,:),1));
    text(xCenter, y, num2str(e), ...
         'FontSize', 8, ...
         'Color', 'b', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom', ...
         'Clipping', 'off');
    labelPoints(e,:) = [xCenter, y, 0];
end

expand_limits(labelPoints, 1);

end

% -------------------------------------------------------------------------
function y = bar_label_height()

yl = ylim;
span = yl(2) - yl(1);
if span == 0
    span = 1.0;
end

y = 0 + 0.015*span;

end

% -------------------------------------------------------------------------
function scale = auto_deformation_scale(model, U)

Umat = vec2mat(U, model.info.DOF);
maxU = max(abs(Umat(:)));

if maxU == 0
    scale = 1.0;
    return
end

span = max(model.coord, [], 1) - min(model.coord, [], 1);
extent = max(span);
if extent == 0
    extent = 1.0;
end

scale = 0.1*extent/maxU;

end

% -------------------------------------------------------------------------
function coordDef = deformed_coordinates(model, U, scale)

Umat = vec2mat(U, model.info.DOF);
coordDef = model.coord;
coordDef(:,1:model.info.DIM) = coordDef(:,1:model.info.DIM) + ...
                               scale*Umat(:,1:model.info.DIM);

end

% -------------------------------------------------------------------------
function set_scale(model, coord)

c = coord_to_3d(coord);
span = max(c,[],1) - min(c,[],1);
tol = 0.05*max(span);
if tol == 0
    tol = 1;
end

xlim([min(c(:,1))-tol, max(c(:,1))+tol]);

if model.info.DIM >= 2
    ylim([min(c(:,2))-tol, max(c(:,2))+tol]);
else
    ylim([-tol, tol]);
end

if model.info.DIM == 3
    zlim([min(c(:,3))-tol, max(c(:,3))+tol]);
    view(3);
else
    view(2);
end

axis equal;

end

% -------------------------------------------------------------------------
function c = coord_to_3d(coord)

c = zeros(size(coord,1),3);
c(:,1:size(coord,2)) = coord;

end

% -------------------------------------------------------------------------
function offset = label_offset(coord, DIM)

span = max(coord,[],1) - min(coord,[],1);
delta = 0.004*max(span);
if delta == 0
    delta = 0.004;
end

offset = zeros(1,3);
offset(1:min(DIM,3)) = delta;

end

% -------------------------------------------------------------------------
function p = plot_text(point, offset, label, color, DIM, addBlank)

if nargin < 6
    addBlank = true;
end

p = zeros(1,3);
p(1:numel(point)) = point;
p(1:numel(offset)) = p(1:numel(offset)) + offset;

if addBlank
    labelText = [' ' label];
else
    labelText = label;
end

if DIM <= 2
    text(p(1), p(2), labelText, ...
         'FontSize', 5, ...
         'Color', color, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'Clipping', 'off');
else
    text(p(1), p(2), p(3), labelText, ...
         'FontSize', 5, ...
         'Color', color, ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'Clipping', 'off');
end

end

% -------------------------------------------------------------------------
function expand_limits(points, DIM)

if isempty(points)
    return
end

xl = xlim;
yl = ylim;
xlim([min([xl(1); points(:,1)]), max([xl(2); points(:,1)])]);

if DIM >= 2
    ylim([min([yl(1); points(:,2)]), max([yl(2); points(:,2)])]);
end

if DIM == 3
    zl = zlim;
    zlim([min([zl(1); points(:,3)]), max([zl(2); points(:,3)])]);
end

end
