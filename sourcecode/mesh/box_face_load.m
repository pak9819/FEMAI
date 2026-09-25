function fnode = box_face_load(coord, quads, dir, q)
%BOX_FACE_LOAD Consistent nodal loads of a constant surface traction.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Converts a constant surface load q (force per area, dead load in the
%   reference configuration) acting in global direction dir on the
%   bilinear face quads into equivalent nodal loads (2x2 Gauss on each
%   face quad).
%
% INPUT
%   coord  (nnode x 3) node coordinates
%   quads  (nf x 4) face quads (e.g. faceQuads.x1 of create_model_data_box)
%   dir    load direction 1/2/3
%   q      load intensity
%
% OUTPUT
%   fnode  [node, dir, value] rows (merged per node)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

a = 1/sqrt(3);
gp = [-a -a; a -a; a a; -a a];
nnode = size(coord, 1);
f = zeros(nnode, 1);
for e = 1:size(quads, 1)
    X = coord(quads(e,:), :);
    for g = 1:4
        r = gp(g,1);  s = gp(g,2);
        h   = 0.25 * [(1-r)*(1-s); (1+r)*(1-s); (1+r)*(1+s); (1-r)*(1+s)];
        h_r = 0.25 * [-(1-s); (1-s); (1+s); -(1+s)];
        h_s = 0.25 * [-(1-r); -(1+r); (1+r); (1-r)];
        dA  = norm(cross(X.' * h_r, X.' * h_s));
        f(quads(e,:)) = f(quads(e,:)) + q * h * dA;
    end
end
nodes = find(f ~= 0);
fnode = [nodes, dir * ones(numel(nodes), 1), f(nodes)];
end
