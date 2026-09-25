function [coord, elem, faces, faceQuads] = create_model_data_box(Lx, Ly, Lz, nx, ny, nz, distortion, seed)
%CREATE_MODEL_DATA_BOX Structured brick8 mesh of the box [0,Lx]x[0,Ly]x[0,Lz].
% ------------------------------------------------------------------------
% DESCRIPTION
%   Creates a structured hexahedral mesh with the brick8 node numbering of
%   shape_brick8.m (nodes 1-4 counter-clockwise at the lower z-level,
%   5-8 above). Optionally the INTERIOR nodes are shifted randomly by
%   +-distortion * element size per direction (boundary faces stay plane,
%   supports and loads are unaffected).
%
% INPUT
%   Lx, Ly, Lz   box dimensions
%   nx, ny, nz   number of elements per direction
%   distortion   (optional) interior node shift, fraction of element size
%   seed         (optional) random seed for the distortion (default 7)
%
% OUTPUT
%   coord      (nnode x 3) node coordinates
%   elem       (nel x 8)   connectivity
%   faces      struct with node lists x0, x1, y0, y1, z0, z1
%   faceQuads  struct with boundary face quads (nf x 4 node ids, outward
%              normal by right-hand rule) x0 ... z1 -- for surface loads,
%              see box_face_load.m
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
%
% COPYRIGHT AND LICENSE
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

if nargin < 7 || isempty(distortion), distortion = 0; end
if nargin < 8 || isempty(seed), seed = 7; end

x = linspace(0, Lx, nx+1);
y = linspace(0, Ly, ny+1);
z = linspace(0, Lz, nz+1);
[X, Y, Z] = ndgrid(x, y, z);               % x fastest
coord = [X(:), Y(:), Z(:)];

nid = @(i, j, k) i + (j-1)*(nx+1) + (k-1)*(nx+1)*(ny+1);

[I, J, K] = ndgrid(1:nx, 1:ny, 1:nz);
I = I(:);  J = J(:);  K = K(:);
elem = [nid(I,J,K),   nid(I+1,J,K),   nid(I+1,J+1,K),   nid(I,J+1,K), ...
        nid(I,J,K+1), nid(I+1,J,K+1), nid(I+1,J+1,K+1), nid(I,J+1,K+1)];

tol = 1e-9 * max([Lx, Ly, Lz]);
faces.x0 = find(abs(coord(:,1))      < tol);
faces.x1 = find(abs(coord(:,1) - Lx) < tol);
faces.y0 = find(abs(coord(:,2))      < tol);
faces.y1 = find(abs(coord(:,2) - Ly) < tol);
faces.z0 = find(abs(coord(:,3))      < tol);
faces.z1 = find(abs(coord(:,3) - Lz) < tol);

% Boundary face quads (outward normals)
[a, b] = ndgrid(1:ny, 1:nz);  a = a(:);  b = b(:);
faceQuads.x0 = [nid(1,a,b),    nid(1,a,b+1),    nid(1,a+1,b+1),    nid(1,a+1,b)];
faceQuads.x1 = [nid(nx+1,a,b), nid(nx+1,a+1,b), nid(nx+1,a+1,b+1), nid(nx+1,a,b+1)];
[a, b] = ndgrid(1:nx, 1:nz);  a = a(:);  b = b(:);
faceQuads.y0 = [nid(a,1,b),    nid(a+1,1,b),    nid(a+1,1,b+1),    nid(a,1,b+1)];
faceQuads.y1 = [nid(a,ny+1,b), nid(a,ny+1,b+1), nid(a+1,ny+1,b+1), nid(a+1,ny+1,b)];
[a, b] = ndgrid(1:nx, 1:ny);  a = a(:);  b = b(:);
faceQuads.z0 = [nid(a,b,1),    nid(a,b+1,1),    nid(a+1,b+1,1),    nid(a+1,b,1)];
faceQuads.z1 = [nid(a,b,nz+1), nid(a+1,b,nz+1), nid(a+1,b+1,nz+1), nid(a,b+1,nz+1)];

if distortion > 0
    inner = coord(:,1) > tol & coord(:,1) < Lx - tol & ...
            coord(:,2) > tol & coord(:,2) < Ly - tol & ...
            coord(:,3) > tol & coord(:,3) < Lz - tol;
    rs = RandStream('mt19937ar', 'Seed', seed);
    n  = nnz(inner);
    h  = [Lx/nx, Ly/ny, Lz/nz];
    coord(inner,:) = coord(inner,:) + distortion * h .* (2*rand(rs, n, 3) - 1);
end
end
