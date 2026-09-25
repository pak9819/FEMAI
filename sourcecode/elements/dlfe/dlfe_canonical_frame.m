function [chat, Lc, Rc, xbar] = dlfe_canonical_frame(coord_e)
%DLFE_CANONICAL_FRAME Geometrie-Kanonisierung fuer DLFE-Energienetze (2D/3D).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Bringt die Elementgeometrie in den kanonischen Rahmen des Netzes:
%
%     xbar = (coord - centroid) / Lc          Lc = mittlerer Knotenabstand
%     chat = xbar * Rc'                       Orientierung normiert
%
%   2D: Rc dreht die Kante 1->2 auf +x (identisch quad4_nl_ai_energy.m).
%   3D: e1 = Kante 1->2, e2 = Kante 1->4 orthogonalisiert, e3 = e1 x e2;
%       Rc = [e1; e2; e3] (echte Drehung, det = +1).
%   Python-Gegenstueck: training/brick8/brick8_nl_ref.py (canonical_frame).
%
%   Die Zustandsgroesse der Gram-Kette ist rotationsinvariant und braucht
%   Rc NICHT -- sie arbeitet mit xbar in physikalischer Orientierung.
%
% INPUT
%   coord_e  (n x dim) Knotenkoordinaten
%
% OUTPUT
%   chat     (n*dim x 1) kanonische Koordinaten [x1;y1;(z1);x2;...]
%   Lc       charakteristische Laenge
%   Rc       (dim x dim) Drehung physikalisch -> kanonisch
%   xbar     (n x dim) zentrierte, skalierte Koordinaten (physikalisch)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

n  = size(coord_e, 1);
xc = coord_e - sum(coord_e, 1) / n;
Lc = sum(sqrt(sum(xc.^2, 2))) / n;
if Lc <= eps
    error('dlfe_canonical_frame:DegenerateElement', ...
          'Degeneriertes Element: Lc ist numerisch null.');
end
xbar = xc / Lc;

dim = size(coord_e, 2);
if dim == 2
    e12 = xbar(2,:) - xbar(1,:);
    phi = atan2(e12(2), e12(1));
    c = cos(phi);  s = sin(phi);
    Rc = [c, s; -s, c];
elseif dim == 3
    e1 = xbar(2,:) - xbar(1,:);
    e1 = e1 / norm(e1);
    a  = xbar(4,:) - xbar(1,:);
    e2 = a - (a * e1.') * e1;
    e2 = e2 / norm(e2);
    e3 = cross(e1, e2);
    Rc = [e1; e2; e3];
else
    error('dlfe_canonical_frame:Dim', 'Nur 2D und 3D.');
end

chat = reshape((xbar * Rc.').', [], 1);
end
