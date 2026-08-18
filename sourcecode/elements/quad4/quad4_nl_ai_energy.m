function [Wphys, Finte, Ke, dg, meta] = quad4_nl_ai_energy(coord_e, mat_e, Ue)
%QUAD4_NL_AI_ENERGY Kanonisierungs- und Ko-Rotations-Kette des KI-quad4.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Bringt ein physisches Element in den kanonischen Rahmen, wertet dort das
%   Energiemodell aus (quad4_nl_ai_model) und transformiert Energie, erste
%   und zweite Ableitung EXAKT zurueck:
%
%     W_phys = E*d*Lc^2 * What(chat, z)
%     Finte  = dW_phys/dUe
%     Ke     = d2W_phys/dUe2 = dFinte/dUe        (Konsistenz per Konstruktion)
%
%   Die Kette besteht aus (alles exakt, nichts gelernt):
%     1. Geometrie-Kanonisierung  chat = Rc*(coord - centroid)/Lc
%     2. Zustands-Kanonisierung   v    = P*(Tc*Ue)/Lc
%     3. Ko-Rotation              z    = P*(B(theta)*(chat+v) - chat)
%     4. Rueckskalierung          Tc', P, E*d(*Lc)
%
%   Der Ko-Rotationswinkel theta = atan2(b, a) haengt von Ue ab und wird
%   NICHT eingefroren, sondern per Kettenregel exakt mitgefuehrt (a, b sind
%   linear in v -> geschlossene Formeln fuer grad(theta) und hess(theta)).
%   Ein Einfrieren waere nur bis O(Netzfehler) korrekt und wuerde die
%   FD-Gates reissen.
%
%   Verifiziert (Gates a/b in training/quad4/quad4_nl_ref.py, ohne Netz):
%   die Kette mit ANALYTISCHER Energie reproduziert das analytische Element
%   auf ~1e-13 und die zentralen Differenzen des Komposits auf ~1e-9.
%
% INPUT
%   coord_e  (4x2) Elementknotenkoordinaten
%   mat_e    (1xn) Materialkarte [E, nu, d, ...]
%   Ue       (8x1) Elementverschiebungen
%
% OUTPUT
%   Wphys    Formaenderungsenergie des Elements
%   Finte    (8x1) innerer Kraftvektor
%   Ke       (8x8) tangentiale Steifigkeitsmatrix
%   dg       Diagnose (maxE am Mittelpunkt, Lc, theta) -- nur auf Anforderung
%   meta     Netz-Metadaten -- nur auf Anforderung
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-08-18
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

persistent P Sigma

if isempty(P)
    tx = repmat([1; 0], 4, 1);
    ty = repmat([0; 1], 4, 1);
    P     = eye(8) - (tx*tx.' + ty*ty.') / 4;      % Translationsprojektor
    Sigma = kron(eye(4), [0 1; -1 0]);             % dR(-th)/dth = R(-th)*Sigma
end

Emod = mat_e(1);
d    = mat_e(3);

% ------------------------------------------------------------------------
% 1. Geometrie-Kanonisierung
% ------------------------------------------------------------------------
centroid = mean(coord_e, 1);
Lc       = mean(sqrt(sum((coord_e - centroid).^2, 2)));
if Lc <= eps
    error('quad4_nl_ai_energy:DegenerateElement', ...
          'Degeneriertes Element: Lc ist numerisch null.');
end
coords_norm = (coord_e - centroid) / Lc;

e12  = coords_norm(2, :) - coords_norm(1, :);
phi  = atan2(e12(2), e12(1));
cphi = cos(phi);  sphi = sin(phi);
Rc   = [ cphi, sphi; -sphi, cphi ];                % dreht Kante 1->2 auf +x
coords_canon = coords_norm * Rc.';                 % (4x2)
xhat = reshape(coords_canon.', [], 1);             % (8x1)
Tc   = kron(eye(4), Rc);

% ------------------------------------------------------------------------
% 2. Zustands-Kanonisierung
% ------------------------------------------------------------------------
v = P * (Tc * Ue) / Lc;

% ------------------------------------------------------------------------
% 3. Ko-Rotation mit exakten Ableitungen (a, b sind LINEAR in v)
% ------------------------------------------------------------------------
[~, dh0, ~] = shape_quad4(coords_canon, [0 0]);
hx0 = dh0(:, 1);  hy0 = dh0(:, 2);
a1 = zeros(8, 1);  b1 = zeros(8, 1);
a1(1:2:7) =  hx0;  a1(2:2:8) =  hy0;               % a = tr(F) = 2 + a1'*v
b1(1:2:7) = -hy0;  b1(2:2:8) =  hx0;               % b = F21 - F12 = b1'*v

aa = 2 + a1.' * v;
bb = b1.' * v;
r2 = aa*aa + bb*bb;
th = atan2(bb, aa);

gth = (aa * b1 - bb * a1) / r2;
Hth = ( (bb*bb - aa*aa) * (b1*a1.' + a1*b1.') ...
      + 2*aa*bb * (a1*a1.' - b1*b1.') ) / (r2*r2);

cth = cos(th);  sth = sin(th);
Bm  = kron(eye(4), [cth, sth; -sth, cth]);         % B = kron(I4, R(-theta))

y = xhat + v;
z = P * (Bm * y - xhat);

% ------------------------------------------------------------------------
% 4. Kanonisches Energiemodell (K0-Split + Residual-Netz)
% ------------------------------------------------------------------------
if nargout > 4
    [What, p, H, meta] = quad4_nl_ai_model(xhat, z);
else
    [What, p, H] = quad4_nl_ai_model(xhat, z);
end

% ------------------------------------------------------------------------
% 5. Kette: kanonische Ableitungen -> physikalische Groessen
%    dz/dv = P*Gt  mit  Gt = B + (B*Sigma*y)*grad(theta)'
%    C_v   = Beitrag der theta-Abhaengigkeit zur zweiten Ableitung
% ------------------------------------------------------------------------
pt  = P * p;
BSy = Bm * (Sigma * y);
Gt  = Bm + BSy * gth.';

g_v = Gt.' * pt;

SBp = Sigma * (Bm.' * pt);
C_v = -(SBp * gth.') - (gth * SBp.') ...
      - (pt.' * (Bm * y)) * (gth * gth.') ...
      + (pt.' * BSy) * Hth;
K_v = Gt.' * P * H * P * Gt + C_v;

Wphys = Emod * d * Lc^2 * What;
Finte = Emod * d * Lc * (Tc.' * (P * g_v));
Ke    = Emod * d * (Tc.' * (P * K_v * P) * Tc);
Ke    = 0.5 * (Ke + Ke.');                         % Rundungspolitur

% ------------------------------------------------------------------------
% 6. Diagnose: billiger OOD-Proxy (E_green am Elementmittelpunkt).
%    Der Verschiebungsgradient liegt fuer die Ko-Rotation ohnehin vor.
% ------------------------------------------------------------------------
if nargout > 3
    Un   = reshape(v, 2, 4).';
    Fdef = eye(2) + Un.' * dh0;
    Eg   = 0.5 * (Fdef.' * Fdef - eye(2));
    dg.maxE  = norm(Eg, 'fro');
    dg.Lc    = Lc;
    dg.theta = th;
end

% ------------------------------------------------------------------------
end
