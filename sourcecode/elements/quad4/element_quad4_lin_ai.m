function [Ke, Fbe, Fte, Finte, history] = element_quad4_lin_ai(coord_e, mat_e, b_e, DeltaT_e, Ue, history, gp, w, MATNAME, MATCOND, opts)
%ELEMENT_QUAD4_LIN_AI Computes the element stiffness with a learned Ke matrix.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Implementiert den "Deep Learned Finite Elements"-Ansatz fuer das
%   bilineare Viereckselement (quad4, lineare Analyse) in der Variante
%   "ganze Steifigkeitsmatrix".
%
%   Anders als die B-Matrix-Variante lernt das Netz hier nicht die
%   Formfunktionsableitungen, sondern direkt die gesamte symmetrische
%   Element-Steifigkeitsmatrix Ke (8x8). Es gibt keine Gauss-Schleife und
%   keine B-Matrix mehr fuer die Steifigkeit -- ein einziger Forward-Pass
%   liefert die komplette Matrix.
%
%   Zwei physikalische Eigenschaften werden ausgenutzt:
%     1. Groessen- UND Rotationsinvarianz: In der ebenen Elastizitaet ist Ke
%        unabhaengig von der absoluten Elementgroesse. Die Geometrie wird auf
%        Schwerpunkt und charakteristische Laenge Lc normalisiert. ZUSAETZLICH
%        wird die Rotation entfernt (Kanonisierung): das Element wird so
%        gedreht, dass die Kante Knoten1->Knoten2 auf der +x-Achse liegt. Das
%        Netz lernt Ke der KANONISCHEN Form; die Vorhersage wird hier exakt
%        zurueckgedreht:  Ke_hat = Tc' * Khat_canon * Tc  mit
%        Tc = blockdiag(Rc,Rc,Rc,Rc) (Rc = Kanonisierungs-Drehung). Ke ist
%        rotationsequivariant, die Ruecktransformation ist exakt.
%     2. Linearitaet in E*d: Ke = E * d * Khat. Das Netz liefert die
%        dimensionslose Form-Steifigkeit Khat (trainiert mit E=d=1), hier
%        wird mit dem tatsaechlichen E*d multipliziert.
%
%   Querkontraktionszahl nu und ebener Zustand (planeStrain/planeStress)
%   sind FEST ins Netz eintrainiert (siehe quad4_K_network.mat).
%
%   Netzarchitektur (muss zur trainierten Variante passen):
%     3 verdeckte Schichten a 32 Neuronen, Aktivierung GELU
%     (FC(8)-GELU-FC(32)-GELU-FC(32)-GELU-FC(32)-FC(36)).
%     Die Hidden-Groesse wird dynamisch aus den Gewichten gelesen; nur die
%     Aktivierung (GELU) und die Schichtanzahl (4 Gewichtslagen W1..W4)
%     sind im Forward-Pass unten festgelegt.
%
%   Netzwerkein- und -ausgabe:
%     Eingang  (8):  [x1_c, y1_c, ..., x4_c, y4_c]  (kanonisiert: Schwerpunkt/Lc
%                    + Kante Knoten1->Knoten2 auf +x gedreht)
%     Ausgang (36):  oberes Dreieck (column-major) von Khat der KANONISCHEN Form
%
%   Die Lastvektoren Fbe (Volumenlast) und Fte (Temperatur) werden weiterhin
%   analytisch ueber die Formfunktionen berechnet -- nur die Steifigkeit
%   stammt aus dem Netz.
%
% PREREQUISITE
%   Run train_quad4_K_network.py first to generate quad4_K_network.mat.
%
% INPUT / OUTPUT
%   Identisch zu element_quad4_lin.m.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-06-25
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

% ------------------------------------------------------------------------
% Netzwerk einmalig laden
% ------------------------------------------------------------------------

persistent net_quad4 W1 W2 W3 W4 b1 b2 b3 b4 recon_map cfg_checked inv_sqrt2

if isempty(net_quad4)
    netFile = fullfile(fileparts(mfilename('fullpath')), 'quad4_K_network.mat');
    if ~exist(netFile, 'file')
        error(['quad4_K_network.mat nicht gefunden.\n' ...
               'Bitte zuerst train_quad4_K_network.py ausfuehren.']);
    end
    net_quad4 = load(netFile);   % W1..W4, b1..b4, nu_train, condition, ...

    % Gewichte und Bias-Spaltenvektoren EINMALIG aufbereiten (nicht pro Aufruf).
    W1 = net_quad4.W1;  b1 = net_quad4.b1(:);
    W2 = net_quad4.W2;  b2 = net_quad4.b2(:);
    W3 = net_quad4.W3;  b3 = net_quad4.b3(:);
    W4 = net_quad4.W4;  b4 = net_quad4.b4(:);

    % Rekonstruktionstabelle: 64x1-Map, die den 36-Vektor (oberes Dreieck,
    % column-major -- identisch zum Python-Export) in einem einzigen Gather
    % zur vollen symmetrischen 8x8 expandiert. Keine Temp-Matrizen pro Aufruf.
    M           = zeros(8);
    M(triu(true(8))) = 1:36;        % obere Eintraege -> Position im 36-Vektor
    M           = M + triu(M, 1).'; % spiegeln -> volle 8x8 Index-Tabelle
    recon_map   = M(:);             % 64x1, Werte 1..36

    inv_sqrt2   = 1 / sqrt(2);      % Konstante fuer die GELU-Aktivierung
    cfg_checked = false;            % Konsistenzpruefung nur einmal ausfuehren
end

% ------------------------------------------------------------------------
% Elementparameter
% ------------------------------------------------------------------------

DIM    = size(coord_e, 2);
DOF    = DIM;
NNEL   = size(coord_e, 1);
NDOFEL = NNEL * DOF;

Ke    = zeros(NDOFEL, NDOFEL);
Fbe   = zeros(NDOFEL, 1);
Fte   = zeros(NDOFEL, 1);
Finte = zeros(NDOFEL, 1);

Emod   = mat_e(1);
nu     = mat_e(2);
d      = mat_e(3);
alphaT = mat_e(4);

% Konsistenzpruefung: Netz ist nur fuer sein Trainingsmaterial gueltig
% (E und d werden analytisch beruecksichtigt, nu und Zustand sind fest).
% Laeuft nur EINMAL pro Session (nicht im Hot-Path jedes Elementaufrufs).
if ~cfg_checked
    if isfield(net_quad4, 'nu_train') && abs(nu - net_quad4.nu_train) > 1e-6
        warning('element_quad4_lin_ai:Material', ...
            ['Das Netz wurde fuer nu = %.3f trainiert, das Modell verwendet ' ...
             'nu = %.3f. Die gelernte Steifigkeit ist nur fuer das ' ...
             'Trainings-nu exakt.'], net_quad4.nu_train, nu);
    end
    if isfield(net_quad4, 'condition') && ~strcmpi(strtrim(char(net_quad4.condition)), MATCOND)
        warning('element_quad4_lin_ai:Condition', ...
            ['Das Netz wurde fuer "%s" trainiert, das Modell verwendet "%s".'], ...
            strtrim(char(net_quad4.condition)), MATCOND);
    end
    % Aktivierung muss zum Forward-Pass passen (hier GELU). Schuetzt davor,
    % versehentlich ein mit anderer Aktivierung trainiertes Netz zu laden.
    if isfield(net_quad4, 'activation') && ~strcmpi(strtrim(char(net_quad4.activation)), 'GELU')
        warning('element_quad4_lin_ai:Activation', ...
            ['Das geladene Netz nutzt die Aktivierung "%s", dieses Element ' ...
             'rechnet aber mit GELU. Forward-Pass und Netz passen nicht zusammen.'], ...
            strtrim(char(net_quad4.activation)));
    end
    % Dieses Element kanonisiert die Eingabe (Kante 1->2 -> +x) und dreht die
    % Vorhersage zurueck. Ein Netz, das OHNE diese Kanonisierung trainiert wurde,
    % passt nicht -> warnen (Feld fehlt => altes Netz).
    canon_expected = 'edge_n1n2_to_pos_x_after_centroid_Lc';
    if ~isfield(net_quad4, 'canonicalization') || ...
            ~strcmpi(strtrim(char(net_quad4.canonicalization)), canon_expected)
        warning('element_quad4_lin_ai:Canonicalization', ...
            ['Das geladene Netz traegt nicht die erwartete Kanonisierung "%s". ' ...
             'Dieses Element dreht Eingabe/Vorhersage rotationskanonisch -- ein ' ...
             'ohne diese Kanonisierung trainiertes Netz liefert falsche Ke.'], ...
            canon_expected);
    end
    cfg_checked = true;
end

% ------------------------------------------------------------------------
% Normalisierung (Schwerpunkt + charakteristische Laenge Lc)
% ------------------------------------------------------------------------

centroid    = mean(coord_e, 1);
dists       = sqrt(sum((coord_e - centroid).^2, 2));
Lc          = mean(dists);
if Lc <= eps
    error('element_quad4_lin_ai:DegenerateElement', 'Degeneriertes Element: Lc ist numerisch null.');
end
coords_norm = (coord_e - centroid) / Lc;

% Rotations-Kanonisierung (Feature): Kante Knoten1->Knoten2 auf die +x-Achse
% drehen. Muss EXAKT zur Trainings-Kanonisierung passen (canonicalize_coords in
% train_quad4_K_network.py). Alle rotierten Varianten einer Form ergeben so
% dieselbe Netzeingabe.
e12  = coords_norm(2, :) - coords_norm(1, :);
phi  = atan2(e12(2), e12(1));
cphi = cos(phi);  sphi = sin(phi);
Rc   = [ cphi, sphi; -sphi, cphi ];           % Drehung um -phi (Kante 1->2 -> +x)
coords_canon = coords_norm * Rc.';            % (4x2) kanonisch orientiert
coords_flat  = reshape(coords_canon.', [], 1);  % (8x1): [x1c; y1c; x2c; y2c; ...]

% ------------------------------------------------------------------------
% Netzwerkvorhersage der gesamten Steifigkeitsmatrix (ein Forward-Pass)
% ------------------------------------------------------------------------

% GELU-Aktivierung in exakter erf-Form -- identisch zu PyTorch nn.GELU()
% (Default approximate='none'):  GELU(z) = 0.5 * z * (1 + erf(z / sqrt(2))).
% Die Pre-Aktivierung z wird zwischengespeichert, damit W*x+b nur einmal
% berechnet wird.
z       = W1 * coords_flat + b1;  x = 0.5 * z .* (1 + erf(z * inv_sqrt2));
z       = W2 * x + b2;            x = 0.5 * z .* (1 + erf(z * inv_sqrt2));
z       = W3 * x + b3;            x = 0.5 * z .* (1 + erf(z * inv_sqrt2));
ke_triu = W4 * x + b4;              % (36x1) oberes Dreieck (column-major)

% Symmetrische Matrix der KANONISCHEN Form rekonstruieren -- ein einziger
% Gather + reshape. recon_map wurde einmalig beim ersten Aufruf berechnet.
% Symmetrie damit strukturell exakt.
Khat_canon = reshape(ke_triu(recon_map), 8, 8);

% Ruecktransformation in die tatsaechliche Orientierung (Rotationsequivarianz):
%   Ke_hat = Tc' * Khat_canon * Tc,   Tc = blockdiag(Rc, Rc, Rc, Rc).
% Rc ist orthogonal -> exakt. kron(eye(4),Rc) ist die 8x8-Blockdiagonale.
Tc   = kron(eye(NNEL), Rc);
Khat = Tc.' * Khat_canon * Tc;

% E und d analytisch: Ke = E * d * Khat
Ke = Emod * d * Khat;

% ------------------------------------------------------------------------
% Starrkoerpermoden exakt erzwingen (Konsistenz / Patch-Test)
%
%   Die analytische Ke hat einen exakten Nullraum aus 3 Starrkoerpermoden
%   (2 Translationen + 1 Rotation). Das gelernte Ke erfuellt das nur
%   naeherungsweise -> kleine Matrixfehler erzeugen spurious Verspannung,
%   die sich beim Loesen stark verstaerkt. Die Moden sind aus der Geometrie
%   exakt bekannt; die symmetrische Projektion  Ke <- P*Ke*P  mit
%   P = I - Q*Q' (Q = orthonormale Basis der Moden) setzt Ke*Mode = 0 exakt.
%
%   Die drei Moden sind hier bereits paarweise orthogonal:
%     Tx'*Ty   = 0 (disjunkte Eintraege)
%     Tx'*Rrot = -sum(y - yc) = 0  (yc = Mittelwert)
%     Ty'*Rrot =  sum(x - xc) = 0  (xc = Mittelwert)
%   Deshalb genuegt SPALTENWEISE NORMIERUNG -- kein orth()/SVD pro Aufruf.
% ------------------------------------------------------------------------
Tx = repmat([1; 0], NNEL, 1);
Ty = repmat([0; 1], NNEL, 1);
Rrot = zeros(NDOFEL, 1);
Rrot(1:2:end) = -(coord_e(:,2) - centroid(2));   % -(y - yc)
Rrot(2:2:end) =  (coord_e(:,1) - centroid(1));   %  (x - xc)
Qrb = [Tx/norm(Tx), Ty/norm(Ty), Rrot/norm(Rrot)];   % orthonormal (s.o.)
P   = eye(NDOFEL) - Qrb * Qrb';
Ke  = P * Ke * P;
Ke  = 0.5 * (Ke + Ke.');                          % Symmetrie nach Rundung sichern

% ------------------------------------------------------------------------
% Lastvektoren analytisch (Volumenlast Fbe, Temperaturlast Fte)
% ------------------------------------------------------------------------

if isempty(DeltaT_e)
    DeltaT = 0;
else
    DeltaT = DeltaT_e(1);
end
epsT = alphaT * DeltaT * [1; 1; 0];

needFte = any(epsT ~= 0);
needFbe = any(b_e ~= 0);
if needFte
    GradU  = zeros(2);
    [~, C] = material_elasticity(mat_e, GradU, MATNAME, MATCOND);
end

% Gauss-Schleife nur, wenn ueberhaupt eine Volumen- oder Temperaturlast
% vorliegt. Bei reinen Steifigkeitsproblemen (Fbe = Fte = 0) wird sie
% komplett uebersprungen -> spart 4 shape_quad4-Auswertungen pro Element.
numgp = size(gp, 1);
if ~(needFbe || needFte)
    numgp = 0;
end
for i = 1:numgp
    rst_loc = gp(i, :);

    [h, dh_dx, detJ] = shape_quad4(coord_e, rst_loc);

    H = [ h(1)  0     h(2)  0     h(3)  0     h(4)  0
          0     h(1)  0     h(2)  0     h(3)  0     h(4) ];

    dV = detJ * d * w(i);

    if needFbe
        Fbe = Fbe + H' * b_e * dV;
    end

    if needFte
        h_x = dh_dx(:,1);
        h_y = dh_dx(:,2);
        B = [ h_x(1)  0       h_x(2)  0       h_x(3)  0       h_x(4)  0
              0       h_y(1)  0       h_y(2)  0       h_y(3)  0       h_y(4)
              h_y(1)  h_x(1)  h_y(2)  h_x(2)  h_y(3)  h_x(3)  h_y(4)  h_x(4) ];
        Fte = Fte + B' * C * epsT * dV;
    end
end

% Innere Kraefte
Finte = Ke * Ue;

% ------------------------------------------------------------------------
end
