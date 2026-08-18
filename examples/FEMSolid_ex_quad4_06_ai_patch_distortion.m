%FEMSOLID_EX_QUAD4_06_AI_PATCH_DISTORTION Verzerrungs-Robustheit des KI-quad4.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Untersucht die Generalisierungsfaehigkeit des Deep-Learned-quad4-Elements
%   (ganze Steifigkeitsmatrix Ke aus dem Netz) gegenueber GEOMETRISCH
%   VERZERRTEN Elementen -- ein Aspekt, den der regulaere Benchmark
%   (FEMSolid_ex_quad4_05_ai_benchmark.m) mit seinen nahezu quadratischen
%   Elementen nicht abdeckt.
%
%   Aufbau: ein Patch aus VIER quad4-Elementen mit einem gemeinsamen
%   mittleren Knoten (9-Knoten-Patch). Der aeussere Rand (8 Knoten) bleibt
%   fest; der mittlere Knoten wird schrittweise entlang einer Richtung aus
%   seiner idealen Mittelposition verschoben (Verschiebungsradius r). Mit
%   wachsendem r werden die vier Elemente zunehmend unterschiedlich stark
%   verzerrt (gestaucht, gestreckt, geschert).
%
%   Fuer jede Verschiebung wird je Element verglichen:
%     - Ke_analytisch : klassische 2x2-Gauss-Integration (element_quad4_lin)
%     - Ke_KI         : Netz-Vorhersage            (element_quad4_lin_ai)
%   und der relative Frobenius-Fehler  ||Ke_KI - Ke_analytisch|| / ||Ke_analytisch||
%   ausgewertet.
%
%   Es sind KEINE Randbedingungen, Lasten oder globalen FEM-Loesungen noetig
%   -- verglichen werden ausschliesslich die Element-Steifigkeitsmatrizen.
%
%   FEST (Gueltigkeitsbereich des Netzes): Hooke, planeStrain, nu = 0.3.
%   E und Dicke d sind frei (Ke = E*d*Khat) und beeinflussen den relativen
%   Fehler nicht.
%
% PREREQUISITE
%   Run train_quad4_K_network.py first to generate quad4_K_network.mat.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-07-01
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

close all; clear; clc;

fprintf('=== quad4 Patch-Test: KI-Robustheit gegen Elementverzerrung ===\n');
fprintf('    4 Elemente, gemeinsamer Mittelknoten wird verschoben\n\n');

% ------------------------------------------------------------------------
% Material (nu und Zustand fest ins Netz eintrainiert; E, d frei)
% ------------------------------------------------------------------------
E_MOD   = 3.0e7;     % E-Modul [kN/m^2] (~ C30/37)
NU      = 0.3;       % Querkontraktion (MUSS zum Netz passen)
D_THICK = 0.30;      % Dicke [m]
ALPHAT  = 0;         % kein Temperaturanteil noetig
mat_e   = [E_MOD, NU, D_THICK, ALPHAT];
MATNAME = 'Hooke';
MATCOND = 'planeStrain';

% ------------------------------------------------------------------------
% Aufrufargumente der Element-Routinen (keine Last, kein DeltaT)
% 2x2-Gauss identisch zur Regel, mit der das Netz trainiert wurde.
% ------------------------------------------------------------------------
gauss  = gauss_library('quad4', '2x2');
gp     = gauss.gp;
w      = gauss.w;
b_e    = zeros(2, 1);
Ue     = zeros(8, 1);
DeltaT = 0;
opts   = struct();

% ------------------------------------------------------------------------
% Basis-Patch: 2x2-Quadrat, 8 aeussere Knoten + Mittelknoten (Nr. 9)
%
%     7----6----5          Knoten 1..8 : fester Rand
%     |  E4 |  E3 |         Knoten 9    : Mittelknoten (wird verschoben)
%     8----9----4          Elemente (jeweils gegen den Uhrzeigersinn):
%     |  E1 |  E2 |           E1 = 1-2-9-8   E2 = 2-3-4-9
%     1----2----3            E3 = 9-4-5-6   E4 = 8-9-6-7
% ------------------------------------------------------------------------
baseNodes = [ -1 -1;    % 1  Ecke unten links
               0 -1;    % 2  Mitte unten
               1 -1;    % 3  Ecke unten rechts
               1  0;    % 4  Mitte rechts
               1  1;    % 5  Ecke oben rechts
               0  1;    % 6  Mitte oben
              -1  1;    % 7  Ecke oben links
              -1  0;    % 8  Mitte links
               0  0];   % 9  Mittelknoten (ideal zentriert)
centerId = 9;
elem = [ 1 2 9 8
         2 3 4 9
         9 4 5 6
         8 9 6 7 ];
nEl = size(elem, 1);

% ------------------------------------------------------------------------
% Verschiebung des Mittelknotens: feste Richtung, wachsender Radius r
% ------------------------------------------------------------------------
theta_deg = 45;                                  % Richtung (Grad); 45 = zur Ecke 5
dir_vec   = [cosd(theta_deg), sind(theta_deg)];
radii     = linspace(0, 1.2, 49);                % Verschiebungsradius
nR        = numel(radii);

errFro   = nan(nR, nEl);   % rel. Frobenius-Fehler je Radius/Element [%]
elemValid = true(nR, nEl); % detJ > 0 an allen Gausspunkten?

for ir = 1:nR
    coord = baseNodes;
    coord(centerId, :) = baseNodes(centerId, :) + radii(ir) * dir_vec;

    for e = 1:nEl
        ce = coord(elem(e, :), :);

        % Zulaessigkeit: Element darf nicht entarten/umklappen
        if ~element_is_valid(ce, gp)
            elemValid(ir, e) = false;
            continue;
        end

        Kf = element_quad4_lin(   ce, mat_e, b_e, DeltaT, Ue, [], gp, w, MATNAME, MATCOND, opts);
        Ka = element_quad4_lin_ai(ce, mat_e, b_e, DeltaT, Ue, [], gp, w, MATNAME, MATCOND, opts);

        errFro(ir, e) = norm(Ka - Kf, 'fro') / max(norm(Kf, 'fro'), eps) * 100;
    end
end

allValid = all(elemValid, 2);                    % alle 4 Elemente zulaessig?
meanErr  = mean(errFro, 2, 'omitnan');           % Mittel ueber die 4 Elemente

% Mittelkurve nur zeigen, wo ALLE vier Elemente gueltig sind -- sonst
% taeuscht der Wegfall des (grossen) E3-Fehlers einen kuenstlichen
% Fehlerabfall vor.
meanErrPlot            = meanErr;
meanErrPlot(~allValid) = NaN;

% letzter Radius, an dem ALLE vier Elemente noch gueltig sind
rValidMax = radii(find(allValid, 1, 'last'));

% ------------------------------------------------------------------------
% Konsolenausgabe (Kurztabelle)
% ------------------------------------------------------------------------
fprintf('Richtung der Verschiebung: %g Grad | Radien: %g .. %g (%d Schritte)\n', ...
        theta_deg, radii(1), radii(end), nR);
fprintf('Alle 4 Elemente gueltig bis r = %.3f\n\n', rValidMax);
fprintf('   r     | E1     | E2     | E3     | E4     | Mittel\n');
fprintf('  -------|--------|--------|--------|--------|--------\n');
for ir = 1:4:nR
    fprintf('  %5.3f  | %6s | %6s | %6s | %6s | %6s\n', radii(ir), ...
        fmt(errFro(ir,1)), fmt(errFro(ir,2)), fmt(errFro(ir,3)), ...
        fmt(errFro(ir,4)), fmt(meanErr(ir)));
end

% ------------------------------------------------------------------------
% (A) Hauptgrafik: relativer Fehler ueber Verschiebungsradius
% ------------------------------------------------------------------------
figure('Name', 'Patch-Test: KI-Fehler vs. Verzerrung', 'NumberTitle', 'off', ...
       'Position', [80 80 900 560]);
colors = lines(nEl);
hold on;
for e = 1:nEl
    plot(radii, errFro(:, e), '-o', 'Color', colors(e,:), ...
         'MarkerSize', 4, 'LineWidth', 1.2, 'DisplayName', sprintf('Element %d', e));
end
plot(radii, meanErrPlot, 'k-', 'LineWidth', 2.4, ...
     'DisplayName', 'Mittel (alle 4 gueltig)');
if ~isempty(rValidMax)
    xline_compat(rValidMax, 'E3 wird ungueltig');
end
xlabel('Verschiebungsradius r des Mittelknotens [-]');
ylabel('rel. Ke-Fehler (Frobenius) [%]');
title(sprintf(['KI-quad4: Fehler vs. Elementverzerrung ' ...
               '(Verschiebungsrichtung %g' char(176) ')'], theta_deg));
legend('Location', 'northwest');
grid on; box on;

% ------------------------------------------------------------------------
% (B) Richtungsunabhaengigkeit des Trends (mehrere Verschiebungsrichtungen)
%   Zeigt, dass der Fehleranstieg nicht an einer speziellen Richtung haengt.
% ------------------------------------------------------------------------
dirsDeg = 0:15:165;                              % 12 Richtungen (0..180, symm.)
meanErrDir = nan(nR, numel(dirsDeg));
for id = 1:numel(dirsDeg)
    dv = [cosd(dirsDeg(id)), sind(dirsDeg(id))];
    for ir = 1:nR
        coord = baseNodes;
        coord(centerId, :) = baseNodes(centerId, :) + radii(ir) * dv;
        ev = zeros(1, nEl);
        for e = 1:nEl
            ce = coord(elem(e, :), :);
            if ~element_is_valid(ce, gp)
                ev(e) = NaN; continue;
            end
            Kf = element_quad4_lin(   ce, mat_e, b_e, DeltaT, Ue, [], gp, w, MATNAME, MATCOND, opts);
            Ka = element_quad4_lin_ai(ce, mat_e, b_e, DeltaT, Ue, [], gp, w, MATNAME, MATCOND, opts);
            ev(e) = norm(Ka - Kf, 'fro') / max(norm(Kf, 'fro'), eps) * 100;
        end
        % Nur werten, wenn ALLE vier Elemente gueltig sind -- sonst wuerde der
        % Wegfall des groessten Element-Fehlers einen Abfall vortaeuschen.
        if ~any(isnan(ev))
            meanErrDir(ir, id) = mean(ev);
        end
    end
end

% Achse auf den Bereich begrenzen, in dem ALLE Richtungen voll vergleichbar
% sind (in jeder Richtung alle vier Elemente gueltig).
rCleanMax = radii(find(all(~isnan(meanErrDir), 2), 1, 'last'));

figure('Name', 'Patch-Test: Richtungsunabhaengigkeit', 'NumberTitle', 'off', ...
       'Position', [120 120 900 560]);
hold on;
hgray = plot(radii, meanErrDir, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
band_lo = min(meanErrDir, [], 2, 'omitnan');
band_hi = max(meanErrDir, [], 2, 'omitnan');
band_mu = mean(meanErrDir, 2, 'omitnan');
valid_b = ~isnan(band_mu);
fill([radii(valid_b), fliplr(radii(valid_b))], ...
     [band_lo(valid_b)', fliplr(band_hi(valid_b)')], ...
     [0.30 0.55 0.85], 'FaceAlpha', 0.20, 'EdgeColor', 'none');
plot(radii, band_mu, 'b-', 'LineWidth', 2.6);
xlabel('Verschiebungsradius r des Mittelknotens [-]');
ylabel('mittl. rel. Ke-Fehler (4 Elemente) [%]');
title('Fehleranstieg unabhaengig von der Verschiebungsrichtung');
legend([hgray(1), plot(nan,nan,'b-','LineWidth',2.6)], ...
       {'einzelne Richtungen', 'Mittel ueber Richtungen'}, 'Location', 'northwest');
if ~isempty(rCleanMax)
    xlim([0 rCleanMax]);
end
grid on; box on;

% ------------------------------------------------------------------------
% (C) Patch-Geometrie bei ausgewaehlten Radien (Anschauung)
% ------------------------------------------------------------------------
rShow = [0, 0.3, 0.6, min(0.9, rValidMax)];
figure('Name', 'Patch-Test: Geometrie bei wachsender Verzerrung', ...
       'NumberTitle', 'off', 'Position', [160 160 1200 320]);
for k = 1:numel(rShow)
    subplot(1, numel(rShow), k);
    coord = baseNodes;
    coord(centerId, :) = baseNodes(centerId, :) + rShow(k) * dir_vec;
    draw_patch(coord, elem);
    title(sprintf('r = %.2f', rShow(k)));
    axis equal; axis([-1.15 1.15 -1.15 1.15]); box on;
end
sgtitle('Verzerrung des Patches: Mittelknoten wird aus der Mitte verschoben');

fprintf('\nFertig. Grafiken erzeugt.\n');


% ========================================================================
% Hilfsfunktionen
% ========================================================================
function ok = element_is_valid(ce, gp)
%ELEMENT_IS_VALID True, wenn detJ an allen Gausspunkten positiv ist.
    ok = true;
    for g = 1:size(gp, 1)
        [~, ~, detJ] = shape_quad4(ce, gp(g, :));
        if detJ <= 1e-12
            ok = false;
            return;
        end
    end
end

function s = fmt(v)
%FMT Formatiert einen Fehlerwert (oder '  -  ' bei ungueltigem Element).
    if isnan(v)
        s = '  -   ';
    else
        s = sprintf('%5.2f%%', v);
    end
end

function draw_patch(coord, elem)
%DRAW_PATCH Zeichnet die vier Elemente des Patches mit Knoten.
    hold on;
    cols = lines(size(elem, 1));
    for e = 1:size(elem, 1)
        n = coord(elem(e, :), :);
        patch('XData', n(:,1), 'YData', n(:,2), ...
              'FaceColor', cols(e,:), 'FaceAlpha', 0.25, ...
              'EdgeColor', cols(e,:), 'LineWidth', 1.4);
        c = mean(n, 1);
        text(c(1), c(2), sprintf('E%d', e), 'HorizontalAlignment', 'center', ...
             'FontSize', 8, 'FontWeight', 'bold');
    end
    plot(coord(:,1), coord(:,2), 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', 5);
    plot(coord(end,1), coord(end,2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7);
end

function xline_compat(xval, label)
%XLINE_COMPAT Vertikale Referenzlinie (auch ohne xline-Funktion).
    yl = ylim; hold on;
    plot([xval xval], yl, 'k--', 'LineWidth', 1.2, 'HandleVisibility', 'off');
    text(xval, yl(1) + 0.5*(yl(2)-yl(1)), [' ' label], ...
         'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
         'FontSize', 8, 'Rotation', 90);
end
