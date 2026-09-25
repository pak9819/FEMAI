function T = matlab_element_timing(netName, nRep)
%MATLAB_ELEMENT_TIMING Mikrosekunden pro Elementaufruf in MATLAB (brick8).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Gegenstueck zu brick8_bench.c: misst element_brick8_nl (analytisch,
%   StVenant) und element_brick8_nl_ai (Netz netName) auf 200 zufaelligen
%   verzerrten Hexaedern mit moderaten Zustaenden. Median aus 5 Laeufen.
%   Die CPU sollte dabei frei sein.
%
%   T = matlab_element_timing('brick8_nl_W_network_h32d3.mat')
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-25
% ------------------------------------------------------------------------

if nargin < 2, nRep = 10; end
setenv('BRICK8_NET_FILE', netName);
clear element_brick8_nl_ai brick8_nl_ai_energy
rng(2026);
mat_e = [1000 0.3 NaN 0];
g = gauss_library('brick8', 'default');
b0 = zeros(3,1);  op = struct();
ne = 200;
X = cell(ne,1);  U = cell(ne,1);
base = [-1 -1 -1; 1 -1 -1; 1 1 -1; -1 1 -1; -1 -1 1; 1 -1 1; 1 1 1; -1 1 1] / 2;
for e = 1:ne
    S = diag(exp(log(0.5) + (log(2.0) - log(0.5)) * rand(1,3)));
    Xe = base * S * (eye(3) + 0.15 * (rand(3) - 0.5) .* (1 - eye(3))).' + 0.03 * (rand(8,3) - 0.5);
    X{e} = Xe;
    U{e} = reshape(((Xe - mean(Xe,1)) * (0.04 * randn(3)).').', [], 1);
end
old = warning('off', 'element_brick8_nl_ai:OutOfHull');
element_brick8_nl_ai(X{1}, mat_e, b0, 0, U{1}, [], g.gp, g.w, 'StVenant', '3D', op);   % Netz laden
tA = zeros(5,1);  tK = zeros(5,1);
for run = 1:5
    t0 = tic;
    for r = 1:nRep
        for e = 1:ne
            element_brick8_nl(X{e}, mat_e, b0, 0, U{e}, [], g.gp, g.w, 'StVenant', '3D', op);
        end
    end
    tA(run) = toc(t0) / (nRep * ne);
    t0 = tic;
    for r = 1:nRep
        for e = 1:ne
            element_brick8_nl_ai(X{e}, mat_e, b0, 0, U{e}, [], g.gp, g.w, 'StVenant', '3D', op);
        end
    end
    tK(run) = toc(t0) / (nRep * ne);
end
warning(old);
T = struct('net', netName, 'analytisch_us', 1e6 * median(tA), 'ki_us', 1e6 * median(tK));
fprintf('MATLAB %s: analytisch %.1f us | KI %.1f us | Speedup %.2fx\n', ...
    netName, T.analytisch_us, T.ki_us, T.analytisch_us / T.ki_us);
end
