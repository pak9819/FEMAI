%FEMSOLID_EX_BRICK8_08_AI_NL_CHECK Isolierter Element-Check brick8 (ohne Solver).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Vergleicht element_brick8_nl_ai DIREKT mit element_brick8_nl an EINEM
%   verzerrten Hexaeder. Der Verschiebungszustand (affin + nicht-affin +
%   Starrkoerperdrehung) wird von 0 bis ueber die Trainingshuelle
%   hochskaliert; je Amplitude: max ||E_green|| ueber die GP, "in Huelle",
%   rel. Fehler Ke (Frobenius, Spektral) und Finte.
%
%   Zusaetzlich: Skalierungscheck Ke ~ E und Ke(s*X, s*U) = s*Ke(X, U)
%   (exakte Faktorisierung W = E*Lc^3*What, unabhaengig von der Netzguete).
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

close all; clear all; clc; %#ok<CLALL>

E_MOD = 1000;  NU = 0.3;  MATNAME = 'StVenant';  MATCOND = '3D';
mat_e = [E_MOD NU NaN 0];
g = gauss_library('brick8', 'default');  gp = g.gp;  w = g.w;
b0 = zeros(3,1);  opts = struct();

NET = load(brick8_nl_ai_network_file(MATNAME));
E_MAX = double(NET.state_E_max);

X = [0 0 0; 1 0.1 0; 1.2 1 0.05; 0.1 0.9 0; 0 0.05 1; 1.1 0 0.9; 1.1 1.1 1.2; 0 1 1];
G = [0.08 0.05 -0.03; -0.04 0.10 0.02; 0.03 -0.02 0.06];
Ub = X * G.' + [0 0 0; 0.03 0 0.01; 0.05 0.04 0; 0 0.02 0.03; 0.01 0 0; 0 0.02 0; 0.03 0 0.02; 0 0 0.01];
th = 0.6;  ax = [1 2 3] / norm([1 2 3]);
Kx = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
R  = eye(3) + sin(th)*Kx + (1-cos(th))*Kx*Kx;

fprintf('=== brick8 nl KI-Element Check (vs. analytisch, ohne Solver) ===\n');
fprintf('Trainingshuelle ||E_green|| <= %.2f\n\n', E_MAX);
fprintf(' amp  | max||Eg|| | Huelle | KeErr Frob [%%] | KeErr Spek [%%] | FinteErr [%%]\n');
for a = [0 0.1 0.25 0.5 1.0 1.5 2.0 3.0]
    U  = (X + a*Ub) * R.' - X;              % mit Starrkoerperdrehung
    if a == 0, U = zeros(8,3); end
    Ue = reshape(U.', [], 1);
    mE = 0;
    for i = 1:8
        [~, dh] = shape_brick8(X, gp(i,:));
        F = eye(3) + U.' * dh;
        mE = max(mE, norm(0.5*(F.'*F - eye(3)), 'fro'));
    end
    [Ka,~,~,Fa] = element_brick8_nl(   X, mat_e, b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
    [Kk,~,~,Fk] = element_brick8_nl_ai(X, mat_e, b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
    fe = NaN;
    if norm(Fa) > 0, fe = norm(Fk - Fa) / norm(Fa) * 100; end
    hs = 'nein';  if mE <= E_MAX, hs = 'JA'; end
    fprintf(' %4.2f | %9.4f | %6s | %14.2f | %14.2f | %11.2f\n', a, mE, hs, ...
        norm(Kk-Ka,'fro')/norm(Ka,'fro')*100, norm(Kk-Ka,2)/norm(Ka,2)*100, fe);
end

Ue = reshape(((X + 0.5*Ub) * R.' - X).', [], 1);
[K1,~,~,F1] = element_brick8_nl_ai(X, [1 NU NaN 0], b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
[K2,~,~,F2] = element_brick8_nl_ai(X, [E_MOD NU NaN 0], b0, 0, Ue, [], gp, w, MATNAME, MATCOND, opts);
s = 3.7;
[K3,~,~,F3] = element_brick8_nl_ai(s*X, [1 NU NaN 0], b0, 0, s*Ue, [], gp, w, MATNAME, MATCOND, opts);
fprintf('\nSkalierung: ||K(E=1000) - 1000 K(E=1)|| / ||.|| = %.2e | ||F(E) - E F|| = %.2e\n', ...
    norm(K2 - E_MOD*K1,'fro')/norm(K2,'fro'), norm(F2 - E_MOD*F1)/norm(F2));
fprintf('            ||K(sX,sU) - s K(X,U)|| / ||.||   = %.2e | ||F(sX,sU) - s^2 F|| = %.2e\n', ...
    norm(K3 - s*K1,'fro')/norm(K3,'fro'), norm(F3 - s^2*F1)/norm(F3));
