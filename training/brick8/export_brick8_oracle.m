%EXPORT_BRICK8_ORACLE MATLAB-Element als Oracle fuer Gate a' (brick8).
% ------------------------------------------------------------------------
% DESCRIPTION
%   Liest brick8_oracle_states.mat (python brick8_nl_ref.py --oracle-states),
%   wertet element_brick8_nl (StVenant, E = 1000, nu = 0.3) aus und
%   schreibt Finte, Ke nach brick8_oracle.mat. Python vergleicht dagegen.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

here = fileparts(mfilename('fullpath'));
run(fullfile(here, '..', '..', 'startup.m'));

S = load(fullfile(here, 'brick8_oracle_states.mat'));
N = size(S.Ue, 2);
g = gauss_library('brick8', 'default');
mat_e = [1000, 0.3, NaN, 0];
Finte = zeros(24, N);
Ke = zeros(24, 24, N);
for i = 1:N
    [K, ~, ~, F] = element_brick8_nl(S.coords(:,:,i), mat_e, zeros(3,1), 0, ...
        S.Ue(:,i), [], g.gp, g.w, 'StVenant', '3D', struct());
    Finte(:,i) = F;
    Ke(:,:,i) = K;
end
coords = S.coords;  Ue = S.Ue; %#ok<NASGU>
save(fullfile(here, 'brick8_oracle.mat'), 'coords', 'Ue', 'Finte', 'Ke');
fprintf('brick8_oracle.mat geschrieben (%d Zustaende).\n', N);
