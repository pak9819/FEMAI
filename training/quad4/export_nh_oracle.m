%EXPORT_NH_ORACLE Oracle fuer Gate a' (Neo-Hooke): MATLAB-Element auf Python-Zustaenden.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Wertet das ANALYTISCHE Element element_quad4_nl mit 'NeoHooke' auf den
%   Zustaenden aus nh_oracle_states.mat aus und schreibt Finte und Ke nach
%   nh_oracle.mat. quad4_nh_ref.py (Gate a') vergleicht damit seine
%   Python-Referenz -- die MATLAB-Materialroutine ist die Wahrheit, gegen die
%   spaeter auch die Benchmarks rechnen.
%
%   Ablauf:
%     python quad4_nh_ref.py --oracle-states     (schreibt die Zustaende)
%     export_nh_oracle                           (dieses Skript, MATLAB)
%     python quad4_nh_ref.py                     (Gate a' laeuft mit)
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-17
% ------------------------------------------------------------------------

thisDir = fileparts(mfilename('fullpath'));
run(fullfile(thisDir, '..', '..', 'startup.m'));

S = load(fullfile(thisDir, 'nh_oracle_states.mat'));
n = size(S.Ue, 2);

gp = [-1 -1; 1 -1; 1 1; -1 1] / sqrt(3);
w  = ones(4, 1);

coords = S.coords;
Ue     = S.Ue;
mat    = S.mat;
Finte  = zeros(8, n);
Ke     = zeros(8, 8, n);
for i = 1:n
    [Ke(:,:,i), ~, ~, Finte(:,i)] = element_quad4_nl(coords(:,:,i), mat(:,i).', ...
        [0; 0], 0, Ue(:,i), [], gp, w, 'NeoHooke', 'planeStrain', struct());
end

save(fullfile(thisDir, 'nh_oracle.mat'), 'coords', 'Ue', 'mat', 'Finte', 'Ke', '-v7');
fprintf('nh_oracle.mat geschrieben (%d Faelle).\n', n);
