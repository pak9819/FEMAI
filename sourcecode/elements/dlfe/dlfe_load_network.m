function NET = dlfe_load_network(netFile, n, dim)
%DLFE_LOAD_NETWORK Laedt ein Gram-Energienetz und prueft die Metadaten HART.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Generisches Laden fuer DLFE-Energienetze mit Metrik-Eingang
%   (state_form = 'gram_metric_centered_triu' oder
%   'gram_modal_F0_hourglass_triu'), exportiert von
%   training/common/dlfe_gram.py. Alles, was stumm falsche Physik ergaebe,
%   ist ein error (kein warning).
%
%   Zusaetzlich werden die Indexmengen der Gram-Kette vorberechnet
%   (triu-Maske, Jacobi-Indizes), damit der Hot-Path nur noch rechnet.
%
% INPUT
%   netFile  Pfad der .mat-Datei
%   n, dim   erwartete Knotenzahl und Dimension des Elements
%
% OUTPUT
%   NET      Struktur: L, W, b, c_mean, c_std, D_scale, Metadaten, Indizes
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

if ~exist(netFile, 'file')
    [~, name, ext] = fileparts(netFile);
    error('dlfe_load_network:NetworkMissing', ...
        '%s nicht gefunden -- zuerst das Trainingsskript ausfuehren.', [name ext]);
end
S = load(netFile);

req = {'model_form', 'activation', 'state_form', 'canonicalization', ...
       'num_linear_layers', 'input_norm_c_mean', 'input_norm_c_std', ...
       'input_norm_D_scale', 'material', 'nu_train', 'condition'};
for i = 1:numel(req)
    if ~isfield(S, req{i})
        error('dlfe_load_network:MetaMissing', ...
            'Pflicht-Metadatum "%s" fehlt im Netz %s -- Netz neu trainieren.', ...
            req{i}, netFile);
    end
end
assert_meta(S, 'model_form', 'total_energy_subtract_f0_gradf0');
assert_meta(S, 'activation', 'GELU_erf');
sf = strtrim(char(S.state_form));
switch sf
    case 'gram_metric_centered_triu',    modal = false;
    case 'gram_modal_F0_hourglass_triu', modal = true;
    otherwise
        error('dlfe_load_network:MetaMismatch', ...
            'Unbekannte Zustandsform "%s" -- Netz neu trainieren.', sf);
end
if dim == 2
    assert_meta(S, 'canonicalization', 'edge_n1n2_to_pos_x_after_centroid_Lc');
else
    assert_meta(S, 'canonicalization', 'edge_n1n2_x_node4_xy_after_centroid_Lc');
end

ndof = n * dim;
if modal, k = n - 1; else, k = n; end
m    = k * (k + 1) / 2;

NET.L = round(double(S.num_linear_layers));
NET.W = cell(NET.L, 1);
NET.b = cell(NET.L, 1);
for l = 1:NET.L
    NET.W{l} = double(S.(sprintf('W%d', l)));
    bl = double(S.(sprintf('b%d', l)));
    NET.b{l} = bl(:);
end
if size(NET.W{1}, 2) ~= ndof + m
    error('dlfe_load_network:InputSize', ...
        'Erste Gewichtslage erwartet %d Eingaenge (%d Geometrie + %d Metrik), hat %d.', ...
        ndof + m, ndof, m, size(NET.W{1}, 2));
end
if size(NET.W{NET.L}, 1) ~= 1
    error('dlfe_load_network:OutputSize', 'Ausgabeschicht muss skalar sein.');
end

NET.c_mean  = double(S.input_norm_c_mean(:));
NET.c_std   = double(S.input_norm_c_std(:));
NET.D_scale = double(S.input_norm_D_scale(:));

NET.nu_train  = double(S.nu_train);
NET.material  = strtrim(char(S.material));
NET.condition = strtrim(char(S.condition));
NET.state_form = sf;
NET.modal = modal;

NET.E_max = 0.2;
if isfield(S, 'state_E_max'), NET.E_max = double(S.state_E_max); end
NET.state_measure = 'E_green';
NET.H_max = Inf;  NET.J_min = 0;  NET.J_max = Inf;
if isfield(S, 'state_measure')
    NET.state_measure = strtrim(char(S.state_measure));
end
if strcmpi(NET.state_measure, 'hencky_J')
    NET.E_max = Inf;
    NET.H_max = double(S.state_H_max);
    NET.J_min = double(S.state_J_min);
    NET.J_max = double(S.state_J_max);
end

% --- Indexmengen der Gram-Kette -----------------------------------------
%   D(q) = M(i_q, j_q), q laeuft column-major ueber triu(k x k) (== Python).
NET.n = n;  NET.dim = dim;  NET.ndof = ndof;  NET.k = k;  NET.m = m;
mask = triu(true(k));
[ik, jk] = find(mask);                 % column-major, identisch Python
NET.triuMask = mask;
NET.Pn = eye(n) - ones(n) / n;
NET.Phi = dlfe_mode_matrix(n, dim);    % isoparametrische Moden (modal)

%   Jacobi dD/dZ (m x k*dim): Zeile q, Block j_q = Z(i_q,:), Block i_q
%   += Z(j_q,:). Lineare Indizes vorberechnen.
rows = repmat((1:m).', 1, dim);
dd   = repmat(1:dim, m, 1);
colJ = (jk - 1) * dim + dd;
colI = (ik - 1) * dim + dd;
NET.jyIdxJ = sub2ind([m, k*dim], rows(:), colJ(:));
NET.jyIdxI = sub2ind([m, k*dim], rows(:), colI(:));
NET.jyValJ = sub2ind([k, dim], repmat(ik, dim, 1), dd(:));   % Z(i_q, d)
NET.jyValI = sub2ind([k, dim], repmat(jk, dim, 1), dd(:));   % Z(j_q, d)

%   Lineare Indizes fuer kron(G, I_dim) (n x n -> ndof x ndof), Reihenfolge
%   passend zu repmat(G(:), dim, 1): erst alle (a,b) fuer d = 1, dann d = 2 ...
[aa, bb] = ndgrid(1:n, 1:n);
kI = [];
for d = 1:dim
    kI = [kI; sub2ind([ndof, ndof], (aa(:)-1)*dim + d, (bb(:)-1)*dim + d)]; %#ok<AGROW>
end
NET.kronIdx = kI;
end


function assert_meta(S, field, expected)
val = strtrim(char(S.(field)));
if ~strcmpi(val, expected)
    error('dlfe_load_network:MetaMismatch', ...
        ['Netz-Metadatum "%s" ist "%s", erwartet "%s".\n' ...
         'Das Element wuerde damit stumm falsch rechnen -- Netz neu trainieren.'], ...
        field, val, expected);
end
end
