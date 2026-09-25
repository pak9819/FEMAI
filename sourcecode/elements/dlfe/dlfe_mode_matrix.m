function Phi = dlfe_mode_matrix(n, dim)
%DLFE_MODE_MATRIX Isoparametrische Moden [linear | Hourglass] / 2^dim.
% ------------------------------------------------------------------------
% DESCRIPTION
%   quad4  (n = 4, dim = 2): Spalten r, s | rs
%   brick8 (n = 8, dim = 3): Spalten r, s, t | rs, st, rt, rst
%   jeweils ausgewertet an den Knoten (natuerliche Koordinaten +-1) und
%   durch 2^dim geteilt -- identisch training/common/dlfe_gram.py
%   (mode_matrix). Alle Spalten summieren zu null.
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-09-23
% ------------------------------------------------------------------------

if n == 4 && dim == 2
    r = [-1; 1; 1; -1];  s = [-1; -1; 1; 1];
    Phi = [r, s, r.*s] / 4;
elseif n == 8 && dim == 3
    r = [-1; 1; 1; -1; -1; 1; 1; -1];
    s = [-1; -1; 1; 1; -1; -1; 1; 1];
    t = [-1; -1; -1; -1; 1; 1; 1; 1];
    Phi = [r, s, t, r.*s, s.*t, r.*t, r.*s.*t] / 8;
else
    Phi = [];
end
end
