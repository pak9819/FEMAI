function [U, result] = newton(model, U, loadscale)
%NEWTON Solves one nonlinear load step with Newton iteration.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Solves one nonlinear load step with Newton iteration.
%
% INPUT
%   model      FE model structure
%   U          Global displacement vector
%   loadscale  Load factor for the current load step
%
% OUTPUT
%   U          Global displacement vector
%   result     Results structure for the current load step
%
% ------------------------------------------------------------------------
% LAST MODIFIED
%   2026-05-06
%
% COPYRIGHT AND LICENSE
%   Copyright (c) 2026 Daniel Materna
%   Section of Mathematics and Computer Simulation
%   OWL University of Applied Sciences and Arts
%
%   Licensed under the MIT License. See LICENSE file in the project root.
% ------------------------------------------------------------------------

dofsfree = model.dofs.free;
dofsfix  = model.dofs.fixed;
valuesfix = model.dofs.fixedValues;

maxit = model.solver.maxIter;
tolR  = model.solver.tolR;
tolU  = model.solver.tolU;

converged = false;

% ------------------------------------------------------------------------
% Dirichlet-Randbedingungen setzen
% Vorgegebene Verschiebungen wachsen mit dem Lastfaktor (Standard)
Ub = loadscale * valuesfix;
U(dofsfix) = Ub;

% oder vorgegebene Verschiebungen gehen voll ein (eigentlich nicht
% konsistent)
%U(dofsfix) = valuesfix;

% ------------------------------------------------------------------------
if model.solver.verbose
    fprintf('\n-------------------------------------------\n');
    fprintf('start Newton\n');
    fprintf('loadscale: %g\n', loadscale);
    fprintf('max iter:  %i\n', maxit);
    fprintf('tol R:     %.4e\n', tolR);
    fprintf('tol U:     %.4e\n', tolU);
    fprintf('-------------------------------------------\n');
    fprintf(' iter     ||Rred||        ||dUred||\n');
end

% ------------------------------------------------------------------------
for it = 0:maxit

    % Assembly am aktuellen Zustand
    [K, Fext0, Fint] = assemble(model, U);

    % Außere Lasten skalieren
    Fext = loadscale * Fext0;
    
    % Residuum
    R = Fint - Fext;

    % Nur freie DOFs fuer Konvergenz pruefen
    Ra = R(dofsfree);
    normRred = norm(Ra,2);

    % Falls Konvergenz vor der Korrekturberechnung erreicht ist,
    % gibt es in dieser Iteration kein neues Inkrement.
    normdUa = 0.0;

    % Konvergenztest
    if it > 0 && normRred <= tolR
        converged = true;
        if model.solver.verbose
            fprintf('%4i   %12.4E   %12.4E\n', it, normRred, normdUa);
        end
        break
    end

    % Maximale Iterationszahl erreicht
    if it == maxit
        break
    end

    % Bestimme dU:
    % Newton-Korrektur auf den freien Freiheitsgraden
    Kaa = K(dofsfree,dofsfree);
    dUa = solve_system(model, Kaa, -Ra);
    
    normdUa = norm(dUa,2);

    % Update nur auf den freien Freiheitsgraden.
    % Die Dirichlet-DOFs bleiben unveraendert auf Gamma_D.
    U(dofsfree) = U(dofsfree) + dUa;
    
    if model.solver.verbose
        fprintf('%4i   %12.4E   %12.4E\n', it, normRred, normdUa);
    end

end

% ------------------------------------------------------------------------
% Keine erneute Assembly noetig, da Konvergenz direkt nach assemble
% geprueft wird. K, Fext, Fint und R gehoeren zum letzten Zustand.

result = init_results(model);
result = result(1);

result.U = U;
result.loadscale = loadscale;

result.K    = K;
result.Fext = Fext;
result.Fint = Fint;

% ------------------------------------------------------------------------
% Lagerkraefte
Freact = compute_reaction_forces(model, Fint, Fext);

result.R = R;
result.Freact = Freact;

% ------------------------------------------------------------------------
result.nIter = it;
result.converged = converged;
result.normR = normRred;
result.normdU = normdUa;

% ------------------------------------------------------------------------
if model.solver.verbose
    fprintf('-------------------------------------------\n');
    if converged
        fprintf('Newton converged after %i iterations.\n', it);
    else
        fprintf('Newton did not converge after %i iterations.\n', maxit);
    end
    fprintf('-------------------------------------------\n\n');
end

if ~converged
    warning('Newton did not converge.');
end


% ------------------------------------------------------------------------
end
