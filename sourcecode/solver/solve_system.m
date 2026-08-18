function x = solve_system(model, A, b)
%SOLVE_SYSTEM Solves a linear equation system.
% ------------------------------------------------------------------------
% DESCRIPTION
%   Solves the reduced linear equation system with the selected linear
%   solver.
%
% INPUT
%   model    FE model structure
%   A        Linear system matrix
%   b        Linear system right-hand side
%
% OUTPUT
%   x        Linear system solution vector
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

solverType = lower(model.solver.linear.type);

switch solverType

    case 'direct'
        % Direkter Loeser
        x = A \ b;

    case 'pcg'
        % Iterativer Loeser: Preconditioned Conjugate Gradient
        %
        % Geeignet fuer symmetrische positiv definite Matrizen.
        tol     = model.solver.linear.tol;
        maxIter = model.solver.linear.maxIter;

        [x,flag,relres,iter] = pcg(A, b, tol, maxIter);

        if flag ~= 0
            warning(['PCG did not fully converge. ', ...
                     'flag = %i, relres = %.4e, iter = %i'], ...
                     flag, relres, iter);
        end

    otherwise
        error('Unknown linear solver type: %s', solverType);

end

end
