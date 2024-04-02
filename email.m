function p = email(p)
    p.name = 'polbook';
    edgefile = 'dataset/email-univ.edges';
    E = load(edgefile);
    p.dim = size(E, 1);
    p.numVar = max(max(E));
    p.bounds = [zeros(1, p.numVar); ones(1, p.numVar)];
    p.adj = Adjreverse(E, p.numVar);
    p.sparsem = sparse(p.adj);
    p.degree = sum(p.adj);
    p.func = @evaluate;

        function f = evaluate(pop, popsize)
            % SolutionEvaluation
            f = zeros(popsize, 2);
            for i = 1:popsize
                v = Objfun(p.adj, p.degree, Decode(pop(i,:)));
                f(i, :) = v;
            end
        end
end