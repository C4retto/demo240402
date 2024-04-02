function main
    pname = 'email';
    embfile = sprintf('dataset/embeddings.emb');
    NetEmb = EmbRead(embfile);

    p = testnetwork(pname);
    popsize = 102;
    param = 0:0.1:1;
    runTime = 20;
    
    for prm = param
        result = zeros(3,runTime);
        for runtime = 1:runTime
            [pop,~] = ncmoeainit(p, NetEmb, popsize, prm);
            modular = zeros(1,popsize);
            for ii = 1:popsize
                modular(1,ii) = modularity(p.adj, Decode(pop(ii,:)));
            end
            result(1,runtime) = min(modular);
            result(2,runtime) = std(modular);
            result(3,runtime) = mean(modular);

        end
        disp(prm);
        disp(result);
    end
end
