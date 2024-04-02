function p = testnetwork(pname)
    p = struct('name', [], 'dim', [], 'bounds', [], 'numVar', [], 'adj', [], 'sparsem', [], 'degree', [], 'func', []);
    switch lower(pname)
        case 'email'
            p = email(p);
        otherwise
            error('Network %s does not exist', pname);
    end
end