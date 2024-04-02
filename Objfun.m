function values = Objfun(adj, degree, labels)
    sum_v_in=0;
    sum_v_out=0;
    comm_num = 0;
    t=max(labels);
    for i=1:t
        index=find(labels==i);
        if ~isempty(index)
            comm_num = comm_num + 1;
            m=adj(index,index);
            edges_in=(sum(sum(m))-sum(diag(m)))/2+sum(diag(m));
            edges_out=sum(degree(index))-edges_in*2;
            n=length(index);
            sum_v_in=edges_in/n+sum_v_in;
            sum_v_out=edges_out/n+sum_v_out;
        end
    end
    values(1)=2*(length(degree)-comm_num)-sum_v_in*2;
    values(2)=sum_v_out;
end