function [ LOC_BIF, LOC_RID]  = fun_BIFURCATION_RID_END_DETECT(BIN_THINNED, MASK)
% Kannan Karthik
% Bifurcation detection based on CROSSING NUMBERS '3'
B = BIN_THINNED;
[h,w] = size(B);
cb = 1;
cr = 1;
LOC_BIF ={};
LOC_RID = {};
for i = 2:h-1,
    for j = 2:w-1,
        if B(i,j) == 1 && MASK(i,j) == 1,
        c = 1;
        LOCS{1} = [i-1,j-1];
        LOCS{2} = [i-1,j];
        LOCS{3} = [i-1,j+1];
        LOCS{4} = [i,j+1];
        LOCS{5} = [i+1,j+1];
        LOCS{6} = [i+1,j];
        LOCS{7} = [i+1,j-1];
        LOCS{8} = [i,j-1];
        L = length(LOCS);
        for k = 1:L,
            A = LOCS{k};
            Q(k) = B(A(1),A(2));
        end
        RQ = circshift(Q,1,2);
        CN = sum(abs(Q-RQ))/2;
        if CN == 3 && sum(Q)==3,
           LOC_BIF{cb} = [i,j];
           cb = cb+1;
        end
        if CN == 1,% && sum(Q)==1,
            LOC_RID{cr} = [i,j];
            cr = cr+1;
        end
        
        end
    end
end
end

