function [LOC_BIF_TRIM] = fun_BIFUR_WEEDING(BIN0, LOC_BIF)
% Kannan Karthik
% Aug 31, 2018
L = bwlabeln(BIN0);
figure;
MAX = max(L(:));
imshow(uint8(255*L/MAX));
for i = 1:length(LOC_BIF),
    A = LOC_BIF{i};
    LAB(i) = L(A(1), A(2));
end
SET = [];
for i = 1:length(LOC_BIF),
    for j = i+1:length(LOC_BIF),
        A = LOC_BIF{i};
        B = LOC_BIF{j};
        LABA = L(A(1), A(2));
        LABB = L(B(1), B(2));
        DAB = norm(A-B);
        T = 5;
        if DAB < T && LABA == LABB,
            SET = union(SET,[i,j]);
        end
    end
end
c = 1;
for i = 1:length(LOC_BIF),
    BT = SET == i;
    TEST = sum(BT);
    if TEST == 0,
    LOC_BIF_TRIM{c} = LOC_BIF{i};
    c = c+1;
    end
end

    
        
end

