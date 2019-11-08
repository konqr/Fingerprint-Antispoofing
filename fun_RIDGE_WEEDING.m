function [LOC_RIDGE_TRIM, LOC_BIF_TRIM] = fun_RIDGE_WEEDING(BIN0, LOC_RIDGE, LOC_BIF)
% Kannan Karthik
% Aug 31, 2018
L = bwlabeln(BIN0);
%figure;
MAX = max(L(:));
%imshow(uint8(255*L/MAX));
LOC_RIDGE_TRIM = LOC_RIDGE;
LOC_BIF_TRIM = LOC_BIF;
for i = 1:length(LOC_RIDGE),
    A = LOC_RIDGE{i};
    LRIDGE(i) = L(A(1), A(2));
end
for i = 1:length(LOC_BIF),
    A = LOC_BIF{i};
    LBIF(i) = L(A(1), A(2));
end

SET_RIDGE_WEED = [];
SET_BIF_WEED = [];

for i = 1:length(LOC_RIDGE),
    for j = 1:length(LOC_BIF),
        A = LOC_RIDGE{i};
        B = LOC_BIF{j};
        LABA = L(A(1), A(2));
        LABB = L(B(1), B(2));
        DAB = norm(A-B);
        T = 5;
        if DAB < T && LABA == LABB,
            LOC_RIDGE_TRIM{i} = [];
        end
    end
end

SET_RIDGE_WEED2 = [];

for i = 1:length(LOC_RIDGE),
    for j = i+1:length(LOC_RIDGE),
        A = LOC_RIDGE{i};
        B = LOC_RIDGE{j};
        LABA = L(A(1), A(2));
        LABB = L(B(1), B(2));
        DAB = norm(A-B);
        T = 5;
        if DAB < T && LABA == LABB,
            LOC_RIDGE_TRIM{j} = [];            
        end
    end
end

% SET_RIDGE_WEED_FINAL = union(SET_RIDGE_WEED, SET_RIDGE_WEED2);
% c = 1;
% for i = 1:length(LOC_RIDGE),
%     BT = SET_RIDGE_WEED_FINAL == i;
%     TEST = sum(BT);
%     if TEST == 0,
%     LOC_RIDGE_TRIM{c} = LOC_RIDGE{i};
%     c = c+1;
%     end
% end
% 
% 
% c = 1;
% for i = 1:length(LOC_BIF),
%     BT = SET_BIF_WEED == i;
%     TEST = sum(BT);
%     if TEST == 0,
%     LOC_BIF_TRIM{c} = LOC_BIF{i};
%     c = c+1;
%     end
% end

LOC_BIF_TRIM = LOC_BIF_TRIM(~cellfun('isempty',LOC_BIF_TRIM));
LOC_RIDGE_TRIM = LOC_RIDGE_TRIM(~cellfun('isempty',LOC_RIDGE_TRIM));

        
end

