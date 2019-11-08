function [CORE] = fun_CORE_POINT_DETECTION2(IMF, IM0, CURVATURE)
% Kannan Karthik/ Aug 21, 2018
% CORE POINT DETECTION based on GRADIENT ANGLE DIVERSITY
[h,w] = size(IMF);
NB = 3;% BLOCK SIZE
Mrows = ceil(h/NB)*NB;
Mcols = ceil(w/NB)*NB;


H = ones(3,3)/9;
%MAGF = imfilter(MAG,H,'same');
A = CURVATURE;
D = blockproc(A, [NB, NB], @fun_PATCH_STAT);
[a,b] = size(D)
DMAX = 0;
for i = 1:a,
    for j = 1:b,
        TERM = D(i,j);
        if TERM > DMAX,
            DMAX = TERM;
            LOC_MAX = [i,j];
        end
    end
end

X0 = LOC_MAX(1);
Y0 = LOC_MAX(2);
C_LOC = [(X0-1)*NB+NB/2,(Y0-1)*NB+NB/2];
CORE = round(C_LOC)
[h,w]
figure;
imshow(uint8(IM0)); hold;
plot(CORE(2), CORE(1),'r*');

% figure;
% imshow(uint8(255*MAG/MAXG)); hold;
% plot(CORE(1), CORE(2),'g*');


end

