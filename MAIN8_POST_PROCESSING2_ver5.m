% AUthor: Dr. Kannan Karthik/towards EE660
% Aug 7, 2018; Modified: Aug 12, 2018
% ONE DIMENSIONAL NOISE SIMULATION ALONG and ORTHOGONAL TO THE RIDGES
% FILTERING USING THE GABOR WAVELET
% VERSION-2 + FILTERING and recovery
% VERSION-4 + GABOR SINE works for pulse restoration
% Claim: Note ridge endings and bifurcation points have nothing to do with the
% ridge spacings. Hence by ironing out the slices, very little information
% pertaining to the biometric is lost.
% VERSION-5 actual implementation on images: Observations and
% interpretations
% RIDGE BREAK STATISTICS
% GABOR SMOOTHING on TOY IMAGES
% MORPHOLOGICAL OPERATIONS included
% ANGULAR ESTIMATE USED through HOUGH TRANSFORMS

clear all;
close all;

%% GABOR SINE

TR = 4;
sigma1 = 2;
G1 = fun_GABOR_SINE_BASE(TR, sigma1);

TR2 = 4; sigma3 = 2;
G1_LAYER2 = fun_GABOR_SINE_BASE(TR2, sigma3);

TR2 = 4; sigma4 = 2;
G1_LAYER3 = fun_GABOR_SINE_BASE(TR2, sigma4);

%% GABOR SMOOTHER
TR = 1;
sigma2 = 3; % sigma2 = 5 (originally for ver3)
G2 = fun_GABOR_SMOOTHER_BASE(sigma2)

GSM = G1'*G2;%conv2([1 0; 0 0],G2,'same');
GSM_TH = imrotate(GSM,-45,'bilinear');

GSM2 = G1_LAYER2'*G2;
GSM3 = G1_LAYER3'*G2;
figure;
subplot(2,1,1);
mesh(GSM); title('Original Gaussian');
subplot(2,1,2);
mesh(GSM_TH); title('Rotated by 45 \degree');

clear G

% IMAGE PROCESSING
% RIDGE BREAK FILLING
% BREAK STATISTICS
%IM = imread('TEST_IMG1.jpg');
IM = imread('102_3.jpg');
figure; imshow(IM); title('ORIGINAL NOISY FINGERPRINT');
%IM = rgb2gray(IM);
[a0,b0] = size(IM);
IM0 = 255-double(IM);
% IMF = imfilter(IM0, GSM);
% TH = 0.75;
% BIN = IM0/255 > TH;
% BIN_THIN = bwmorph(BIN,'thin');
% figure;
% stem(G_SMOOTH);
% figure;
% subplot(4,1,1); imshow(uint8(IM0));
% subplot(4,1,2); imshow(uint8(IMF));
% subplot(4,1,3); imshow(uint8(255*BIN));
% subplot(4,1,4); imshow(uint8(255*BIN_THIN));

% [H,T,R] = hough(BIN,'RhoResolution',3,'Theta',-89:1:89);
% HMAX = max(H(:));
% H0 = H/HMAX;
% figure; mesh(T,R,H0);
% figure;
% P  = houghpeaks(H,12,'threshold',ceil(0.3*max(H(:))));
% lines = houghlines(BIN,T,R,P,'FillGap',5,'MinLength',15);
% % figure; imshow(uint8((BIN)*255));
%  figure, imshow(uint8(BIN)); hold on;
% % 
% %% Ref: ASSISTANCE from MATLAB HELP ROUTINE
% max_len = 0;
% for k = 1:length(lines)
%     xy = [lines(k).point1; lines(k).point2];
%     plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
%     
%     % plot beginnings and ends of lines
%     plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
%     plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
%     
%     % determine the endpoints of the longest line segment
%     SEG_LEN(k) = norm(lines(k).point1-lines(k).point2);
%     RHO_EST(k) = lines(k).rho; 
%     THETA_EST(k) = abs(lines(k).theta);
% %     len = norm(lines(k).point1 - lines(k).point2);
% %     if ( len > max_len)
% %         max_len = len;
% %         xy_long = xy;
% %     end
% end
% 
% % RIDGE STATISTIC COMPUTATION including the estimated break size in terms of pixels
% MT = mean(THETA_EST);
% TOL = 0.05*MT; % 5\% is sufficiently small to provide good resolution.
% THQ1 = round(THETA_EST/TOL)*TOL;
% THQ = union(THQ1,[]);
% MR = mean(RHO_EST);
% TOL2 = 0.05*MR;
% RQ1 = round(RHO_EST/TOL2)*TOL2;
% RQ = union(RQ1,[]);
% 
% SCOMP = THQ1.*RQ1; % COMPOSITE STATISTIC
% SCOMP_REF = union(SCOMP,[]);
% LCOMP = length(SCOMP_REF);
% % CLASS ASSIGNMENT
% for i = 1:length(THQ1);
%     for j = 1:LCOMP,
%         DIFF(j) = abs(SCOMP(i)-SCOMP_REF(j));
%     end
%     [VAL, IND_MIN] = min(DIFF);
%     CLASS(i) = IND_MIN;
% end
% 
% for i = 1:LCOMP,
%     B = CLASS == i;
%     IND = find(B==1);
%     SUM = 0;
%     LIND = length(IND);
%     c = 1;
%     for p = 1:LIND,
%         for q = p+1:LIND,
%             D(c) =  norm(lines(IND(p)).point1-lines(IND(q)).point2);
%             c = c+1;
%         end
%         SUM = SUM + SEG_LEN(IND(p));
%     end
%     DMAX(i) = max(D);
%     RBREAK(i) = (1-(SUM/DMAX(i)))*DMAX(i)/(LIND-1) ;
% end
% disp('RIDGE GAP estimated in terms of pixels is...');
% MU_BREAK = median(RBREAK) % For the above parameter settings works
% % out to around 12 pixels
% 

[IMF1] = fun_GABOR_FILTER_BANK(IM0, GSM);
[IMF2] = fun_GABOR_FILTER_BANK(IMF1, GSM2);
[IMF3] = fun_GABOR_FILTER_BANK(IMF2, GSM3);

figure;
%subplot(2,2,1); imshow(IM); title('RAW');
figure; imshow(uint8(255*IMF1/max(IMF1(:)))); title('FILTERED-1');
figure; imshow(uint8(255*IMF2/max(IMF2(:)))); title('FILTERED TWICE');
figure; imshow(uint8(255*IMF3/max(IMF3(:)))); title('FILTERED THRICE');



Q = IMF3;

QMAX = max(Q(:));
SD = sqrt(var(Q(:)));

figure; imshow(uint8(255*Q/QMAX)); title('Final fused - MAX-MAX');


alpha = 1;
delta = mean(Q(:));
IND = Q > delta;
[a,b] = size(IND);
c = 1;
for i = 1:a,
    for j = 1:b,
        if IND(i,j) == 1,
           LIST(c) = Q(i,j);
           c = c+1;
        end
    end
end

BINF = Q > delta;


%% FINAL PLOT with THINNING
% figure; 
% %subplot(1,2,1); imshow(uint8(Q)); title('Filtered/Quant');
% subplot(3,1,1); imshow(uint8(IM0));
% subplot(3,1,2); imshow(uint8(255*BINF)); title('BINARIZED');
% subplot(3,1,3); imshow(uint8(255*BIN0)); title('THINNED');

figure; 
%subplot(1,2,1); imshow(uint8(Q)); title('Filtered/Quant');
subplot(2,3,1); imshow(uint8(IM0)); title('ORIGINAL');
subplot(2,3,2); imshow(uint8(255*Q/QMAX)); title('FILTERED and MAX fused');
subplot(2,3,3); imshow(uint8(255*BINF)); title('BINARIZED');
ST1 = strel('disk',1);
ST2 = strel('disk',3);
%BINF2 = imerode(BINF,ST);
BINF2 = imerode(BINF,ST1);
BINF3 = imdilate(BINF2,ST2);
subplot(2,3,4); imshow(uint8(255*BINF3)); title('OPENING: EROSION+DILATION');
BIN0 = bwmorph(BINF2,'skel',Inf);
subplot(2,3,5); imshow(uint8(255*BIN0)); title('THINNED');
H = [0 1 0; 1 1 1; 0 0 0];
STR = strel('arbitrary',H);
BIN_THINNED = BIN0;
% % MASK GENERATION
Y = double(Q/mean(Q(:)));
[h,w] = size(Y);
VD = zeros(h,w);
for i = 6:h-6,
    for j = 6:w-6,
        c = 1;
        for p= i-5:1:i+5,
            for q = j-5:1:j+5,
                DATA(c) = Y(p,q);
                c = c+1;
            end
        end
        VD(i,j) = var(DATA);
        
    end
end
MU_VAR = mean(VD(:));
VB = VD > 0.5*MU_VAR;
R0 = 3;
STR = strel('disk',R0);
[h,w] = size(BIN_THINNED);

MASK = imerode(VB, STR);
%MASK = ones(h,w);
figure;
imshow(uint8(255*MASK));
[ LOC_BIF, LOC_RID]  = fun_BIFURCATION_RID_END_DETECT(BIN_THINNED, MASK);
figure; imshow(uint8(255*BIN0)); hold on;
for k1 = 1:length(LOC_BIF),
    A = LOC_BIF{k1};
    plot(A(2), A(1),'r.'); hold on;
end
for k1 = 1:length(LOC_RID),
    A = LOC_RID{k1};
    plot(A(2), A(1),'b.'); hold on;
end
title('WITHOUT BRIDGE PAIRING REMOVAL and RIDGE TRIMMING');
% CONNECTED COMPONENTS BASED BIFURCATION POINT TRIMMING

[LOC_RID_TRIM, LOC_BIF_TRIM] = fun_RIDGE_WEEDING(BIN0, LOC_RID, LOC_BIF);
figure; imshow(uint8(255*BIN0)); hold on;
for k1 = 1:length(LOC_RID_TRIM),
    A = LOC_RID_TRIM{k1};
    plot(A(2), A(1),'b.'); hold on;
end
for k1 = 1:length(LOC_BIF_TRIM),
    A = LOC_BIF_TRIM{k1};
    plot(A(2), A(1),'r.'); hold on;
end
title('MINUTIA with RIDGE TRIMMING');


LOC_BIF_TRIM = fun_BIFUR_WEEDING2(BIN0, LOC_BIF_TRIM);
figure; imshow(uint8(255*BIN0)); hold on;
for k1 = 1:length(LOC_BIF_TRIM),
    A = LOC_BIF_TRIM{k1};
    plot(A(2), A(1),'r.'); hold on;
end
for k1 = 1:length(LOC_RID_TRIM),
    A = LOC_RID_TRIM{k1};
    plot(A(2), A(1),'b.'); hold on;
end
title('MINUTIA with BIFURCATION TRIMMING');

figure; imshow(uint8(255*BIN0)); hold on;
for k1 = 1:length(LOC_BIF_TRIM),
    A = LOC_BIF_TRIM{k1};
    plot(A(2), A(1),'r*'); hold on;
end
for k1 = 1:length(LOC_RID_TRIM),
    A = LOC_RID_TRIM{k1};
    plot(A(2), A(1),'g*'); hold on;
end
title('FINAL MINUTIA SET');
