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

TR = 15;
sigma1 = 3;
LL = -3*sigma1;
UL = 3*sigma1;
tg = LL:1:UL;
Lg = length(tg);
for i = 1:Lg,
    x = tg(i);
    G(i) = exp(-x^2/(2*sigma1^2))*cos(2*pi*x/TR);
end
EG = sqrt(sum(G.^2));
G_SINE = G/EG;
N = 5;
G1 = repmat(G_SINE,N,1);

clear G

%% GABOR SMOOTHER

TR = 1;
sigma2 = 5; % sigma2 = 5 (originally for ver3)
% Will sigma2 = 12 work for ver3 as well??
% Anticipate bridges?? YES, see thinned plot.

LL = -3*sigma2;
UL = 3*sigma2;
tg = LL:1:UL;
Lg = length(tg);
for i = 1:Lg,
    x = tg(i);
    G(i) = exp(-x^2/(2*sigma2^2));
end
EG = sqrt(sum(G.^2));
G_SMOOTH = G/EG;
N = 5;
G2 = repmat(G_SMOOTH,N,1);
GSM = G1'*G2;%conv2([1 0; 0 0],G2,'same');
GSM_TH = imrotate(GSM,-45,'bilinear');

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
IM = imread('D:\Work\Acad\sem 7\BTP\data\testGreenBit\017_21_0\Live\pres_A_01\FLAT_THUMB_LEFT.bmp');
%IM = rgb2gray(IM);
[a0,b0] = size(IM);
IM0 = 255-double(IM);
IMF = imfilter(IM0, GSM);
TH = 0.75;
BIN = IM0/255 > TH;
BIN_THIN = bwmorph(BIN,'thin');
figure;
stem(G_SMOOTH);
figure;
subplot(4,1,1); imshow(uint8(IM0));
subplot(4,1,2); imshow(uint8(IMF));
subplot(4,1,3); imshow(uint8(255*BIN));
subplot(4,1,4); imshow(uint8(255*BIN_THIN));

[H,T,R] = hough(BIN,'RhoResolution',3,'Theta',-89:1:89);
HMAX = max(H(:));
H0 = H/HMAX;
figure; mesh(T,R,H0);
figure;
P  = houghpeaks(H,12,'threshold',ceil(0.3*max(H(:))));
lines = houghlines(BIN,T,R,P,'FillGap',5,'MinLength',15);
% figure; imshow(uint8((BIN)*255));
 figure, imshow(uint8(255*BIN)); hold on;
% 
%% Ref: ASSISTANCE from MATLAB HELP ROUTINE
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
    
    % plot beginnings and ends of lines
    plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
    plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
    
    % determine the endpoints of the longest line segment
    SEG_LEN(k) = norm(lines(k).point1-lines(k).point2);
    RHO_EST(k) = lines(k).rho; 
    THETA_EST(k) = abs(lines(k).theta);
%     len = norm(lines(k).point1 - lines(k).point2);
%     if ( len > max_len)
%         max_len = len;
%         xy_long = xy;
%     end
end

% RIDGE STATISTIC COMPUTATION including the estimated break size in terms of pixels
MT = mean(THETA_EST);
TOL = 0.05*MT; % 5\% is sufficiently small to provide good resolution.
THQ1 = round(THETA_EST/TOL)*TOL;
THQ = union(THQ1,[]);
MR = mean(RHO_EST);
TOL2 = 0.05*MR;
RQ1 = round(RHO_EST/TOL2)*TOL2;
RQ = union(RQ1,[]);

SCOMP = THQ1.*RQ1; % COMPOSITE STATISTIC
SCOMP_REF = union(SCOMP,[]);
LCOMP = length(SCOMP_REF);
% CLASS ASSIGNMENT
for i = 1:length(THQ1);
    for j = 1:LCOMP,
        DIFF(j) = abs(SCOMP(i)-SCOMP_REF(j));
    end
    [VAL, IND_MIN] = min(DIFF);
    CLASS(i) = IND_MIN;
end

for i = 1:LCOMP,
    B = CLASS == i;
    IND = find(B==1);
    SUM = 0;
    LIND = length(IND);
    c = 1;
    for p = 1:LIND,
        for q = p+1:LIND,
            D(c) =  norm(lines(IND(p)).point1-lines(IND(q)).point2);
            c = c+1;
        end
        SUM = SUM + SEG_LEN(IND(p));
    end
    DMAX(i) = max(D);
    RBREAK(i) = (1-(SUM/DMAX(i)))*DMAX(i)/(LIND-1) ;
end
disp('RIDGE GAP estimated in terms of pixels is...');
MU_BREAK = median(RBREAK) % For the above parameter settings works
% out to around 12 pixels



% ACTUAL FILTERING

figure;
LTH = length(THQ);
for i = 1:LTH,
    G{i} = imrotate(GSM,-THQ(i)+90,'bilinear');
    B = G{i};
    B0 = B(:);
    E = sqrt(sum(B0.^2));
    G{i} = G{i}/E;
    IMFILT_BANK{i} = imfilter(double(IM0),G{i});
end
figure;
t = ceil(sqrt(2+LTH));
subplot(t,t,1); imshow(uint8(IM0));
for j = 1:LTH,
subplot(t,t,1+j); imshow(uint8(IMFILT_BANK{j}));
title(strcat('GABOR \phi =',num2str(THQ(j))));
end
[a1,b1] = size(IMFILT_BANK{1});
for i = 1:a1,
    for j = 1:b1,
        for c = 1:LTH,
            B = IMFILT_BANK{c};
            VAL(c) = B(i,j);
        end
        MAX_FILT(i,j) = max(VAL);
        MEAN_FILT(i,j) = mean(VAL);
        GMEAN(i,j) = sqrt(prod(VAL));
        clear VAL
    end
end

Q = MAX_FILT;
QMAX = max(Q(:));

subplot(t,t,2+LTH); imshow(uint8(255*Q/QMAX)); title('Final fused - MAX');

figure; imshow(uint8(255*Q/QMAX)); title('Final fused - MAX');

BINF = Q > mean(Q(:));


%% FINAL PLOT with THINNING
% figure; 
% %subplot(1,2,1); imshow(uint8(Q)); title('Filtered/Quant');
% subplot(3,1,1); imshow(uint8(IM0));
% subplot(3,1,2); imshow(uint8(255*BINF)); title('BINARIZED');
% subplot(3,1,3); imshow(uint8(255*BIN0)); title('THINNED');

figure; 
%subplot(1,2,1); imshow(uint8(Q)); title('Filtered/Quant');
subplot(2,2,1); imshow(uint8(IM0)); title('ORIGINAL');
subplot(2,2,2); imshow(uint8(255*BINF)); title('BINARIZED');
STREL = strel('disk',2);
BINF_P = imerode(BINF,STREL);
subplot(2,2,3); imshow(uint8(255*BINF_P)); title('ERODED');
BIN0 = bwmorph(BINF_P,'skel',Inf);
subplot(2,2,4); imshow(uint8(255*BIN0)); title('THINNED');
