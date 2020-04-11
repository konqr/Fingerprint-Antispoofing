%Konark Jain
%27th Sept 2019

imagePaths = glob('D:\Work\Acad\BTP\data\t*Orcathus\*\*\*\*\*.png');
done = length(glob('D:\Work\Acad\BTP\data\t*Orcathus\*\*\*\*\*_set.csv'));
% imagePaths = imagePaths(done+1:end);
% imagePaths = ['D:\Work\Acad\sem 7\BTP\data\testGreenBit\017_21_0\Live\pres_A_01\FLAT_THUMB_LEFT.bmp'];
%imagePaths = glob('D:\Work\Acad\BTP\data\testGreenBit\017_21_0\Live\pres_A_01\FLAT_THUMB_LEFT.bmp');

for i =done-1:length(imagePaths),
    imageName = imagePaths{i}
    sigma = 0;
    if (~isempty(strfind(imageName,'Digital')))||(~isempty(strfind(imageName,'Orcathus')))
        sigma = 1;
    end
    %% GABOR SINE

    TR = 4 + sigma;
    sigma1 = 2 + sigma;
    G1 = fun_GABOR_SINE_BASE(TR, sigma1);

    TR2 = 4 + sigma; sigma3 = 2 + sigma;
    G1_LAYER2 = fun_GABOR_SINE_BASE(TR2, sigma3);

    TR2 = 4+sigma; sigma4 = 2 + sigma;
    G1_LAYER3 = fun_GABOR_SINE_BASE(TR2, sigma4);

    %% GABOR SMOOTHER
    TR = 1;
    sigma2 = 3 + sigma; % sigma2 = 5 (originally for ver3)
    G2 = fun_GABOR_SMOOTHER_BASE(sigma2);

    GSM = G1'*G2;%conv2([1 0; 0 0],G2,'same');
    GSM_TH = imrotate(GSM,-45,'bilinear');

    GSM2 = G1_LAYER2'*G2;
    GSM3 = G1_LAYER3'*G2;

    clear G

    IM = imresize(imread(imageName),[380 380]);
    if strfind(imageName,'Digital') > 0
        IM = rgb2gray(IM);
    end
    if strfind(imageName,'Orcathus')>0
        IM0 = double(IM);
    else
        IM0 = double(imcomplement(IM));
    end
    blob = bwareafilt(IM0>0, 1);
    rows = find(any(blob));
    cols= find(any(blob'));
    IM0 = imcrop(IM0, [rows(1), cols(1), rows(end)-rows(1),cols(end)-cols(1)]);
    %figure; imshow(IM0);
    imwrite(255-uint8(IM0),[imageName(1:end-4),'_crop.bmp']);
    [a0,b0] = size(IM0);
    M = mean(mean(IM0));
    VAR = std(double(IM0(:)));
    M0 = 200;
    VAR0 = 200;
    G = IM0;
    for i = 1:a0,
        for j = 1:b0,
            if IM0(i,j) > M,
                G(i,j) = M0 + sqrt(VAR0*((IM0(i,j) - M)^2)/VAR);
            else
                G(i,j) = M0 - sqrt(VAR0*((IM0(i,j) - M)^2)/VAR);
            end
        end
    end
    IM1 = G;
    %figure;imshow(uint8(G));
    [IMF1] = fun_GABOR_FILTER_BANK(IM1, GSM);
    [IMF2] = fun_GABOR_FILTER_BANK(IMF1, GSM2);
    [IMF3] = fun_GABOR_FILTER_BANK(IMF2, GSM3);
    
    Q = IMF3;
    
    QMAX = max(Q(:));
    SD = sqrt(var(Q(:)));
    alpha = 1;
    delta = mean(Q(:)) + SD/10;
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
    grayImage = uint8(255 * BINF);  % BIN is from Joseph's code.
    imwrite(255-uint8(grayImage),[imageName(1:end-4),'_enh.bmp']);
    blackImage = zeros(size(grayImage), 'uint8');
    rgbImage = cat(3, blackImage , grayImage, blackImage); % Only green channel is non-zero.
    %figure; imshow(rgbImage);
%     ST1 = strel('disk',1);
%     ST2 = strel('disk',3);
%     %BINF2 = imerode(BINF,ST);
%     BINF2 = imerode(BINF,ST1);
%     %% figure; imshow(BINF2);
%     BINF3 = imdilate(BINF2,ST2);
%     %% figure; imshow(BINF3);
    BIN0 = bwmorph(BINF,'skel',Inf);
    BIN0 = bwmorph(BIN0, 'bridge');
    grayImage = uint8(255 * BIN0);  % BIN is from Joseph's code.
    blackImage = zeros(size(grayImage), 'uint8');
    rgbImage = cat(3, blackImage , grayImage, blackImage); % Only green channel is non-zero.
    %figure; imshow(rgbImage);
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


    [ LOC_BIF, LOC_RID]  = fun_BIFURCATION_RID_END_DETECT(BIN_THINNED, MASK);
    [LOC_RID_TRIM, LOC_BIF_TRIM] = fun_RIDGE_WEEDING(BIN0, LOC_RID, LOC_BIF);
    LOC_BIF_TRIM = fun_BIFUR_WEEDING2(BIN0, LOC_BIF_TRIM);
    
%     ridge_signal = [];
%     [h,w] = size(BIN0);
%     for i = 2:h-1,
%     for j = 2:w-1,
%         if MASK(i,j) == 1,
%             ridge_signal  = [ridge_signal IM0(i,j)];
%         end
%     end
%     end
    
    %Orientation
    [orientIm, rel] = fun_RIDGE_ORIENTATION(uint8(IM0),1,3,3);
    orientIm = rad2deg(orientIm);
    filename = [imageName(1:end-4),'_set.csv'];
    [fid, msg] = fopen(filename, 'wt');
    if fid < 0
      error('Could not open file "%s" because "%s"', fid, msg);
    end
    for k1 = 1:length(LOC_BIF_TRIM),
        A = LOC_BIF_TRIM{k1};
        theta = orientIm(A(1),A(2));
        fprintf(fid, '%s,%d,%d,%f\n', 'BIF',A(2), A(1),theta);
    end
    
    for k1 = 1:length(LOC_RID_TRIM),
        A = LOC_RID_TRIM{k1};
        theta = orientIm(A(1),A(2));
        fprintf(fid, '%s,%d,%d,%f\n', 'RID',A(2), A(1),theta);
    end
    fclose(fid);
    
    num_minutiae = length(LOC_BIF_TRIM) + length(LOC_RID_TRIM)
%     filename = 'D:\Work\Acad\sem 7\BTP\num_minutiae_Digital_380x380.csv';
%     [fid, msg] = fopen(filename, 'a+');
%     className = imageName(51:54);
%     if sum(className ~= 'Live')>0
%         className = imageName(56:58);
%     end
%     fprintf(fid, '%s,%d \n', className,num_minutiae);
%     fclose(fid);
    %% figure; imshow(uint8(255*BIN0));
%     figure; imshow(255-uint8(IM0)); hold on;
%     for k1 = 1:length(LOC_BIF_TRIM),
%         A = LOC_BIF_TRIM{k1};
%         plot(A(2), A(1),'r*'); hold on;
%         theta = orientIm(A(1),A(2));
%         u = cos(theta)*10;
%         v = sin(theta)*10;
%         quiver(A(2), A(1),u,v);hold on;
%     end
%     for k1 = 1:length(LOC_RID_TRIM),
%         A = LOC_RID_TRIM{k1};
%         plot(A(2), A(1),'g*'); hold on;
%         theta = orientIm(A(1),A(2));
%         u = cos(theta)*10;
%         v = sin(theta)*10;
%         quiver(A(2), A(1),u,v);hold on;
%     end
%     title('FINAL MINUTIA SET');
%     hold off;
end