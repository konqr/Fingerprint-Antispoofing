% Konark Jain

imageName = 'D:\Work\Acad\sem 7\BTP\data\testGreenBit\017_21_0\Live\pres_A_01\FLAT_THUMB_LEFT.bmp'

IM = imresize(imread(imageName),[250 250]);
points = readtable([imageName(1:end-4), '_250x250.csv']);

figure; imshow(IM); hold on;
for k1 = 1:height(points),
    A = points{k1,2:3};
    if points{k1,1}{1} == 'BIF',
        plot(A(1), A(2),'r*'); hold on;
    else
        plot(A(1), A(2),'g*'); hold on;
    end
end
height(points)
title('FINAL MINUTIA SET - INDEX LEFT Live');