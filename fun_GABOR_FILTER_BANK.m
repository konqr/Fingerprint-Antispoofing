function [FINAL_FUSED] = fun_GABOR_FILTER_BANK(IM0, GSM)

% Kannan Karthik
% Reference: Fingerprint Recognition, AK JAIN et al.
% Aug-Sept, 2018

THQ = [0:5:179];

% ACTUAL FILTERING


LTH = length(THQ);
for i = 1:LTH,
    G{i} = imrotate(GSM,-THQ(i)+90,'bilinear');
    B = G{i};
    B0 = B(:);
    E = sqrt(sum(B0.^2));
    G{i} = G{i}/E;
    IMFILT_BANK{i} = imfilter(double(IM0),G{i});
end

[a1,b1] = size(IMFILT_BANK{1});
for i = 1:a1,
    for j = 1:b1,
        for c = 1:LTH,
            B = IMFILT_BANK{c};
            VAL(c) = B(i,j);
        end
        MAX_FILT(i,j) = max(VAL);
        clear VAL
    end
end

FINAL_FUSED = MAX_FILT;

end

