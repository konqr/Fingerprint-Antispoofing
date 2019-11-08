function G2 = fun_GABOR_SMOOTHER_BASE(sigma2)
% Kannan Karthik
% Ref: Fingerprint Recognition, AK. Jain et al.
% Aug-Sept 2018

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
N = 1;
G2 = G_SMOOTH;


end

