function G1 = fun_GABOR_SINE_BASE(TR, sigma1)
% Kannan Karthik
% Ref: Fingerprint Recognition, AK. Jain et al.
% Aug-Sept 2018

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
N = 1;
G1 = G_SINE;


end

