function [ STAT ] = fun_PATCH_STAT(PATCH)
% Kannan Karthik
%   Detailed explanation goes here
PATCH = PATCH.data;
STAT = median(PATCH(:));

end

