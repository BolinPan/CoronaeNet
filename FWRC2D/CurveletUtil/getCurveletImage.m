function x = getCurveletImage(C, cs_transform, addNoise, mode, format)
%
% GETCURVELETIMAGE compute Curvelet coefficients in image domain
%
% Inputs
%   C         - Cell array containing curvelet coefficients (see
%               description in fdct_wrapping.m)
%   cs_transform - a structure contains:
%           zeroMaskC - Cell array containing zero curvelet coefficients
%             is_real - As used in fdct_wrapping.m
%                M, N - Size of the image to be recovered (not necessary if
%                       finest = 2)
%   addNoise - add gaussian noise to images
%   mode - mode for calculating the Curvelet
%
% Outputs
%   x         - curvelet coefficients in image domain
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke


% obtain cs_transform info
zeroMaskC = cs_transform.S;
coarseMask = zeros(size(zeroMaskC{1}));
zeroMaskC{1} = [];
zeroMaskC{1}{1} = coarseMask;
zeroMaskC{1}{2} = coarseMask;

is_real = cs_transform.real;
M = cs_transform.imageSize(1);
N = cs_transform.imageSize(2);

% initialization
x = [];
switch format
    case 'split'
        % nothing need to do
    case 'non-split'
        Xvis = 0;
        Xinv = 0;
end

% get restricted Curvelet indices
rCurvelets = cs_transform.rCurvelets_p0;
switch mode
    case 'all'
        % nothing need to do
    case 'invisible'
        rCurvelets{1} = 2; % second one is the invisible coarse coefficient
end

% loop through all scales and angles from semi-finest scale
for s = 1:length(C)
    % loop through all angles
    for w = 1:length(C{s})
        % assign one Curvelet
        maskC = zeroMaskC;
        maskC{s}{w} = C{s}{w};
        % compute image Curvelet
        xTemp = computeCurveletImage(maskC, is_real, M, N, s, mode);
%         switch mode
%             case 'all'
%                 % nothing to do
%             case 'invisible'
%                 if ismember(w, rCurvelets{s})
%                     xTemp = zeros(size(xTemp));
%                 end
%         end
        % choose format
        switch format
            case 'split'
                % add noise
                if addNoise
                    x{s}(w,:,:) = 1e-4*randn(size(xTemp)) + xTemp;
                end
                x{s}(w,:,:) = xTemp;
            case 'non-split'
                % add Noise 
                if addNoise
                    xTemp = 1e-4*randn(size(xTemp)) + xTemp;
                end
                if s == 1
                    x{s}(w,:,:) = xTemp;
                else
                    if ismember(w,rCurvelets{s})
                        % invisible part
                        xTempFFTinv = fftshift(fft2(ifftshift(xTemp)))/sqrt(prod(size(xTemp)));
                        Xinv = Xinv + xTempFFTinv;    
                    else
                        % visible part
                        xTempFFTvis = fftshift(fft2(ifftshift(xTemp)))/sqrt(prod(size(xTemp)));
                        Xvis = Xvis + xTempFFTvis; 
                    end
                end
        end
    end
    % output
    switch format
        case 'split'
            % nothing to do
        case 'non-split'
            if s > 1
                % taking invser FFT to recover the images
                x{s}(1,:,:) = real(fftshift(ifft2(ifftshift(Xvis)))*sqrt(prod(size(Xvis)))); % visible
                x{s}(2,:,:) = real(fftshift(ifft2(ifftshift(Xinv)))*sqrt(prod(size(Xinv)))); % invisible
            end
            Xvis = 0;
            Xinv = 0;
    end
end

