function decompC = decompCoarseScale(C,theta,setting)
%
% DECOMPCOARSESCALE decompose coarse scale into visible and invisible parts
%
%
%  INPUTS:
%   C       - a structure contains all coefficients
%   theta   - visible angle
%   setting - a structure contains image grid information
%
%  OUTPUTS:
%   decompC  - Curvelet representation with visible and invisible coarse scale
%
%
% Copy right (C) 2022 Bolin Pan & Marta M. Betcke


% initialization
decompC = C;

% visible filter in coarse scale coefficient
switch setting.dim
    case 2
        % take the coarse scale coefficient
        coarseScaleCoeff = C{1}{1};
        
        % mirror the coarse scale coefficient for fft use
        coarseScaleCoeffMask = [flipdim(coarseScaleCoeff, 1); coarseScaleCoeff];

        % compute the unitary FFT of the coarse scale coefficient to yield p(kx, ky)
        coarseScaleCoeffMask_kxky = fftshift(fftn(ifftshift(coarseScaleCoeffMask)))./sqrt(numel(coarseScaleCoeffMask));
        
        % get size of FFT image and compute kgrid
        [Nx,Ny] = size(coarseScaleCoeffMask_kxky);
        coarseGrid = kWaveGrid(Nx, setting.dx, Ny, setting.dy);
        
        % apply visible filter (exclude the invisible part)
        coarseScaleCoeffMask_kxkyVis = coarseScaleCoeffMask_kxky;
        coarseScaleCoeffMask_kxkyInv = coarseScaleCoeffMask_kxky;
        coarseScaleCoeffMask_kxkyVis(abs(coarseGrid.kx) < abs(coarseGrid.ky/tan(theta))) = 0;
        coarseScaleCoeffMask_kxkyInv(abs(coarseGrid.kx) >= abs(coarseGrid.ky/tan(theta))) = 0;

        % compute the inverse FFT
        coarseScaleCoeffRecVis = real(fftshift(ifftn(ifftshift(coarseScaleCoeffMask_kxkyVis)))*sqrt(numel(coarseScaleCoeffMask_kxkyVis)));
        coarseScaleCoeffRecInv = real(fftshift(ifftn(ifftshift(coarseScaleCoeffMask_kxkyInv)))*sqrt(numel(coarseScaleCoeffMask_kxkyInv)));

        % take the bottom half of the recovery
        coarseScaleCoeffVis = coarseScaleCoeffRecVis(Nx/2+1:end,:);
        coarseScaleCoeffInv = coarseScaleCoeffRecInv(Nx/2+1:end,:);

        % put coarse Scale back to structure
        decompC{1} = [];
        decompC{1}{1} = coarseScaleCoeffVis;
        decompC{1}{2} = coarseScaleCoeffInv;
    case 3
        % no implementation
        noImple
end