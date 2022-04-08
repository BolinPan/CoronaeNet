function visibleC = visibleCoarseScale(C,theta,setting)
%
% VISIBLECOARSESCALE applied the visible filter in coarse scale Curvelet 
% in frequency domain.
%
%  INPUTS:
%   C       - a structure Curvelet
%   theta   - visible angle
%   setting - a structure contains image grid information
%
%  OUTPUTS:
%   visibleC  - Curvelet representation with visible coarse scale
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke

% initialization
visibleC = C;

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
        coarseScaleCoeffMask_kxky(abs(coarseGrid.kx) < abs(coarseGrid.ky/tan(theta))) = 0;
        
        % compute the inverse FFT
        coarseScaleCoeffRec = real(fftshift(ifftn(ifftshift(coarseScaleCoeffMask_kxky)))*sqrt(numel(coarseScaleCoeffMask_kxky)));
        
        % take the bottom half of the recovery
        coarseScaleCoeffVis = coarseScaleCoeffRec(Nx/2+1:end,:);
        
        % put coarse Scale back to structure
        visibleC{1}{1} = coarseScaleCoeffVis;
    case 3
        % no implementation
        noImple
end

