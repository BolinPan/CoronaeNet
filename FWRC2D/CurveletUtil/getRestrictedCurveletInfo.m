function restrictedCurveletInfo = getRestrictedCurveletInfo(theta,allCurveletAngles)
% GETRESTRICTEDCURVELETINFO gets information of restricted Curvelets in all
% coarse scale according to angle theta and specific structure of Curvelet transform
%
%  rCurvelets = getRestrictedCurvelet(theta,cs_transform)
%
%  INPUTS:
%   theta - the maximum angle that wave front impinges on the detector in
%           radian (0, pi/2)
%
%   allCurveletAngles - angles of all Curvelet wedges in image domain
%           obtaine from getRestrictedCurvelet.m 
%
%  OUTPUTS:
%   restrictedCurveletInfo - a structure containing the infomation of
%           restricted Curvelets, 1st Column - indices of Curvelet wedges
%            2nd column - the wedges angles (0,pi/2), 3rd column -
%            indicates restricted Curvelets (0) and healthy Curvelets (1)
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke

% change theta into degree
thetaD = (theta/pi*180); % in degree

% initialization
restrictedCurveletInfo = [];
angleCInfo = [];

% non directional coarse scale
restrictedCurveletInfo{1} = [];

% loop through all scales and angles
for s = 2:length(allCurveletAngles)
    % angle indices
    angleCInfo(:,1) = [1:length(allCurveletAngles{s})]';
    % all Curvelet angles
    angleCInfo(:,2) = allCurveletAngles{s};
    % indicate restricted Curvelet as zero
    indicator = allCurveletAngles{s};
    indicator(indicator>(theta/pi*180)) = 0;
    indicator(indicator~=0) = 1;
    angleCInfo(:,3) = indicator;
    restrictedCurveletInfo{s} = angleCInfo;
    angleCInfo = [];
end
