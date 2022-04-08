function maskC = restrictCurvelet2D(C,rAngle,type)
%
% RESTRICTCURVELET2D constructs Curvelet mask for 3D data curvelet representation restricting
%   curvelet angles to be 0 (due to causality of wave propagation)
%
%
%  INPUTS:
%   C - a dummy curvelet of 0-image
%
%  OUTPUTS:
%   maskC  - Curvelet mask for 3D data restrict curvelet representation
%   maskVC - vectorized maskC
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke


% construct all-1 curvelet with structure of C
maskC = constCurvelet(C,true);

% total scales of curvelet
scales = length(maskC); 


% Set maskC to 0 at scales and angles which should be excluded
switch type
    case {'auto'}
        %loop through scales
        for s = 2:scales
            if s == 2
                angles = [rAngle, rAngle+(length(maskC{2})/2)];
            elseif s == 3
                rAngleUp = (2*rAngle(1)-1):(2*rAngle(end)+1);
                angles = [rAngleUp, rAngleUp+length(maskC{2})];
            elseif s == 4
                rAngleUp = (2*rAngle(1)-1):(2*rAngle(end)+1);
                angles = [rAngleUp, rAngleUp+length(maskC{2})];
            end
            %loop through angles. Those are infact wedges of tan(angles)
            for w = angles
                maskC{s}{w}(:) = false;  
            end
        end
    case {'explicit'}
        %loop through scales
        for s = 2:scales
            % length of restricted wedges
            L = length(rAngle{s});
            % remove restricted wedges
            for w = 1:L
                maskC{s}{rAngle{s}(w)}(:) = false;  
            end
        end
end
end

