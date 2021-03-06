function C = unvectCurvelet(x, S, maskC)
% UNVECTCURVELET Reshapes vector of Curvelet coefficients x to Curvelet with structure S and mask maskC
%  C = unvectCurvelet(x, S, maskC)
%
% Copyright (C) 2013 Marta M. Betcke

C = S; %initialize with S for speed

if nargin < 3 || isempty(maskC)
  % If no mask specified, construct all-1 mask
  maskC = constCurvelet(S,1);
end

i = 0;
for s = 1:length(S) %loop through scales
  for w = 1:length(S{s}) %loop through angles
    sizeCsw = size(S{s}{w});
    %C{s}{w} = reshape(x(i + (1:prod(sizeCsw))), sizeCsw(1), sizeCsw(2));
    %i = i + prod(sizeCsw);
    C{s}{w} = zeros(sizeCsw);
    lCsw = nnz( maskC{s}{w}(:) );
    C{s}{w}( logical( maskC{s}{w}(:) ) ) = x( i + ( 1:lCsw ) );    
    i = i + lCsw;
  end
end