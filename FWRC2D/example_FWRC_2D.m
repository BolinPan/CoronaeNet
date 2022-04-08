% An example to demonstrate fully wedge restricted Curvelet transform on 2D 
% ellipse phantom with line sensors geometry.
%
% Add k-wave and Curvelet toolbox into the path for running this script.
%
%
% Copyright (C) 2022 Bolin Pan & Marta M. Betcke


clear all; close all; clc;

% Reset rand, randn, randi to default seed
rng('default')

% define path
path = 'CoronaeNet/';

% load phantom
load([path,'measurements/ellipsePhantom.mat']) 

% define maximum limited angle
MLangle = 45; % 45 degree


%% construct setting
setting        = [];
setting.dim        = 2;
setting.Ny         = size(p0,1);
setting.Nx         = size(p0,2);
setting.dx         = 1e-4; % grid point spacing in the x direction [m]
setting.dy         = 1e-4; % grid point spacing in the x direction [m]
setting.soundSpeed = 1500; % Sound speed in the medium in [m/s]

% make kgrid
setting.kgrid = makeGrid(setting.Nx, setting.dx, setting.Ny, setting.dy);

% set up time steps manually, matching the dx discretisation 1:1
domDiam =  sqrt((setting.Nx*setting.dx)^2 + (setting.Ny*setting.dy)^2); % domain diameter
setting.dt = setting.dx/setting.soundSpeed;
setting.Nt = ceil(domDiam/(setting.soundSpeed*setting.dt));
setting.t_array = (0:setting.Nt-1)*setting.dt;
theta = MLangle/180*pi;


%% restricted Curvelets transform of p0
cs_transform.class          = 'curvelet'; 
cs_transform.nscales        = 3;  % number of scales, higher means more feature windows
cs_transform.nangles_coarse = 32; % coarse angles, depend on the restricted angle 
cs_transform.type           = 'db4'; % Type of wavelets 'db1' for Haar, 'db4' etc.
cs_transform.real           = 1; % real curvelets (true/false)
cs_transform.finest         = 1;  
cs_transform.imageSize      = size(p0);
cs_transform.theta          = theta; % angle for coarse scale visible filter
cs_transform.coarseVisible  = false; % standard Curvelet transform

% obtain restricted Curvelets and angle information 
[rCurvelets_p0, non_rCurvelets_p0, allCurveletAnglesInfo_p0] = getRestrictedCurvelet(cs_transform);

% handle to the constructor function of Curvelet tranform
disp(['Constructing standard transform (p0): ' cs_transform.class])
cs_transform.constructor = str2func(['@(T) ' cs_transform.class 'CSTransform2D_(T)']);

%%% construct standard transform %%%
[cs_transform_p0, cs_transform_p0_tag] = cs_transform.constructor(cs_transform);

% update to restricted Curvelet 
cs_transform_p0_ = cs_transform_p0;
cs_transform_p0_.coarseVisible  = false; % visible filter in coarse scale
cs_transform_p0_.setting = setting; % require for coarse scale visible filter

% define restircted angles on the frist scale (nagnles_coarse = 128) of upper tilling
cs_transform_p0_.maskC = restrictCurvelet2D(cs_transform_p0_.S,rCurvelets_p0,'explicit');

%%% construct restricted Curvelet
[cs_transform_p0_, cs_transform_p0_tag_] = cs_transform_p0_.constructor(cs_transform_p0_);

% update to restricted Curvelet with visible filter in coarse scale
cs_transform_p0_f = cs_transform_p0;
cs_transform_p0_f.coarseVisible  = true; % visible filter in coarse scale
cs_transform_p0_f.setting        = setting; % require for coarse scale visible filter

% define restircted angles on the frist scale (nagnles_coarse = 128) of upper tilling
cs_transform_p0_f.maskC = restrictCurvelet2D(cs_transform_p0_f.S,rCurvelets_p0,'explicit');

%%% construct fully wedge restricted Curvelet transform %%%
[cs_transform_p0_f, cs_transform_p0_f_tag_] = cs_transform_p0_f.constructor(cs_transform_p0_f);


%% forward backward Curvelet transform
p0rec = cs_transform_p0.iPsi(cs_transform_p0.Psi(p0));       % C
p0rec_ = cs_transform_p0_.iPsi(cs_transform_p0_.Psi(p0));    % WRC
p0rec_f = cs_transform_p0_f.iPsi(cs_transform_p0_f.Psi(p0)); % FWRC


%% display and compare
% Curvelet coefficients
figure;
subplot(1,3,1);imagesc(abs(cs_transform_p0.imagePsi(cs_transform_p0.Psi(p0))));axis image;title('C') 
subplot(1,3,2);imagesc(abs(cs_transform_p0_.imagePsi(cs_transform_p0_.Psi(p0))));axis image;title('WRC') 
subplot(1,3,3);imagesc(abs(cs_transform_p0_f.imagePsi(cs_transform_p0_f.Psi(p0))));axis image;title('FWRC') 

% compare recovery
figure
subplot(2,3,1);imagesc(p0rec);title('p0C');axis image;colorbar
subplot(2,3,2);imagesc(p0rec_);title('p0WRC');axis image;colorbar
subplot(2,3,3);imagesc(p0rec_f);title('p0FWRC');axis image;colorbar
subplot(2,3,4);imagesc(p0rec-p0);title('p0C - p0');axis image;colorbar
subplot(2,3,5);imagesc(p0rec_-p0);title('p0WRC - p0');axis image;colorbar
subplot(2,3,6);imagesc(p0rec_f-p0);title('p0FWRC - p0');axis image;colorbar

