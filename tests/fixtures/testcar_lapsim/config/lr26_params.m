function veh = lr26_params()
% FICTIONAL TEST CAR for the automated tests (CI). Not the LR26: round, textbook values of a generic
% rear-wheel-drive Formula Student electric car. The file has the LR26 name because the tools look for
% config/lr26_params.m. Never use it for decisions.
veh.name_id   = 0;          % [-]
veh.lf        = 0.85;       % [m]    CG to front axle
veh.lr        = 0.70;       % [m]    CG to rear axle
veh.h_cog     = 0.30;       % [m]    CG height
veh.sf        = 1.20;       % [m]    front track width
veh.sr        = 1.15;       % [m]    rear track width
veh.m         = 300;        % [kg]   mass incl. driver
veh.f_roll    = 0.015;      % [-]    rolling resistance coefficient
veh.g         = 9.81;       % [m/s^2]
veh.c_w        = 1.20;      % [-]    drag coefficient
veh.c_z        = 3.00;      % [-]    downforce coefficient
veh.A          = 1.00;      % [m^2]  frontal area
veh.aero_eff   = 1.00;      % [-]    aero efficiency factor
veh.aero_bal_f = 0.45;      % [-]    front share of downforce
veh.rho        = 1.20;      % [kg/m^3]
veh.drive_awd   = 0;        % [-]    0 = RWD
veh.P_max       = 80000;    % [W]
veh.P_at_accumulator = 0;   % [-]
veh.eta_motor   = 0.90;     % [-]
veh.T_motor     = 60;       % [Nm]   per driven axle
veh.ax_max_launch = 12;     % [m/s^2]
veh.i_trans  = 11;          % [-]
veh.n_max    = 16000;       % [rpm]
veh.e_i      = 1.05;        % [-]
veh.eta_g    = 0.96;        % [-]
veh.circ_ref = 1.50;        % [m]
veh.Fz0      = 750;         % [N]
veh.mux      = 1.50;        % [-]
veh.muy      = 1.50;        % [-]
veh.dmux_dFz = -1.0e-4;     % [1/N]
veh.dmuy_dFz = -1.0e-4;     % [1/N]
veh.tire_exp = 2;           % [-]
veh.brake_bias = NaN;       % [-]    ideal
veh.P_regen   = 20000;      % [W]
veh.eta_regen = 0.70;       % [-]
veh.util_x = 1.0;           % [-]
veh.util_y = 1.0;           % [-]
veh.P_max_endurance        = 30000;  % [W]
veh.endurance_torque_scale = 1.0;    % [-]
veh.v_lim = 33;             % [m/s]
end
