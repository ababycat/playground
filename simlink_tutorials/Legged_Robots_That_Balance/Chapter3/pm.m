% unit m, kg/(m/s^2)
% 初始化角度用的角度单位，计算时内部使用弧度为单位
body_angle_kp = 60;

% 
body_x = 0.97;
body_y = 0.1;
body_height = 0.1;
body_density = 3;

% joint 
upper_leg_joint_damping = 0.01;
lower_leg_spring_stiffness = 4;
lower_leg_joint_damping = 0.3;

% 
upper_leg_length = 0.39;
upper_leg_radius = 0.02;
upper_leg_density = 1;

%
lower_leg_length = 0.3;
lower_leg_radius = 0.01;
lower_leg_density = 1;

% 
foot_radius = 0.05;
foot_density = 1;

% 
plane_x = 100;
plane_y = 2;
plane_height = 0.2;
plane_density = 1000;

% 
body_height_init = 1.5;
contact_stiffness = 800;
contact_damping = 500;
hip_angle_init = 1;
plane_depth_to_ref_frame = plane_height/2+1e-2;
world_to_foot_offset = plane_depth_to_ref_frame+foot_radius;

% %
% global leg_len_init;
% global torl;

leg_len_init = 0.69;
torl = 0.01;
% stance 周期 从仿真结果测量出
Ts = 0.33;
