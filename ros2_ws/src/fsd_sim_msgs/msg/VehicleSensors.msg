# Proprioceptive vehicle sensors (what the VCU / inverters report on CAN).
std_msgs/Header header
float32 wheel_rpm_fl
float32 wheel_rpm_fr
float32 wheel_rpm_rl
float32 wheel_rpm_rr
float32 steering_angle    # measured road-wheel angle [rad], left positive
float32 ground_speed_x    # optical ground-speed sensor [m/s], NaN if not fitted
float32 ground_speed_y    # [m/s]
