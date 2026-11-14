# Autonomous System status. State and mission values match the FS Driverless
# Specification CAN message "DV system status" (0x502) so they can be logged 1:1.
std_msgs/Header header

uint8 AS_OFF=1
uint8 AS_READY=2
uint8 AS_DRIVING=3
uint8 AS_EMERGENCY=4
uint8 AS_FINISHED=5

uint8 MISSION_NONE=0
uint8 MISSION_ACCELERATION=1
uint8 MISSION_SKIDPAD=2
uint8 MISSION_TRACKDRIVE=3
uint8 MISSION_EBS_TEST=4
uint8 MISSION_INSPECTION=5
uint8 MISSION_AUTOCROSS=6

uint8 state
uint8 mission
uint16 lap_count          # completed laps
uint16 laps_target
bool mission_complete     # all laps / runs done: the vehicle must now come to a stop
uint16 cones_hit          # simulator only: cones down / out
uint16 off_course         # simulator only: off-course events
float32 mission_time      # [s] since the start line was crossed
float32 last_lap_time     # [s]
float32 best_lap_time     # [s]
float32 penalty_time      # [s]
string message
