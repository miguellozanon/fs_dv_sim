# A single traffic cone (FS Driverless Specification colours).
uint8 BLUE=0          # left track border
uint8 YELLOW=1        # right track border
uint8 ORANGE_SMALL=2  # entry / exit lanes
uint8 ORANGE_BIG=3    # before / after start, finish and timekeeping lines
uint8 UNKNOWN=4

uint32 id
uint8 color
geometry_msgs/Point position   # in header.frame_id of the enclosing ConeArray
float32 confidence             # 1.0 for ground truth
bool knocked                   # ground truth only: cone is down / out
