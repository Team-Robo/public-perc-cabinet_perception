# One ranked candidate from GetObjectGroup.

string  group_id
uint8   shelf_id
string  side
float32 score          # calibrated [0, 1]; comparable across methods
bool    acceptable     # passes the NO_MATCH threshold; placement may only fall back to acceptable ones
string  reason         # spoken sentence for THIS candidate, so the robot announces the
                       # right shelf even when placement falls back to candidate #2
