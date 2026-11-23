# One cabinet shelf level. Geometry here is what the placement side needs to
# decide whether a candidate group is actually reachable/free.

uint8   shelf_id          # 1 = lowest
float64 height_m          # shelf surface height above the floor (metres)
float64 free_width_m      # widest free span on this shelf (metres); 0 = full
