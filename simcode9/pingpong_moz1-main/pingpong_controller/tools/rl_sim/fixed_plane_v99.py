"""V99 actor history: the current and seven preceding 200 Hz ticks."""

HISTORY_LAGS_TICKS = tuple(range(8))
HISTORY_LENGTH = 8
ACTOR_OBS_DIM = 156
ACTOR_OBSERVATION_SCHEMA = (
    "fixed_plane_v99_contiguous_history_contact_geometry_156d"
)
