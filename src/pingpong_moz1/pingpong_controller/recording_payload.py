"""Copy numeric evidence cheaply; format it only in the recording process."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecordingVector:
    values: np.ndarray
    digits: int = 6

    def __post_init__(self):
        # Queues serialize asynchronously. Never retain a mutable controller
        # view that the following tick can overwrite before it is pickled.
        values = np.array(self.values, dtype=np.float64, copy=True).reshape(-1)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)

    def __reduce__(self):
        # A flat immutable byte payload avoids pickling a separate ndarray
        # header for every vector on the control process's queue feeder.
        return (_restore_recording_vector, (self.values.tobytes(), self.digits))


def _restore_recording_vector(payload, digits):
    result = object.__new__(RecordingVector)
    object.__setattr__(result, "values", np.frombuffer(payload, dtype=np.float64))
    object.__setattr__(result, "digits", digits)
    return result


@dataclass(frozen=True)
class RecordingCameraMeasurement:
    values: dict


def recording_camera_measurement(values, scalar_keys=None):
    if not isinstance(values, dict):
        return None
    # Raw diagnostics can also reference images, arrays and nested receiver
    # state. Carry only the scalar whitelist; never send those buffers merely
    # because numeric formatting moved to another process. An unsupported
    # whitelisted value stays unsupported so the old formatter omits it.
    scalars = {
        key: value if value is None or isinstance(value, (str, bool, int, float, np.number)) else ()
        for key, value in values.items() if scalar_keys is None or key in scalar_keys
    }
    return RecordingCameraMeasurement(scalars)


def finish_recording_payload(value, camera_formatter):
    """Preserve the exact legacy Python rounding and JSON schema off-thread."""
    if isinstance(value, RecordingVector):
        return [round(float(v), value.digits) for v in value.values]
    if isinstance(value, RecordingCameraMeasurement):
        return camera_formatter(value.values)
    if isinstance(value, dict):
        return {key: finish_recording_payload(item, camera_formatter) for key, item in value.items()}
    if isinstance(value, list):
        return [finish_recording_payload(item, camera_formatter) for item in value]
    return value
