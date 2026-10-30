"""Prepare sensor logs at producers and finish them in the recording process.

The 200 Hz thread only transports immutable cache references. None of the
formatting, vector validation, age calculation or JSON work runs there.
"""
import math

import numpy as np


def finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def make_sensor_recording_sample(source, position, velocity, valid, stamp,
                                 measurement, *, frame_aligned=True,
                                 detector_frame="waist03_cartesian"):
    """Called only by the sensor producer, outside the policy thread."""
    measurement = measurement if isinstance(measurement, dict) else {}
    sync = dict(measurement.get("clockSync") or {})
    sensor = {
        "enabled": True, "available": True, "valid": False,
        "position": None, "velocity": None, "controlFrame": "sim_frame",
        "frameAligned": bool(frame_aligned), "detectorFrame": detector_frame,
        "measurementTimestampS": finite_number(stamp),
        "sourceTimestampS": finite_number(measurement.get("sourceTimestampS", sync.get("sourceTimestampS"))),
        "receiverTimestampS": finite_number(measurement.get("receiverTimestampS", sync.get("receiverTimestampS"))),
        "measurementAgeS": None, "clockSync": sync,
        "error": measurement.get("error"),
    }
    if source == "camera":
        sensor["measurement"] = {
            key: (finite_number(value) if isinstance(value, (float, int, np.number))
                  and not isinstance(value, bool) else value)
            for key, value in measurement.items()
            if key != "clockSync" and (value is None or isinstance(value, (str, bool, float, int, np.number)))
        }
    else:
        sensor["measurement"] = {
            key: measurement.get(key) for key in (
                "frameIdx", "markerId", "tracked", "packetCount",
                "parseErrorCount", "velocityRejectCount", "velocityTimeSource", "markerSelection", "udp")
        }
    if valid and sensor["measurementTimestampS"] is not None and sync.get("ready", False):
        position = np.asarray(position, dtype=np.float32).reshape(3)
        velocity = np.asarray(velocity, dtype=np.float32).reshape(3)
        if np.all(np.isfinite(position)) and np.all(np.isfinite(velocity)):
            sensor.update(valid=True, position=position.tolist(), velocity=velocity.tolist())
    return sensor


def complete_ball_sensor_recording_snapshot(snapshot):
    """Validate a transported cache at the tick time, never at disk-write time.

    snapshot = (receiver_time, monotonic_time, camera_envelope, mocap_envelope,
                camera_enabled, mocap_enabled)
    envelope = (cached_packet, error, closed, current_sync_request, timeout_s)
    cached_packet = (prepared_sensor, packet_sync_request, receipt_monotonic)
    """
    receiver_time, monotonic_time, camera, mocap, camera_enabled, mocap_enabled = snapshot
    payload = {"receiverTimeS": finite_number(receiver_time)}
    for source, envelope, enabled in (("camera", camera, camera_enabled), ("mocap", mocap, mocap_enabled)):
        sensor = {
            "enabled": bool(enabled), "available": envelope is not None,
            "valid": False, "position": None, "velocity": None,
            "controlFrame": "sim_frame", "frameAligned": False,
            "measurementTimestampS": None, "sourceTimestampS": None,
            "receiverTimestampS": None, "measurementAgeS": None,
            "clockSync": {}, "measurement": None, "error": None,
        }
        payload[source] = sensor
        if not enabled or envelope is None:
            continue
        packet, error, closed, sync_request, timeout = envelope
        if packet is not None:
            prepared, packet_request, receipt_monotonic = packet
            sensor.update(prepared)
            sync = dict(sensor["clockSync"])
            sensor["clockSync"] = sync
            stamp = sensor["measurementTimestampS"]
            age = None if stamp is None or payload["receiverTimeS"] is None else payload["receiverTimeS"] - stamp
            sensor["measurementAgeS"] = finite_number(age)
            arrival_age = monotonic_time - receipt_monotonic
            receiver_stamp = sensor["receiverTimestampS"]
            clock_ok = bool(receiver_stamp is not None and payload["receiverTimeS"] is not None
                            and abs(payload["receiverTimeS"] - receiver_stamp - arrival_age) <= .25)
            if packet_request != sync_request:
                sync.update(ready=False, status="awaiting_resynchronization")
            elif receiver_stamp is None:
                sync.update(ready=False, status="waiting_for_first_observation")
            elif not clock_ok:
                sync.update(ready=False, status="receiver_clock_changed")
            elif age is None or not 0. <= age < timeout or not 0. <= arrival_age < timeout:
                sync.update(ready=False, status="waiting_for_fresh_observation")
            sensor["valid"] = bool(sensor["valid"] and sync.get("ready", False) and not error and not closed)
        if error or closed:
            sensor["error"] = error or f"{source} source stopped"
            sensor["clockSync"].update(ready=False, status="source_unavailable")
        if not sensor["valid"]:
            sensor["position"] = sensor["velocity"] = None
    camera, mocap = payload["camera"], payload["mocap"]
    payload["bothValid"] = bool(camera["valid"] and mocap["valid"])
    payload["comparisonReady"] = bool(payload["bothValid"] and camera["frameAligned"] and mocap["frameAligned"])
    camera_stamp, mocap_stamp = camera["measurementTimestampS"], mocap["measurementTimestampS"]
    payload["measurementTimeDeltaS"] = None if camera_stamp is None or mocap_stamp is None else camera_stamp - mocap_stamp
    return payload
