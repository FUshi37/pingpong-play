"""Standalone visualization subprocess worker (minimal dependencies)."""
import cv2
import numpy as np


def viz_main(conn, window_name: str) -> None:
    """Standalone process that owns the Qt window.

    Runs in its own Python interpreter so it has a clean main thread for Qt's
    event loop. Receives `(annotated_bgr_ndarray,) | None` over `conn` (an
    `mp.Connection`); a `None` payload signals shutdown.

    Defined in a separate minimal module so multiprocessing's `spawn` can
    import it without loading torch / ultralytics / pyrealsense2 / rclpy, which
    would hang or crash during the child's initialization.
    """
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error:
        # No display / no Qt plugin / etc. Drain the pipe so the parent
        # never blocks on a full pipe, then exit.
        try:
            while True:
                if not conn.poll(timeout=0.1):
                    break
                payload = conn.recv()
                if payload is None:
                    break
        except (EOFError, OSError):
            pass
        return

    last_frame = None
    try:
        while True:
            # Drain everything currently in the pipe so we always show the
            # newest frame; this keeps display latency low even if YOLO
            # produces frames faster than imshow can paint them.
            new_frame = None
            stop = False
            try:
                while conn.poll(timeout=0.0):
                    payload = conn.recv()
                    if payload is None:
                        stop = True
                        break
                    new_frame = payload
            except (EOFError, OSError):
                stop = True

            if stop:
                break

            if new_frame is not None:
                last_frame = new_frame

            if last_frame is not None:
                try:
                    cv2.imshow(window_name, last_frame)
                except cv2.error:
                    break

            # waitKey both pumps Qt events and gives the user a way to
            # close the window. 30 ms ≈ 33 Hz refresh, plenty for a
            # debug view and easy on the CPU.
            try:
                key = cv2.waitKey(30) & 0xFF
            except cv2.error:
                break
            if key in (ord('q'), 27):
                break

            # If no frame has arrived yet, also block briefly on the pipe
            # so we don't spin at 100% CPU before the first detection.
            if new_frame is None and last_frame is None:
                try:
                    if conn.poll(timeout=0.05):
                        continue
                except (EOFError, OSError):
                    break
    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        try:
            conn.close()
        except OSError:
            pass
