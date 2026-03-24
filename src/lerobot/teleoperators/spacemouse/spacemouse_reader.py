"""SpaceMouse reader using libspnav (non-blocking polling in a background thread).

Adapted from spacemouse2rm75b/spacemouse_input.py.
"""

import ctypes
import ctypes.util
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)


# ---------- ctypes event structs (matching spnav.h) ----------

class MotionEvent(ctypes.Structure):
    _fields_ = [
        ("type",   ctypes.c_int),
        ("x",      ctypes.c_int),
        ("y",      ctypes.c_int),
        ("z",      ctypes.c_int),
        ("rx",     ctypes.c_int),
        ("ry",     ctypes.c_int),
        ("rz",     ctypes.c_int),
        ("period", ctypes.c_uint),
        ("data",   ctypes.c_void_p),
    ]


class ButtonEvent(ctypes.Structure):
    _fields_ = [
        ("type",  ctypes.c_int),
        ("press", ctypes.c_int),
        ("bnum",  ctypes.c_int),
    ]


class SpnavEvent(ctypes.Union):
    _fields_ = [
        ("type",   ctypes.c_int),
        ("motion", MotionEvent),
        ("button", ButtonEvent),
    ]


SPNAV_EVENT_MOTION = 1
SPNAV_EVENT_BUTTON = 2

# Common paths to look for libspnav
_HERE = os.path.dirname(os.path.abspath(__file__))

_COMMON_LIBSPNAV_PATHS = [
    # 优先查找本目录下的 bundled 版本（便于跨机器部署）
    os.path.join(_HERE, "libspnav.so.0"),
    os.path.join(_HERE, "libspnav.so.0.4"),
    # 系统路径
    "libspnav.so.0",
    "libspnav.so",
    "/usr/lib/libspnav.so.0",
    "/usr/local/lib/libspnav.so.0",
    "/usr/lib/x86_64-linux-gnu/libspnav.so.0",
    "/usr/lib/aarch64-linux-gnu/libspnav.so.0",
]


def _find_libspnav(lib_path: str | None = None) -> str:
    """Resolve libspnav library path."""
    if lib_path is not None:
        return lib_path
    # Try ctypes.util first
    found = ctypes.util.find_library("spnav")
    if found:
        return found
    # Try common paths
    for path in _COMMON_LIBSPNAV_PATHS:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Cannot find libspnav. Install it with: sudo apt install libspnav-dev spacenavd"
    )


class SpaceMouseReader:
    """Non-blocking SpaceMouse reader running in a daemon thread."""

    def __init__(self, lib_path: str | None = None, deadzone: int = 40):
        resolved_path = _find_libspnav(lib_path)
        self._lib = ctypes.CDLL(resolved_path)
        self._deadzone = deadzone

        # shared state protected by lock
        self._lock = threading.Lock()
        self._axes = [0, 0, 0, 0, 0, 0]        # x, y, z, rx, ry, rz (raw, dead-zoned)
        self._buttons = {}                       # bnum -> pressed (bool)
        self._button_events = []                 # [(bnum, pressed), ...]

        self._stop_event = threading.Event()
        self._thread = None

    # ------ lifecycle ------

    def open(self):
        """Connect to spacenavd."""
        if self._lib.spnav_open() == -1:
            raise RuntimeError("Cannot connect to spacenavd. Is the daemon running?")
        buf = ctypes.create_string_buffer(256)
        self._lib.spnav_dev_name(buf, 256)
        name = buf.value.decode()
        axes = self._lib.spnav_dev_axes()
        buttons = self._lib.spnav_dev_buttons()
        logger.info(f"SpaceMouse connected: {name}  (axes={axes}, buttons={buttons})")

    def start(self):
        """Start the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop polling and disconnect."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._lib.spnav_close()
        logger.info("SpaceMouse disconnected.")

    # ------ public accessors (thread-safe) ------

    def get_axes(self):
        """Return latest [x, y, z, rx, ry, rz] with dead-zone applied."""
        with self._lock:
            return list(self._axes)

    def get_button(self, bnum: int) -> bool:
        """Return current pressed state of button *bnum*."""
        with self._lock:
            return self._buttons.get(bnum, False)

    def pop_button_events(self):
        """Pop and return all queued button events as [(bnum, pressed), ...]."""
        with self._lock:
            events = list(self._button_events)
            self._button_events.clear()
            return events

    # ------ internals ------

    def _apply_deadzone(self, value: int) -> int:
        if abs(value) < self._deadzone:
            return 0
        return value - self._deadzone if value > 0 else value + self._deadzone

    def _poll_loop(self):
        ev = SpnavEvent()
        while not self._stop_event.is_set():
            ret = self._lib.spnav_poll_event(ctypes.byref(ev))
            if ret == 0:
                # no event available
                time.sleep(0.001)
                continue

            if ev.type == SPNAV_EVENT_MOTION:
                m = ev.motion
                axes = [
                    self._apply_deadzone(m.x),
                    self._apply_deadzone(m.y),
                    self._apply_deadzone(m.z),
                    self._apply_deadzone(m.rx),
                    self._apply_deadzone(m.ry),
                    self._apply_deadzone(m.rz),
                ]
                with self._lock:
                    self._axes = axes

            elif ev.type == SPNAV_EVENT_BUTTON:
                b = ev.button
                pressed = bool(b.press)
                with self._lock:
                    self._buttons[b.bnum] = pressed
                    self._button_events.append((b.bnum, pressed))
