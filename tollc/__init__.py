import os
import time
import threading
import contextlib

import cv2

__all__ = ["cap_context"]

def create_cap(width: int, height: int, low_latency=True) -> cv2.VideoCapture:
    """
    Create a cv2.VideoCapture object for Jetson Nano CSI camera

    Create a cap like this:

    >>> cap = create_cap(640, 480)

    And when done, make sure to release it like this:

    >>> cap.release()

    `low_latency` can be turned off if quality is more important than latency.
    Temporal noise reduction will be turned on as well as high quality scaling.
    Neither should have an effect on CPU usage, but ISP and memory use will
    likely be higher.

    >>> cap = create_cap(640, 480, low_latency=False)
    >>> cap.release()
    """
    return cv2.VideoCapture(
        f'nvarguscamerasrc sensor-mode=0 tnr-mode={"0" if low_latency else "2"} '
        f'! nvvidconv interpolation-method={"Nearest" if low_latency else "Nicest"} output-buffers=1 '
        f"! video/x-raw, width={width}, height={height}, format=BGRx "
        # this is unavoidable, unfortunately, because OpenCV does not accept BGRx
        # and nvvidconv does not support BGR (really, BGRx is better cause of
        # alignment issues)
        f"! videoconvert n-threads={os.cpu_count()} "
        f"! video/x-raw, format=(string)BGR "
        f"! appsink"
    )


def clear_buffer(cap: cv2.VideoCapture):
    # releasing cap will terminate this thread
    while cap.isOpened():
        # throw away any internal buffer. We don't really care if it succeeds or
        # not, and it might always not due to race conditions on cleanup.
        cap.grab()
        # this is the same as std::this_thread::yield, so yield to the (main?)
        # thread. The OS will do this anyway given some time, but this will
        # force a context switch.
        time.sleep(0)


@contextlib.contextmanager
def cap_context(width: int, height: int, low_latency=True):
    """
    A cv2.VideoCapture context manager for Jetson Nano CSI cameras. It ensures
    release is called at the end of context and also that any internal buffer
    is cleared, so you always get an up-to-date frame.

    As a consequence, doing a `cap.read()` will block until the next frame is
    ready. If this isn't desired and you want the usual cv2.VideoCapture
    behavior, call `cap_context` with `low_latency=False`

    Example usage/test:

    Within the `with` block, cap will be opened.

    >>> with cap_context(640, 480) as cap:
    ...     ok, img = cap.read()
    ...     cap.isOpened()
    ...
    True

    At the end, `cap.release()` will be called and any related threads joined.

    >>> cap.isOpened()
    False

    The cap returned is just a cv2.VideoCapture() object and should behave
    the same way. It captures BGRx frames from the CSI camera.

    >>> type(img)
    <class 'numpy.ndarray'>
    >>> img.shape[0]
    480
    >>> img.shape[1]
    640
    >>> img.shape[2]
    3
    >>> img.dtype
    dtype('uint8')

    Stress test (to see if the craptastic argus daemon will crash):
    >>> for i in range(10):
    ...     with cap_context(640, 480, low_latency=i % 2 == 0) as cap:
    ...         ok, img = cap.read()
    ...         assert ok
    ...         assert img is not None
    ...         assert img.shape[0] == 480
    ...         assert img.shape[1] == 640
    ...         assert img.shape[2] == 3
    ...         assert img.dtype.name == 'uint8'
    ...     assert not cap.isOpened()
    """
    # create cap
    cap = create_cap(width, height, low_latency=low_latency)
    # continually exhaust the internal buffer in the background
    if low_latency:
        clear_thread = threading.Thread(target=clear_buffer, args=(cap,))
        clear_thread.start()
    try:
        yield cap
    finally:
        # release cap, which will signal `clear_thread` to stop
        cap.release()
        if low_latency:
            # wait until clear_thread exits
            clear_thread.join()
