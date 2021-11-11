# `tollc` low latency OpenCV CSI camera context for Jetson boards

Provides a simple context manager for opening the CSI camera and using GStreamer
and OpenCV to capture frames. It:

* ensures `cap.release()` is called at the end of context
* provides a (default) low-latency mode so cap.read() always returns the latest
  frame possible.

Usage is simply:

```python3
from tollc import cap_context

with cap_context(640, 480) as cap:
    ok, img = cap.read()

    if not ok:
        # handle any error here
        pass

    # do stuff with img here
# cap is released here, and any resources freed
```

# FAQ

* **`cap.read()` is blocking now**. In `low_latency` mode, `cap.read()` needs to
  wait for a new frame before it can return one to you. This is unavoidable
  without breaking the laws of physics if you want the newest frame *when it's
  ready* (and not just some cached frame taken some time ago).

* **`low_latency`** mode still has some latency. This is unavoidable without
  customizing the camera source, the closed source Nvidia `Argus` camera library
  and probably a lot of other things. Also, conversion and copy to CPU is
  expensive. There *may* be room for improvement in the GStreamer pipeline.

* **full sized frames are slow**. It's OpenCV. What were you expecting? Want
  speed? Use Nvidia's solutions instead. This is suitable for developing and
  testing CV algorithms, but probably not for production if you need your stuff
  to go fast.
