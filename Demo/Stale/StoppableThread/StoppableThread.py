import threading
"""
For anyone working with this. Note that this needs to be reimplemented because _stop
is a protected keyword in the threading.thread class. Because of that this function will throw
the following error.

Traceback (most recent call last):
  File "/path/to/threading.py", line 1457 (could be different on another version), in _after_fork
    thread._stop()
TypeError: 'Event' object is not callable

The easy fix for this is to implement another variable for the SIGSTOP. An example
can be seen in Feedback.py.
"""

# ============== Custom Thread Class with a Stop Flag ==============
class StoppableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet() # this is appararently a deprecated alias for below method. TODO: get rid of it.
        return self._stop_event.is_set()