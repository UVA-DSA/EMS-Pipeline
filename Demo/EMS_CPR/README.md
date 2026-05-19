# EMS CPR Depth Sensor

This module contains the Arduino firmware and Python interface for reading CPR
compression depth from a VL6180 distance sensor embedded in a manikin.

The code is intentionally isolated from the GUI for now. It can record raw CPR
sensor readings to CSV and publish the same normalized samples to named pipes so
future GUI/process integrations can consume CPR data the same way the demo
already consumes video, audio, and IMU streams.

## Layout

```text
Demo/EMS_CPR/
├── __init__.py
├── cpr_depth_reader.py
├── cpr_data_visualizer.py
└── arduino/
    └── vl6180_cpr_depth/
        └── vl6180_cpr_depth.ino
```

## Serial Input

The Arduino sketch prints startup/status messages, then emits one reading about
every 20 ms.

Successful range sample:

```text
Range:<range_mm>:Seq:<sequence>
```

Example:

```text
Range:87:Seq:42
```

Sensor error/status samples include a sequence number but no valid range:

```text
No convergence:Seq:43
```

The Python reader normalizes error/status samples with `range_mm = -1`.

## Normalized Data Contract

CSV files and CPR named pipes use newline-terminated UTF-8 CSV text:

```text
timestamp_ns,range_mm,sequence
```

The CSV recording file includes this header:

```text
Timestamp (ns),Range (mm),Sequence
```

The named pipes do not include a header, matching the existing demo stream
pattern where pipe consumers receive only live data rows.

## Named Pipes

The CPR reader defines two pipe paths:

```text
/tmp/emscpr
/tmp/emscprml
```

Use `/tmp/emscpr` for a future GUI/display consumer and `/tmp/emscprml` for a
future processing or ML consumer. Both pipes carry the same normalized CSV line
format.

Unlike the existing SRT receiver, CPR pipe setup currently lives inside this
module so the GUI code does not need to change yet.

## Python Usage

Install the CPR reader and visualizer dependencies in the demo environment if
needed:

```bash
pip install pyserial PyQt5 pyqtgraph
```

`pyserial` is required for the Arduino reader. `PyQt5` and `pyqtgraph` are
required for `cpr_data_visualizer.py`. The main demo environment already lists
PyQt5 and pyqtgraph in the root installation instructions, so this command is
mainly for running the CPR module by itself.

Read from the Arduino, publish CPR samples to `/tmp/emscpr`, and save a CSV
recording:

```python
import time
from multiprocessing import Event

from Demo.EMS_CPR.cpr_depth_reader import ArduinoVL6180Recorder, PIPE_CPR

thread_stop = Event()
recorder = ArduinoVL6180Recorder(
    port="/dev/ttyACM0",
    recording_dir="./recordings",
    thread_stop=thread_stop,
    pipe_paths=(PIPE_CPR,),
    create_pipes=True,
    save_enabled=True,
)

recorder.start_recording()
recorder.start()

try:
    time.sleep(10)
finally:
    recorder.stop_recording()
    recorder.stop()
```

On some systems the Arduino may appear as `/dev/ttyUSB0`. On Windows it may
appear as `COM5` or another `COM` port.

To read/publish live CPR samples without saving a CSV file, set
`save_enabled=False`:

```python
recorder = ArduinoVL6180Recorder(
    port="/dev/ttyACM0",
    recording_dir="./recordings",
    thread_stop=thread_stop,
    pipe_paths=(PIPE_CPR,),
    create_pipes=True,
    save_enabled=False,
)
```

`start_recording()` still starts the live serial read/publish loop. The
`save_enabled` flag only controls whether rows are written to disk.

CSV saving can also be toggled while the recorder process is running:

```python
recorder.set_save_enabled(False)  # keep reading/publishing, stop saving CSV rows
recorder.set_save_enabled(True)   # resume saving rows to the same session CSV
```

## CPR Data Visualizer

`cpr_data_visualizer.py` is a standalone PyQt5/pyqtgraph GUI that reads CPR
samples from `/tmp/emscpr`, plots compression depth over time, and displays:

- estimated CPR rate in compressions/min
- current compression depth in mm
- latest raw sensor range in mm

The compression depth calculation uses this global baseline in
`cpr_data_visualizer.py`:

```python
NORMAL_DISTANCE_BASELINE_MM = 100.0
```

Set this to the normal resting sensor distance when no CPR compression is being
performed. Depth is calculated as:

```text
depth_mm = max(0, NORMAL_DISTANCE_BASELINE_MM - range_mm)
```

You can also override the baseline from the command line:

```bash
python -m Demo.EMS_CPR.cpr_data_visualizer --baseline-mm 92.5
```

Run the visualizer and the Arduino reader in separate terminals. The reader
publishes to `/tmp/emscpr`; the visualizer consumes that same FIFO.

Example reader process:

```python
from multiprocessing import Event

from Demo.EMS_CPR.cpr_depth_reader import ArduinoVL6180Recorder, PIPE_CPR

thread_stop = Event()
recorder = ArduinoVL6180Recorder(
    port="/dev/ttyACM0",
    recording_dir="./recordings",
    thread_stop=thread_stop,
    pipe_paths=(PIPE_CPR,),
    create_pipes=True,
    save_enabled=False,
)
recorder.start_recording()
recorder.start()
```

## Future GUI Integration

When CPR depth is added to the GUI, the reader side should mirror the existing
IMU pattern:

- open `/tmp/emscpr` as a text FIFO
- read one newline-terminated CSV row at a time
- parse `timestamp_ns,range_mm,sequence`
- calculate compression depth from the calibrated resting baseline
- display or process the resulting depth samples

No GUI files are changed by this module yet.
