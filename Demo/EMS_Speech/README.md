# EMS Speech Setup

`EMS_Whisper/whisper.cpp_realtime_stream` is tracked in this repo as a Git submodule.

## Important

A normal parent-repo clone does **not** automatically download submodule contents.

If someone runs:

```bash
git clone <parent-repo-url>
```

the `whisper.cpp_realtime_stream` folder will exist only as a submodule entry until they initialize it.

## Recommended Clone

Clone the parent repo with submodules:

```bash
git clone --recurse-submodules <parent-repo-url>
```

## If The Repo Was Already Cloned

Run:

```bash
git submodule update --init --recursive
```

If the submodule URL ever changes, run:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## EMS Whisper Submodule Path

The Whisper realtime stream submodule lives at:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream
```

## Building The Whisper Realtime Binary

The Python manager expects the realtime binary at:

```text
Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream/build/bin/egosim_stream
```

If it is missing, build it from inside the submodule:

```bash
cd Demo/EMS_Speech/EMS_Whisper/whisper.cpp_realtime_stream
cmake -B build -DWHISPER_SDL2=ON
cmake --build build --config Release
```

## Notes

- The current submodule URL is SSH-based, so GitHub SSH access is required unless the URL is changed to HTTPS.
- After pulling parent-repo changes, submodule updates may also need:

```bash
git submodule update --init --recursive
```
