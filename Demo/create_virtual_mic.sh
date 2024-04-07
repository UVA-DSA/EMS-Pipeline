# set up virtual speaker
pactl load-module module-null-sink sink_name="virtual_speaker" sink_properties=device.description="virtual_speaker"

# set up virtual mic
pactl load-module module-remap-source master="virtual_speaker.monitor" source_name="virtual_mic" source_properties=device.description="virtual_mic"
