# camstrm
Stream camera feed of Android device to computer via ADB

## what does this do 
this repo contain and Android app and a python3 script that runs on a computer.
Android app access the camera of the Android device and sends image frames over TCP connection over ADB.
pytho3 script access these images from the TCP port and display them and stores it as a video at the same time.

## how to use
1. git clone https://github.com/sleekEagle/camstrm.git
2. turn on the Android device
3. establish connection over ADB (USB or WiFi)
4. make sure the TCP port number on both the Android app and the python3 script are the same.
assuming it is 9600,
5. perform port forwarding with the command 
```
adb forward tcp:9600 tcp:9600
```
6. start the Android app first from the device
7. goto the project dir
8. make sure the path is correct (to store the video) 
9. execute the python script
```
python3 write_vid.py
```

## tips
you can use scrcpy program from https://github.com/Genymobile/scrcpy
to mirror screen of an Android device to the computer, so you do not have to look at the 
device screen while you are working. 
 
## Running apps via ADB
start the adb shell by 
```
adb shell
```

start the app by 
```
am start -n com.example.camstrm/com.example.camstrm.MainActivity
``` 

kill the app by 
```
am force-stop com.example.camstrm
```


## Running ADB via Wi-Fi instead of through USB cable
on the computer type:
```
adb tcpip 5555
adb shell ip addr show wlan0
```
and copy the IP address after the "inet" until the "/". 

on the computer type:
```
adb connect ip-address-of-device:5555
```

You can disconnect the USB cable now
use 
```
adb devices
```
to check if the device is still attached. 


