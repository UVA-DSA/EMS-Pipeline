package com.example.camstrm;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.util.Log;

import androidx.core.app.ActivityCompat;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.UnknownHostException;

public class AudioStreamService {

    //audio streaming vars
    private AudioRecord recorder;
    private boolean audioStreamStatus = true;
    private static String TAG2 = "Audio recording";
    private static final int RECORDING_RATE = 16000; //16000;//11025;//44100;
    private static final int CHANNEL = AudioFormat.CHANNEL_CONFIGURATION_MONO; //CHANNEL_IN_MONO
    private static final int FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static int BUFFER_SIZE = AudioRecord.getMinBufferSize(RECORDING_RATE, CHANNEL, FORMAT);
    private String SERVER = "172.28.39.240"; //"xx.xx.xx.xx";
    int port = 50005;
    private Context mContext;

    public AudioStreamService(Context mContext, String server_ip, int port) {
        this.port = port;
        this.SERVER = server_ip;
        this.mContext = mContext;
    }

    public void startStreaming() {
        Thread streamThread = new Thread(new Runnable() {

            @Override
            public void run() {
                try {

                    DatagramSocket socket = new DatagramSocket();
                    Log.d(TAG2, "Socket Created for Audio Stream");

                    byte[] buffer = new byte[BUFFER_SIZE];

                    Log.d(TAG2, "Buffer for audio created of size " + BUFFER_SIZE);
                    DatagramPacket packet;

                    final InetAddress destination = InetAddress.getByName(SERVER);
                    Log.d(TAG2, "Address for audio receiver retrieved");

                    if (ActivityCompat.checkSelfPermission(mContext, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                        // TODO: Consider calling
                        //    ActivityCompat#requestPermissions
                        // here to request the missing permissions, and then overriding
                        //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
                        //                                          int[] grantResults)
                        // to handle the case where the user grants the permission. See the documentation
                        // for ActivityCompat#requestPermissions for more details.
                        return;
                    }
                    recorder = new AudioRecord(MediaRecorder.AudioSource.MIC, RECORDING_RATE, CHANNEL, FORMAT, BUFFER_SIZE * 10);
                    Log.d(TAG2, "Recorder initialized");

                    recorder.startRecording();

                    while(audioStreamStatus == true) {
                        //reading data from MIC into buffer
                        BUFFER_SIZE = recorder.read(buffer, 0, buffer.length);

                        //putting buffer in the packet
                        packet = new DatagramPacket (buffer,buffer.length,destination,port);

                        socket.send(packet);
                        //System.out.println("MinBufferSize: " +BUFFER_SIZE);
                    }

                } catch(UnknownHostException e) {
                    Log.e(TAG2, "UnknownHostException, audio streaming");
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.e(TAG2, "IOException, audio streaming");
                }
            }
        });
        streamThread.start();
    }


}
