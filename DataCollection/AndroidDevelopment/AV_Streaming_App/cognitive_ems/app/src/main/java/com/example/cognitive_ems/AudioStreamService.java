package com.example.cognitive_ems;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.IBinder;
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
    private static final String TAG2 = "Audio Client";
    private static final int RECORDING_RATE = 16000; //16000;//11025;//44100;
    private static final int CHANNEL = AudioFormat.CHANNEL_CONFIGURATION_MONO; //CHANNEL_IN_MONO
    private static final int FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static int BUFFER_SIZE = AudioRecord.getMinBufferSize(RECORDING_RATE, CHANNEL, FORMAT);
    private String SERVER = "172.28.39.240"; //"xx.xx.xx.xx";
    int port = 50005;
    private final Context mContext;

    private SocketIOService socketIOService;

    private boolean mBound = false;

    public AudioStreamService(Context mContext, String server_ip, int port) {
        this.port = port;
        this.SERVER = server_ip;
        this.mContext = mContext;

        // Bind to LocalService.
        Intent intent = new Intent(this.mContext, SocketIOService.class);
        this.mContext.bindService(intent, connection, Context.BIND_AUTO_CREATE);

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
                    Log.d(TAG2, "Recorder initialized " + BUFFER_SIZE );

                    recorder.startRecording();



                    while(true) {
                        BUFFER_SIZE = recorder.read(buffer, 0, buffer.length);

                        if(socketIOService != null){


                            String command = socketIOService.getCommand();
                            Log.d("Audio Client", "C: Command "+ command );

                            audioStreamStatus = command.equals("start");

                            if (audioStreamStatus) {
//                                socketIOService.sendAudio(buffer);
                                Log.d(TAG2, "Sending Audio " + buffer.length + " bytes");

                                //putting buffer in the packet
                                packet = new DatagramPacket (buffer,buffer.length,destination,port);
                                socket.send(packet);

                            }
                        }

                        //reading data from MIC into buffer


                        //System.out.println("MinBufferSize: " +BUFFER_SIZE);
                    }

                } catch(Exception e) {
                    Log.e(TAG2, "Exception, audio streaming");
                    e.printStackTrace();
                }
            }
        });
        streamThread.start();
    }


    private final ServiceConnection connection = new ServiceConnection() {

        @Override
        public void onServiceConnected(ComponentName className,
                                       IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance.
            SocketIOService.SocketIOServiceBinder binder = (SocketIOService.SocketIOServiceBinder) service;
            socketIOService = binder.getService();
            Log.d("Audio Stream Client", "Bounded to service!");
            mBound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mBound = false;
        }
    };


}
