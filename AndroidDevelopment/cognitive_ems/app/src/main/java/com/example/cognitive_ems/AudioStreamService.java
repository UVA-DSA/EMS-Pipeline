package com.example.cognitive_ems;


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
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class AudioStreamService {

    private static final int RECORDING_RATE = 16000; //16000;//11025;//44100;
    private static final int CHANNEL = AudioFormat.CHANNEL_CONFIGURATION_MONO; //CHANNEL_IN_MONO
    private static final int FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final String TAG2 = "Audio recording";
    private static int BUFFER_SIZE = AudioRecord.getMinBufferSize(RECORDING_RATE, CHANNEL, FORMAT);
    int port = 50005;
    //audio streaming vars
    private AudioRecord recorder;
    private boolean audioStreamStatus = true;
    private String SERVER;
    private final Context mContext;
    PayloadType payloadType;
    private AtomicInteger dataBytes = new AtomicInteger(0);
    private AtomicLong dataBytesResetTime = new AtomicLong(0);
    private static final int PAYLOAD_SIZE = 1000;

    public PayloadType getPayloadType() { return payloadType;}

    public AudioStreamService(Context mContext, String server_ip, int port) {
        this.port = port;
        this.SERVER = server_ip;
        this.mContext = mContext;
        this.payloadType = PayloadType.RAW_16BIT;
    }

    public void startStreaming() {
        Thread streamThread = new Thread(new Runnable() {

            @Override
            public void run() {
                try {

                    DatagramSocket socket = new DatagramSocket();
                    Log.d(TAG2, "Socket Created for Audio Stream");

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
                    // Initialize the recorder
                    int format = payloadType.getAudioFormat();
                    int bufferSize = AudioRecord.getMinBufferSize(RECORDING_RATE, CHANNEL, format);
                    if (bufferSize == AudioRecord.ERROR_BAD_VALUE)
                        return;
                    bufferSize = (1+ bufferSize/PAYLOAD_SIZE)*PAYLOAD_SIZE;

                    recorder = new AudioRecord(MediaRecorder.AudioSource.MIC, RECORDING_RATE, CHANNEL, format, bufferSize * 10);
                    Log.d(TAG2, "Recorder initialized");

                    // Initialize the UDP stream
                    byte[] buffer = new byte[bufferSize];
                    DatagramPacket datagramPacket = new DatagramPacket(buffer, buffer.length);
                    datagramPacket.setAddress(destination);
                    datagramPacket.setPort(port);

                    Log.d(TAG2, "Socket and packet initialized: "+ datagramPacket.getAddress() + " " + datagramPacket.getPort());


                    recorder.startRecording();


                    short frameNb = 0;
                    int sampleNb = 0;
                    dataBytes.set(0);
                    dataBytesResetTime.set(System.currentTimeMillis());


                    while (audioStreamStatus && (recorder.getState() == AudioRecord.STATE_INITIALIZED)
                            && (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING)) {
                        //reading data from MIC into buffer
                        try {
                            int sizeToSend = recorder.read(buffer, 0, buffer.length);
                            int index = 0;
                            while(sizeToSend>0) {
                                int packetBufferSize = Math.min(sizeToSend, PAYLOAD_SIZE);
                                byte[] bufferToSend = Arrays.copyOfRange(buffer, index, index + packetBufferSize);
                                StreamPacket rtp_packet = new StreamPacket((byte) payloadType.payloadTypeId,
                                        frameNb++, sampleNb, RECORDING_RATE,
                                        bufferToSend,
                                        bufferToSend.length);
                                byte[] packetBuffer = new byte[rtp_packet.getPacketLength()];
                                rtp_packet.getPacket(packetBuffer);
                                sizeToSend -= packetBufferSize;
                                index += packetBufferSize;
                                sampleNb += packetBufferSize/payloadType.sampleByteSize;
                                datagramPacket.setData(packetBuffer);
                                if(!socket.isClosed()) {
                                    socket.send(datagramPacket);
                                    dataBytes.set(dataBytes.get() + packetBuffer.length);
                                    Log.d(TAG2, "Sent packet of size " + packetBuffer.length);
                                }
                            }
                        } catch (Throwable t) {
                            Log.e(TAG2, t.toString());// gérer l'exception et arrêter le traitement
                        }
                    }
                    // Stop and close everything
                    if (!socket.isClosed())
                        socket.close();
                    if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING)
                        recorder.stop();
                    if (recorder.getState() == AudioRecord.STATE_INITIALIZED)
                        recorder.release();
                    Log.d(TAG2, "Recorder released");
                } catch (UnknownHostException e) {
                    Log.e(TAG2, "UnknownHostException, audio streaming");
                } catch (IOException e) {
                    e.printStackTrace();
                    Log.e(TAG2, "IOException, audio streaming");
                }
            }
        });
        streamThread.start();
    }

    public void stopStreaming() {
        audioStreamStatus = false;
    }


}
