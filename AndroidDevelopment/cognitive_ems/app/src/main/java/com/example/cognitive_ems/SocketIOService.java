package com.example.cognitive_ems;


import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.os.Binder;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import android.view.TextureView;

import java.io.ByteArrayOutputStream;

import io.socket.engineio.parser.Base64;

public class SocketIOService extends Service {

    private final IBinder binder = new SocketIOServiceBinder();
    private SocketStream socketIOAdapter;
    private String serverUrl; //server IP address

    private ImageCompressor imageSender;
    private String TAG = "SocketIOService";

    public SocketIOService() {

    }

    @Override
    public void onCreate() {
        super.onCreate();


        // To retrieve the server IP address
        SharedPreferences preferences = getSharedPreferences("CognitiveEMSConfig", Context.MODE_PRIVATE);
        String serverIp = preferences.getString("socketio_uri", "");

        Log.d(TAG, "Got SocketIO Keshara config: "+ serverIp);
        this.serverUrl = serverIp;


        socketIOAdapter = new SocketStream(serverIp);
        socketIOAdapter.sendMessage("Hello from Service!");

        imageSender = ImageCompressor.getInstance( socketIOAdapter);



    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {



        if (intent != null) {

            Bundle extras = intent.getExtras();
            if (extras == null) {
                Log.d("Service", "null");
                Log.d("SocketIO Service", "null");

            } else {
                Log.d("SocketIO Service", "not null");
                this.serverUrl = (String) extras.get("socketio_uri");

            }

            socketIOAdapter = new SocketStream(this.serverUrl);
            socketIOAdapter.sendMessage("Hello from Service!");

        }

        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        return binder;
    }

    public void sendMessage(String message) {
        Log.d("SocketIO Service", "Sending message");
        socketIOAdapter.sendMessage(message);
    }

    public void sendImage(Bitmap bitmap) {

        // generate bitmap and byte array stream for the compress
        try {


// Create an instance of ImageSender and execute it
            if(imageSender == null) {
                Log.e("SocketIO Service", "ImageSender is null");
                return;
            }
            imageSender.compressAndSend(bitmap);



        } catch (Exception e) {
            Log.e("SocketIO Service", "Error while sending image");
            e.printStackTrace();
        }
    }

    public void sendAudio(byte[] audio) {
        socketIOAdapter.sendAudio(audio);
    }

    public String getCommand() {
//        Log.d("SocketIO Service", "Getting message");
        return socketIOAdapter.getCommand();
    }

    /**
     * Class used for the client Binder.  Because we know this service always
     * runs in the same process as its clients, we don't need to deal with IPC.
     */
    public class SocketIOServiceBinder extends Binder {
        SocketIOService getService() {
            // Return this instance of LocalService so clients can call public methods.
            return SocketIOService.this;
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
    }
}