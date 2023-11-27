package com.example.camstrm;

import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;

public class SocketIOService extends Service {

    private SocketIOAdapter socketIOAdapter;
    private  String SERVER_IP; //server IP address
    private  String SERVER_PORT;

    private final IBinder binder = new SocketIOServiceBinder();


    public SocketIOService() {

    }

    @Override
    public void onCreate() {
        super.onCreate();

    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if(intent != null){

        Bundle extras = intent.getExtras();
        if (extras == null) {
            Log.d("Service", "null");
            Log.d("SocketIO Service","null");

        }
        else {
            Log.d("SocketIO Service","not null");
            this.SERVER_IP= (String) extras.get("server_ip");
            this.SERVER_PORT= (String) extras.get("server_port");

            }

        String serverURL = "http://"+SERVER_IP+ ":" + SERVER_PORT;
        socketIOAdapter = new SocketIOAdapter(serverURL);
        socketIOAdapter.sendMessage("Hello from Service!");

        }

        return super.onStartCommand(intent, flags, startId);
    }

    @Override
    public IBinder onBind(Intent intent) {
        // TODO: Return the communication channel to the service.
        return binder;
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

    public void sendMessage(String message){
        Log.d("SocketIO Service", "Sending message");
        socketIOAdapter.sendMessage(message);
    }
    public void sendVideo(String video){
        socketIOAdapter.sendVideo(video);
    }

    public void sendAudio(byte[] audio){
        socketIOAdapter.sendAudio(audio);
    }
    public String getCommand(){
//        Log.d("SocketIO Service", "Getting message");
        return socketIOAdapter.getCommand();
    }
}