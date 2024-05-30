package com.example.cognitive_ems;

import android.util.Log;

import io.socket.client.IO;
import io.socket.client.Socket;
import io.socket.emitter.Emitter;

import java.net.URISyntaxException;

public class SocketIOAdapter {
    private static Socket socket;
    private static String command = "stop";

    public SocketIOAdapter(String serverUrl) {
        try {
            IO.Options options = IO.Options.builder()
                    .setReconnection(true)
                    .setReconnectionAttempts(20)
                    .setReconnectionDelay(1000)
                    .build();

            Log.d("SocketIO Client", "C: Connecting to " +serverUrl);
            socket = IO.socket(serverUrl, options);

        } catch (URISyntaxException e) {
            Log.d("SocketIO Client", "E: Error!");
            e.printStackTrace();
        }

        // Set up event listeners
        socket.on(Socket.EVENT_CONNECT, new Emitter.Listener() {
            @Override
            public void call(Object... args) {
                // Handle the connection event
                Log.d("SocketIO Client", "C: Connected!");
            }
        });

        socket.on("command", new Emitter.Listener() {
            @Override
            public void call(Object... args) {
                Log.d("SocketIO Client", "R: Received Message! : " + args[0]);

                command = (String) args[0];

            }
        });
        // Add more event listeners here as needed
        socket.connect();
    }

    public String getCommand(){
        return command;
    }
    public void sendMessage(String message) {
        // Emit a message to the server
        socket.emit("message", message);
        Log.d("SocketIO","C: Sent Message!");

    }

    public void sendVideo(String bytes) {
        // Emit a message to the server
        socket.emit("video", bytes);
    }

    public void sendAudio(byte[] bytes) {
        // Emit a message to the server
        socket.emit("audio", bytes);
    }


    public void disconnect() {
        if (socket != null && socket.connected()) {
            socket.disconnect();
        }
    }


}
