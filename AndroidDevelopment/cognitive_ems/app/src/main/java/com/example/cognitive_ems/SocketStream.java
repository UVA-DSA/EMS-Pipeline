package com.example.cognitive_ems;


import android.util.Log;

import java.net.URISyntaxException;


import io.socket.client.IO;
import io.socket.client.Socket;
import io.socket.emitter.Emitter;


public class SocketStream {

    private static Socket socket;
    private static final String command = "stop";
    //implement socket io connection
    private final String TAG = "SocketStream";
    private final int port = 9235;
    private String serverUrl ;

    private FeedbackCallback feedbackCallback;

    private static SocketStream instance;


    // Singleton pattern
    public static SocketStream getInstance() {
        if (instance == null) {
            instance = new SocketStream();
        }
        return instance;
    }

    private SocketStream() {
        // Private constructor to prevent instantiation
    }

    public void initialize(String serverUrl) {
        this.serverUrl = serverUrl;

        try {
            IO.Options options = IO.Options.builder()
                    .setReconnection(true)
                    .setReconnectionAttempts(20)
                    .setReconnectionDelay(1000)
                    .build();

            Log.d("SocketIO Client", "C: Connecting to " + serverUrl);
            socket = IO.socket(serverUrl, options);

        } catch (URISyntaxException e) {
            Log.d("SocketIO Client", "E: Error!");
            e.printStackTrace();
        }

        // Set up event listeners
        socket.on(Socket.EVENT_CONNECT, new Emitter.Listener() {
            @Override
            public void call(Object... args) {
                Log.d("SocketIO Client", "C: Connected!");
            }
        });

        socket.on("command", new Emitter.Listener() {
            @Override
            public void call(Object... args) {
                Log.d("SocketIO Client", "R: Received Message! : " + args[0]);
            }
        });

        socket.on("feedback", new Emitter.Listener() {
            @Override
            public void call(Object... args) {
                Log.d("Feedback Client", "R: Received Feedback! : " + args[0]);

                if (feedbackCallback != null) {
                    feedbackCallback.onFeedbackReceived(args[0].toString());
                }
            }
        });

        socket.on("action", new Emitter.Listener(){
            @Override
            public void call(Object... args) {
                Log.d("SocketIO Client", "R: Received Action! : " + args[0]);
            }
        });
        }

        // Add more event listeners here as needed

        socket.connect();
    }


    public void setFeedbackCallback(FeedbackCallback feedbackCallback) {
        this.feedbackCallback = feedbackCallback;
    }

    public String getCommand() {
        return command;
    }

    public void sendMessage(String message) {
        // Emit a message to the server
        socket.emit("message", message);
        Log.d("SocketIO Client", "C: Sent Message!");

    }

    public void sendVideo(String bytes) {
        // Emit a message to the server
        socket.emit("video", bytes);
        Log.d(TAG, "C: Sent Video!");
    }

    public void sendBytes(byte[] bytes) {
        // Emit a message to the server
        socket.emit("bytes", bytes);
        Log.d(TAG, "C: Sent Bytes!");
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
