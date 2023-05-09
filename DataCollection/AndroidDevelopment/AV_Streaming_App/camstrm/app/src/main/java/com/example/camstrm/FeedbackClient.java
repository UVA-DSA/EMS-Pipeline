package com.example.camstrm;


import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.InetAddress;
import java.net.Socket;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;

public class FeedbackClient {

    private OnMessageReceived mMessageListener = null;
    private BufferedInputStream input;
    private byte[] serverMessage; // message received from server

    private boolean mRun = false;

    /**
     *  Constructor:
     *  OnMessagedReceived listens for the messages received from server.
     */
    public FeedbackClient(OnMessageReceived listener) {
        mMessageListener = listener;
    }

    /**
     * Connection thread
     */
    public void run() {
        mRun = true;

        try {
            // Server address:
            InetAddress serverAddr = InetAddress.getByName("172.27.142.69");

            Log.e("feedback TCP Client", "C: Connecting...");

            // Connection socket:
            Socket socket = new Socket(serverAddr, 7088);
            Log.d("feedback tcp Client", "C: Connected!");

            try {
                // Input stream from server:
                input = new BufferedInputStream(socket.getInputStream());

                while(true) {

                    String feedbackOuptut = "";
                    int bytesRead = input.read();
                    if(bytesRead == -1){
                        Log.d("feedback", "-1 found " + (char) bytesRead);
                        continue;

                    }
//                    Log.d("feedback", "Feedback Data received in Client: " + (char) bytesRead);
                    feedbackOuptut = feedbackOuptut + ((char) bytesRead);

                    while (bytesRead != 0) {
                        bytesRead = input.read();
                        if (bytesRead != 0) {
                            feedbackOuptut = feedbackOuptut + ((char) bytesRead);
                            Log.d("feedback", "Feedback Data received in Client: " + bytesRead);
                        }

                    }
                    Log.d("feedback", "Feedback string " + feedbackOuptut);
                    serverMessage = feedbackOuptut.getBytes(StandardCharsets.UTF_8);
                    if (serverMessage != null && mMessageListener != null) {
                        mMessageListener.messageReceived(serverMessage);
                    }

                    serverMessage = null; // reset serverMessage
                }

            } catch (Exception e) {
                Log.e("TCP feedback", "S: Error", e);
            } finally {
                socket.close();
            }
        } catch (Exception e) {
            Log.e("TCP feedback", "C: Error", e);
        }
    }

    public void stopClient(){
        mRun = false;
    }

    /**
     * The method messageReceived(byte[] message) must be implemented in the
     * MainActivity class at on asynckTask doInBackground
     */
    public interface OnMessageReceived {
        public void messageReceived(byte[] message);
    }

}