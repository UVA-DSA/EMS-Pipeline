package com.example.cognitive_ems;

import android.util.Log;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.Socket;

public class TcpClient {

    private String serverIP = "192.168.1.1"; // Replace with your server IP
    private int serverPort = 12345; // Replace with your server port
    private Socket socket;
    private BufferedWriter writer;
    private BufferedReader reader;

    private String TAG = "TcpClient";

    // Listener interface for receiving messages
    public interface TcpClientListener {
        void onMessageReceived(String message);
    }

    private TcpClientListener listener;

    public TcpClient(String ip, int port, TcpClientListener listener) {
        this.serverIP = ip;
        this.serverPort = port;
        this.listener = listener;
    }

    // Connect to the server
    public void connect() {

        new Thread(() -> {
            try {
                socket = new Socket(serverIP, serverPort);
                writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream()));
                reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }).start();

    }

    // Disconnect from the server
    public void disconnect() {
        try {
            if (socket != null) {
                socket.close();
            }
            if (writer != null) {
                writer.close();
            }
            if (reader != null) {
                reader.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // Send a message to the server
    public void sendMessage(final String message) {
        new Thread(() -> {
            try {
                if (socket != null && writer != null) {
                    writer.write(message);
                    writer.newLine();
                    writer.flush();
                    Log.d(TAG, "Sent message: " + message);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }).start();
    }
}
