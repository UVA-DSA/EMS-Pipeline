package com.example.gesturerecognition;

import static android.content.Context.NOTIFICATION_SERVICE;

import android.annotation.SuppressLint;
import android.app.NotificationManager;
import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Enumeration;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class SendSensorDataWorker extends Worker {

    private static final String DEBUG_TAG = "Worker";
    private String serverIPAddress = "172.27.159.100";
    private int port = 7889;
    public boolean isStopped = false;
    public final BlockingQueue<String> queue = new LinkedBlockingQueue<>();
    private DatagramSocket udpSocket = null;
    private InetAddress serverAddr;
    private static final String LOG_TAG = "SensorDataWorker";
    private NotificationManager notificationManager;

    public SendSensorDataWorker(
            @NonNull Context context,
            @NonNull WorkerParameters params) {
        super(context, params);
        notificationManager = (NotificationManager)
                context.getSystemService(NOTIFICATION_SERVICE);
    }

    @SuppressLint("RestrictedApi")
    @NonNull
    @Override
    public Result doWork() {
        initializeUDPClient();
        SendDataToSocket();
        return Result.success();
    }

    public void initializeUDPClient() {
        while (udpSocket == null || udpSocket.isClosed()) {
            try {
                udpSocket = new DatagramSocket();
                serverAddr = InetAddress.getByName(serverIPAddress);
                Log.d(LOG_TAG, "Worker Initiated! UDP Server Address: " + serverAddr);
            } catch (SocketException e) {
                e.printStackTrace();
            } catch (UnknownHostException e) {
                e.printStackTrace();
            }

            if (udpSocket == null || udpSocket.isClosed()) {
                try {
                    Log.d(LOG_TAG, "Failed to connect, retrying...");
                    Thread.sleep(3000);  // Retry after 3 seconds
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    private byte[] SensorData() {
        try {
            String data = queue.take();
            return data.getBytes();
        } catch (InterruptedException e) {
            return null;
        }
    }

    private void SendDataToSocket() {
        while (!isStopped) {
            try {
                byte[] data = SensorData();
                if (data != null && udpSocket != null && !udpSocket.isClosed()) {
                    DatagramPacket packet = new DatagramPacket(data, data.length, serverAddr, port);
                    udpSocket.send(packet);
                    Log.d(LOG_TAG, "Sent Data to: " + serverAddr);
                }
            } catch (IOException e) {
                e.printStackTrace();
                udpSocket.close();
                initializeUDPClient();  // Reconnect if there's an exception
            }
        }
        if (udpSocket != null && !udpSocket.isClosed()) {
            udpSocket.close();
        }
    }

    @Override
    public void onStopped() {
        super.onStopped();
        isStopped = true;
        if (udpSocket != null && !udpSocket.isClosed()) {
            udpSocket.close();
        }
        Log.d(LOG_TAG, "Worker Cancelled!");
    }
}
