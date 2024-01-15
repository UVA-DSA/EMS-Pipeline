package com.example.gesturerecognition;

import static android.content.Context.NOTIFICATION_SERVICE;

import android.annotation.SuppressLint;
import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.os.Build;
import android.provider.Settings;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.core.app.NotificationCompat;
import androidx.work.ForegroundInfo;
import androidx.work.ListenableWorker;
import androidx.work.WorkManager;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Enumeration;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class SendSensorDataWorker extends Worker {

    private static final String DEBUG_TAG = "Worker";
    private String serverIPAddress = "172.27.159.100";
    private String message = "Hello Android!" ;
    public boolean isStopped = false;
    private int port = 7889;
    public final BlockingQueue<String> queue = new LinkedBlockingQueue<String>();
    public static DatagramSocket udpSocket = null;
    private InetAddress serverAddr ;
    protected static final String LOG_TAG = "SensorDataWorker";
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

        // Do the work here--in this case, upload the images.
//        InitializeUDPClient();
        initializeTCPServer();

        setRunInForeground(true);

//        SendDataToSocket();

        return Result.success();

    }

    public static String getLocalIpAddress() {
        try {
            Enumeration<NetworkInterface> interfaces = NetworkInterface.getNetworkInterfaces();
            while (interfaces.hasMoreElements()) {
                NetworkInterface iface = interfaces.nextElement();
                // filters out 127.0.0.1 and inactive interfaces
                if (iface.isLoopback() || !iface.isUp())
                    continue;

                Enumeration<InetAddress> addresses = iface.getInetAddresses();
                while (addresses.hasMoreElements()) {
                    InetAddress addr = addresses.nextElement();
                    String ip = addr.getHostAddress();
                    // Check if the IP address is in the IPv4 format
                    if (ip.matches("\\d+(\\.\\d+){3}")) {
                        return ip;
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public void initializeTCPServer() {
        try {
            // Create a ServerSocket to listen for incoming connections on a specific port
            ServerSocket serverSocket = new ServerSocket(port);
            Log.d(LOG_TAG, " TCP Server Listening!: " + getLocalIpAddress());

            while (true) {
                // Accept a new client connection
                Socket clientSocket = serverSocket.accept();

                // Handle the client communication in a separate thread
                Thread clientThread = new Thread(new ClientHandler(clientSocket));
                clientThread.start();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }



    // Create a separate class to handle client communication in a thread
    private class ClientHandler implements Runnable {
        private Socket clientSocket;


        public ClientHandler(Socket clientSocket) {
            this.clientSocket = clientSocket;
        }

        @Override
        public void run() {
            try {
                // Get the input and output streams for the client socket
                InputStream inputStream = clientSocket.getInputStream();
                OutputStream outputStream = clientSocket.getOutputStream();

                // Now you can read and write data using inputStream and outputStream

                String clientIP = clientSocket.getInetAddress().getHostAddress();
                // Example: Reading data
                byte[] buffer = new byte[1024];
                int bytesRead;

                Log.d(LOG_TAG, " TCP Client Connected!: " + clientIP);

                while ((bytesRead = inputStream.read(buffer)) != -1) {
                    String receivedData = new String(buffer, 0, bytesRead);
                    // Handle received data as needed
                    Log.d(LOG_TAG, " TCP Client Sent: " + receivedData);

                    byte[] data = SensorData();
                    outputStream.write(data);

                }

//                // Example: Writing data
//                String sendData = "Hello, client!";
//                outputStream.write(sendData.getBytes());
                Log.d(LOG_TAG, " TCP Client Disconnected");

                // Close the client socket
                clientSocket.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    public void InitializeUDPClient() {

        try {
            udpSocket = new DatagramSocket(port);
            serverAddr = InetAddress.getByName(serverIPAddress);
            Log.d(LOG_TAG, " Worker Initiated! UDP Server Address: " + serverAddr);
            udpSocket.setBroadcast(true);
        } catch (SocketException e) {
            e.printStackTrace();
        } catch (UnknownHostException e) {
            e.printStackTrace();
        }

    }

    private byte[] SensorData(){
        try {

            boolean is_interrupted = Thread.currentThread().isInterrupted();
            String data = SensorData.queue.take();
            //handle the data
            // Send data to the running thread
            byte[] buf = data.getBytes();

            return  buf;
        } catch (InterruptedException e) {
            return null;
        }

    }

    private void SendDataToSocket(){
        while(!isStopped){
            Log.d(LOG_TAG, "Sent Data to: " + serverAddr);

            try {

                boolean is_interrupted = Thread.currentThread().isInterrupted();
                String data = SensorData.queue.take();
                //handle the data
                // Send data to the running thread
                byte[] buf = data.getBytes();
                DatagramPacket packet = new DatagramPacket(buf, buf.length,serverAddr, port);
                try {
                    udpSocket.send(packet);

                } catch (IOException e) {
                    e.printStackTrace();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                udpSocket.close();
            }
        }
        // Indicate whether the work finished successfully with the Result
        udpSocket.close();
    }
    @Override
    public void onStopped() {
        super.onStopped();
        isStopped = true;
        Log.d(LOG_TAG, "Worker Cancelled! ");
        udpSocket.close();
    }
}
