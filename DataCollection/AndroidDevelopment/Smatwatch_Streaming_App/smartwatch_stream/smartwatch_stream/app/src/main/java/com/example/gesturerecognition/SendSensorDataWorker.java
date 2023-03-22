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
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class SendSensorDataWorker extends Worker {

    private static final String DEBUG_TAG = "Worker";
    private String serverIPAddress = "172.27.162.26";
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
        InitializeUDPClient();

        setRunInForeground(true);

        SendDataToSocket();

        return Result.success();

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
