package com.example.gesturerecognition;

import android.app.Activity;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.work.Constraints;
import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.ExistingWorkPolicy;
import androidx.work.OneTimeWorkRequest;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;
import androidx.work.WorkRequest;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.InetAddress;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

public class SensorData  implements SensorEventListener {

    private SensorManager sensorManager;
    private Sensor sensor_acc;
    private Sensor sensor_gyro;
    private double avg = 0;
    private String acc_data;
    private String gyro_data;
    private static final double P = 0.7;
    protected static final String LOG_TAG = "SensorData";
    private Context context;
    private  boolean isStarted = false;
    public static BlockingQueue<String> queue = new LinkedBlockingQueue<String>();
    public static Long time_elapsed = Long.valueOf(0);
//    private String watchArm = "left";
    private String watchArm = "right";
    private Long startTime;

    public  void startSensor(){
        Log.d(LOG_TAG, "startSensor initiated");

        if(!isStarted){

        startTime = System.currentTimeMillis();
        WorkRequest uploadWorkRequest =
                new OneTimeWorkRequest.Builder(SendSensorDataWorker.class)
                        .build();
//        PeriodicWorkRequest uploadWorkRequest = new
//                    PeriodicWorkRequest.Builder(SendSensorDataWorker.class, 24, TimeUnit.HOURS)
//                    .setConstraints(new Constraints.Builder()
//                            .setRequiresCharging(true)
//                            .build()
//                    )
//                    .build();
        WorkManager
                .getInstance(this.context)
                .enqueue(uploadWorkRequest);

//        udp_client = new UDP_Client();
//        Log.d(LOG_TAG, "UDP Client" + udp_client);
//         udp_thread = new Thread(udp_client);
//        udp_thread.start();
        isStarted = true;
        }
    }

    public  void stopSensor(){
        if(isStarted){
//            udp_thread.interrupt();
            WorkManager.getInstance(this.context).cancelAllWork();
            isStarted = false;
        }
    }
    public void sendSensorData(String data){
        if(isStarted) {
//            udp_client.queue.offer(data);
            queue.offer(data);
        }
    }

    public SensorData(Context context) {
        this.context = context;
        sensorManager = (SensorManager) context.getSystemService(context.SENSOR_SERVICE);
        sensor_acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensor_gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Log.d(LOG_TAG, "Sensors" + sensor_acc);
        sensorManager.registerListener(this, sensor_acc, SensorManager.SENSOR_DELAY_GAME);
//        sensorManager.registerListener(this, sensor_gyro, SensorManager.SENSOR_DELAY_GAME);

    }


    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {


        String time = Long.toString(System.currentTimeMillis());
        Sensor sensor = sensorEvent.sensor;
        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            //TODO: get values

            // Acquire measurement values from event
            double x = sensorEvent.values[0]; // X axis
            double y = sensorEvent.values[1]; // y axis
            double z = sensorEvent.values[2]; // z axis

            // Do something with the values

            acc_data = x +","+ y + ","+ z;
            Log.d(LOG_TAG, "acc_3_axes: " + acc_data);
            String data_to_send = time + ","+watchArm+ ","+"acc"+","+acc_data;
            sendSensorData(data_to_send);

        }else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            //TODO: get values
            // Acquire measurement values from event
            double x = sensorEvent.values[0]; // X
            double y = sensorEvent.values[1]; // y
            double z = sensorEvent.values[2]; // z

            gyro_data  = x +","+ y + ","+ z;

            Log.d(LOG_TAG, "gyro_data: "+gyro_data);
            String data_to_send = time + ","+watchArm+ ","+"gyro"+","+gyro_data;
            sendSensorData(data_to_send);

        }




    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {

    }

}
