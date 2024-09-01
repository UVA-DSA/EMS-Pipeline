package com.example.gesturerecognition;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.util.Log;

import androidx.work.OneTimeWorkRequest;
import androidx.work.WorkManager;
import androidx.work.WorkRequest;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

public class SensorData implements SensorEventListener {

    private SensorManager sensorManager;
    private SensorDataCallback callback;

    private Sensor sensor_acc;
    private Sensor sensor_gyro;
    private String acc_data;
    private String gyro_data;
    protected static final String LOG_TAG = "SensorData";
    private Context context;
    private boolean isStarted = false;
    public static BlockingQueue<String> queue = new LinkedBlockingQueue<>();
    public static Long serverEpochTime = Long.valueOf(0);
    public static Long epochOffset = Long.valueOf(0);
    private String watchArm = "right";
    private Long accSeqNum = Long.valueOf(0);
    private Long gyroSeqNum = Long.valueOf(0);

    // List to accumulate data points
    private List<String> accumulatedData = new ArrayList<>();
    private static final int MAX_DATA_POINTS = 50;

    public static void calculateEpochOffset(Long time) {
        if( serverEpochTime == 0) {
            serverEpochTime = time;
            Long currentTimeMillis = System.currentTimeMillis();
            epochOffset = currentTimeMillis - serverEpochTime;
        }
    }

    public void startSensor() {
        Log.d(LOG_TAG, "startSensor initiated");

        if (!isStarted) {

            WorkRequest uploadWorkRequest =
                    new OneTimeWorkRequest.Builder(SendSensorDataWorker.class)
                            .build();

            WorkManager
                    .getInstance(this.context)
                    .enqueue(uploadWorkRequest);

            isStarted = true;
        }
    }

    public void stopSensor() {
        if (isStarted) {
            WorkManager.getInstance(this.context).cancelAllWork();
            isStarted = false;
        }
    }

    public void sendSensorData(String data) {
        if (isStarted) {
            accumulatedData.add(data);
            // Check if we have reached 100 data points
            if (accumulatedData.size() >= MAX_DATA_POINTS) {
                // Send all accumulated data at once
                // Concatenate all data in the queue into a single string
                StringBuilder combinedData = new StringBuilder();
                accumulatedData.forEach(combinedData::append);

                String message = combinedData.toString() + "eof";
                queue.offer(message);

                // Clear the list after sending
                accumulatedData.clear();
            }
        }
    }

    public SensorData(Context context, SensorDataCallback sensorDataCallback) {
        this.context = context;
        this.callback = sensorDataCallback;
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        sensor_acc = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        sensor_gyro = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Log.d(LOG_TAG, "Sensors" + sensor_acc);
        sensorManager.registerListener(this, sensor_acc, SensorManager.SENSOR_DELAY_GAME);
//        sensorManager.registerListener(this, sensor_gyro, SensorManager.SENSOR_DELAY_GAME);
    }

    @Override
    public void onSensorChanged(SensorEvent sensorEvent) {
        if(serverEpochTime == 0) {
            return;
        }
        Long currentTimeMillis = System.currentTimeMillis();

        Log.d(LOG_TAG, "Current System Time: " + currentTimeMillis);
        Log.d(LOG_TAG, "Received Server Time: " + serverEpochTime);
        Log.d(LOG_TAG, "Epoch Offset Time: " + epochOffset);

        currentTimeMillis -= epochOffset;
        String time = currentTimeMillis.toString();

        Log.d(LOG_TAG, "Offset Adjusted  Time: " + currentTimeMillis);

        Sensor sensor = sensorEvent.sensor;

        if (sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
            accSeqNum++;
            if (accSeqNum < 0) {
                accSeqNum = 0L;
            }

            double x = sensorEvent.values[0];
            double y = sensorEvent.values[1];
            double z = sensorEvent.values[2];

            acc_data = x + "," + y + "," + z;
            Log.d(LOG_TAG, "acc_3_axes: " + acc_data);
            String data_to_send = time + "," + watchArm + "," + "acc" + "," + acc_data + "," + accSeqNum + ";";
            sendSensorData(data_to_send);
            callback.onSensorDataReceived(String.valueOf(accSeqNum));

        } else if (sensor.getType() == Sensor.TYPE_GYROSCOPE) {
            gyroSeqNum++;
            if (gyroSeqNum < 0) {
                gyroSeqNum = 0L;
            }

            double x = sensorEvent.values[0];
            double y = sensorEvent.values[1];
            double z = sensorEvent.values[2];

            gyro_data = x + "," + y + "," + z;
            Log.d(LOG_TAG, "gyro_data: " + gyro_data);
            String data_to_send = time + "," + watchArm + "," + "gyro" + "," + gyro_data + "," + gyroSeqNum + ";";
            sendSensorData(data_to_send);
            callback.onSensorDataReceived(String.valueOf(gyroSeqNum));
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int i) {
    }

    public interface SensorDataCallback {
        void onSensorDataReceived(String data);
    }
}
