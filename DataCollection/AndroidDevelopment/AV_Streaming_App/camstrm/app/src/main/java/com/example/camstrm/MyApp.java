package com.example.camstrm;


import android.app.Application;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.os.Build;
import android.util.Log;

public class MyApp extends Application {
    public static final String CHANNEL_ID = "autoStartServiceChannel";
    public static final String CHANNEL_NAME = "Auto Start Service Channel";
    protected static final String TAG = "cam_stream";

    @Override
    public void onCreate() {
        super.onCreate();
        Log.i(TAG,"in  my app");
        createNotificationChannel();

    }
    private void createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel serviceChannel = new NotificationChannel(
                    CHANNEL_ID,
                    CHANNEL_NAME,
                    NotificationManager.IMPORTANCE_DEFAULT
            );
            NotificationManager manager = getSystemService(NotificationManager.class);
            manager.createNotificationChannel(serviceChannel);
        }
    }
}