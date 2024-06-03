package com.example.gesturerecognition;

import android.app.Activity;
import android.app.Notification;
import android.content.Context;
import android.net.ConnectivityManager;
import android.net.Network;
import android.net.NetworkCapabilities;
import android.net.NetworkInfo;
import android.net.NetworkRequest;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.core.app.NotificationCompat;
import androidx.core.app.NotificationManagerCompat;

import com.example.gesturerecognition.databinding.ActivityMainBinding;

import java.io.IOException;

public class MainActivity extends Activity  {

    private static TextView mTextView;
    private static TextView mIPTextView;
    private Button mButton;
    private ActivityMainBinding binding;
    private static final String DEBUG_TAG = "NetworkStatusExample";
    private ConnectivityManager connMgr;
    private String welcomeMsg = "NIST - Cognitive EMS";
    private boolean isStarted = false;
    private SensorData mSensor;
//    private String watchArm = "Left Wrist";
    private String message = "DCS - Right Wrist";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());


        mTextView = binding.text;
        mIPTextView = binding.ip;
        mTextView.setText(welcomeMsg);
//        mButton = binding.startBtn;
        connMgr =
                (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        boolean isWifiConn = false;
        boolean isMobileConn = false;
        for (Network network : connMgr.getAllNetworks()) {
            NetworkInfo networkInfo = connMgr.getNetworkInfo(network);
            if (networkInfo.getType() == ConnectivityManager.TYPE_WIFI) {
                isWifiConn |= networkInfo.isConnected();
            }
            if (networkInfo.getType() == ConnectivityManager.TYPE_MOBILE) {
                isMobileConn |= networkInfo.isConnected();
            }
        }
        Log.d(DEBUG_TAG, "Wifi connected: " + isWifiConn);
        Log.d(DEBUG_TAG, "Internet connected: " + isOnline());
        if(isOnline()){
            bindNetwork();
            mSensor = new SensorData(this);
        }

        mTextView.setText(message);
        mIPTextView.setText(SendSensorDataWorker.getLocalIpAddress());
        mSensor.startSensor();

    }

    public boolean isOnline() {
        ConnectivityManager connMgr = (ConnectivityManager)
                getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo networkInfo = connMgr.getActiveNetworkInfo();
        return (networkInfo != null && networkInfo.isConnected());
    }

    public static void setTimeText(Long elapsedTime){
        Long sec = elapsedTime%60;
        Long min = elapsedTime/60;
        String elapsedTimeText = String.format("%02d:%02d", min, sec);
        mTextView.setText(elapsedTimeText);
    }

    public void bindNetwork(){
        ConnectivityManager.NetworkCallback callback = new ConnectivityManager.NetworkCallback() {
            public void onAvailable(Network network) {
                super.onAvailable(network);
                // The Wi-Fi network has been acquired, bind it to use this network by default
                connMgr.bindProcessToNetwork(network);
                Log.d(DEBUG_TAG, "Wifi network Binded: ");

//                UDP_Client p = new UDP_Client();
//                new Thread(p).start();

            }

            public void onLost(Network network) {
                super.onLost(network);
                // The Wi-Fi network has been disconnected
            }
        };
        connMgr.requestNetwork(
                new NetworkRequest.Builder().addTransportType(NetworkCapabilities.TRANSPORT_WIFI).build(),
                callback
        );
    }

//    On click
    public void start_stopUDPClient(View view){

        if(!isStarted) {

            Log.d(DEBUG_TAG, "Button Clicked");
            mTextView.setText(message + " Recording..");
            mSensor.startSensor();
            mButton.setText("Stop");
            isStarted = true;
        }else {
            Log.d(DEBUG_TAG, "Button Clicked");
            mTextView.setText(message);
            mSensor.stopSensor();
            mButton.setText("Start");
            isStarted = false;
        }
    }
}