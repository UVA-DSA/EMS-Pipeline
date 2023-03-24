package com.example.camstrm;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;

public class MainActivity extends AppCompatActivity {
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};
    private static final int CAMERA_REQUEST_CODE = 10;
    protected static final String TAG = "cam_stream";
    public static TcpClient mTcpClient;
    String serverip = "";
    private Context mContext;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow(). addFlags (WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Log.i(TAG,"in main");
        mContext = this;

        //button to close+exit app
        Button btn1 = (Button) findViewById(R.id.btn1);
        btn1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
                System.exit(0);
            }
        });

        //button to close+exit app
        Button startbtn = (Button) findViewById(R.id.startbtn);
        startbtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                EditText serverEditText = (EditText) findViewById(R.id.serverip_input);
                serverip = serverEditText.getText().toString();

                // start server that sends frames to computer over ADB
                //        Server server=new Server();
                //        server.startServer();

                new ConnectTask().execute("");

                //chack of the user has given permission for this app to use camera
                checkPermissionsOrRequest();

                if (hasCameraPermission()) {
                    Intent cameraServiceIntent = new Intent(MainActivity.this, Camera2Service.class);
                    Log.i(TAG,"starting cam..");

                    // camera apis expect the cameraId to be a string
                    // from testing, regular lens = 0, wide angle = 1
                    String idString = Integer.toString(1);
                    cameraServiceIntent.putExtra("cameraId", idString);
                    Log.i(TAG,"starting service...");
                    startService(cameraServiceIntent);
                    //start service which access the camera and the stream of camera image frames
                    //see the class Camera2Service.java class
                    ContextCompat.startForegroundService(mContext, cameraServiceIntent);
                } else {
                    //if the user has not granted permission, request it
                    requestPermission();
                }

                AudioStreamService audioStreamService = new AudioStreamService(mContext, serverip,  Integer.parseInt(getString(R.string.audio_server_port)));
                audioStreamService.startStreaming();
            }
        });
    }

    private void checkPermissionsOrRequest() {
        // The request code used in ActivityCompat.requestPermissions()
        // and returned in the Activity's onRequestPermissionsResult()
        int PERMISSION_ALL = 1;
        String[] permissions = {
                Manifest.permission.READ_EXTERNAL_STORAGE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WAKE_LOCK,
                Manifest.permission.ACCESS_NETWORK_STATE
        };

        if (!hasPermissions(this, permissions)) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_ALL);
        }
    }

    public boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    Log.d(TAG, "hasPermissions: no permission for " + permission);
                    return false;
                } else {
                    Log.d(TAG, "hasPermissions: YES permission for " + permission);
                }
            }
        }
        return true;
    }

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED;
    }
    private void requestPermission() {
        ActivityCompat.requestPermissions(
                this,
                CAMERA_PERMISSION,
                CAMERA_REQUEST_CODE
        );
    }

    public class ConnectTask extends AsyncTask<String, String, TcpClient> {

        @Override
        protected TcpClient doInBackground(String... message) {


            //we create a TCPClient object
            mTcpClient = new TcpClient(new TcpClient.OnMessageReceived()  {
                @Override
                //here the messageReceived method is implemented
                public void messageReceived(String message) {
                    //this method calls the onProgressUpdate
                    publishProgress(message);
                }
            }, serverip, Integer.parseInt(getString(R.string.video_server_port)));
            mTcpClient.run();

            return null;
        }

        @Override
        protected void onProgressUpdate(String... values) {
            super.onProgressUpdate(values);
            //response received from server
            Log.d("test", "response " + values[0]);
            //process server response here....

        }
    }
}
