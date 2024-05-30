package com.example.cognitive_ems;

import androidx.activity.result.ActivityResult;
import androidx.activity.result.ActivityResultCallback;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.attribute.FileTime;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MainActivity extends AppCompatActivity implements ImageViewCallback{
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};
    private static final String[] AUDO_PERMISSION = new String[]{Manifest.permission.RECORD_AUDIO};
    private static final int CAMERA_REQUEST_CODE = 10;
    protected static final String TAG = "cam_stream";
    public static TcpClient mTcpClient;

    String serverip = "172.27.164.148";
    String socketioPort ;
    int audioPort;
    int videoPort;
    private Context mContext;

    // Get a reference to the TextView
    private TextView textView;

    ActivityResultLauncher<Intent> activityResultLauncher;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LayoutInflater inflater = LayoutInflater.from(MainActivity.this); // or (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        View viewMyLayout = inflater.inflate(R.layout.activity_main, null);
        setContentView(viewMyLayout);
//        setContentView(R.layout.activity_main);
        getWindow(). addFlags (WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getSupportActionBar().hide();

        textView = findViewById(R.id.recording_status);

        mContext = this;

        this.serverip = getString(R.string.server_ip);
        this.socketioPort = getString(R.string.socketio_port);
        this.audioPort = Integer.parseInt(getString(R.string.audio_server_port));
        this.videoPort = Integer.parseInt(getString(R.string.video_server_port));

        Intent socketIOserviceIntent = new Intent(this,SocketIOService.class);
        socketIOserviceIntent.putExtra("server_ip",this.serverip);
        socketIOserviceIntent.putExtra("server_port",this.socketioPort);
        startService(socketIOserviceIntent);


        //button to close+exit app
        Button btn1 = (Button) findViewById(R.id.btn1);
        btn1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
                System.exit(0);
            }
        });

        startCameraStream();
        startAudioStream();

    }


    private void startAudioStream(){

        checkPermissionsOrRequest();

        if (hasAudioPermission()) {

            AudioStreamService audioStreamService = new AudioStreamService(mContext, serverip, audioPort);
            audioStreamService.startStreaming();

        } else {
            //if the user has not granted permission, request it
            requestPermission();
        }


    }

    private void startCameraStream(){

        checkPermissionsOrRequest();

        ConnectTask cameraStream = new ConnectTask();
        cameraStream.execute();

        if (hasCameraPermission()) {
            Intent cameraServiceIntent = new Intent(MainActivity.this, Camera2Service.class);
            Log.i(TAG,"starting cam..");

            // camera apis expect the cameraId to be a string
            // from testing, regular lens = 0, wide angle = 1
            String idString = Integer.toString(0);
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

//            AudioStreamService audioStreamService = new AudioStreamService(mContext, serverip, audioPort);
//            audioStreamService.startStreaming();

    }

    private void checkPermissionsOrRequest() {
        // The request code used in ActivityCompat.requestPermissions()
        // and returned in the Activity's onRequestPermissionsResult()
        int PERMISSION_ALL = 1;
        String[] permissions = {
                Manifest.permission.CAMERA,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.WAKE_LOCK,
                Manifest.permission.ACCESS_NETWORK_STATE,
                Manifest.permission.INTERNET
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

    private boolean hasAudioPermission() {
        return ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED;
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

    @Override
    public void updateImageView(byte[] bytes) {

        runOnUiThread(new Runnable() {

            @Override
            public void run() {


            }
        });


    }

    @Override
    public void updateTextMainActivity(String message) {
        runOnUiThread(new Runnable() {

            @Override
            public void run() {

                TextView textView1 = findViewById(R.id.recording_status);

                Log.d(TAG, "updateTextMainActivity: " + message);
                    if(textView1 != null) {
                        textView1.setText(message);
                }else{
                        Log.d(TAG, "Text View NULL");
                    }


            }
        });

    }


    public class ConnectTask {

        private final ExecutorService executor = Executors.newSingleThreadExecutor();
        private Future<?> taskFuture;

        public void execute() {
            taskFuture = executor.submit(() -> {
                // Perform your background operation here

                TcpClient tcpClient = new TcpClient(serverip, videoPort, MainActivity.this, MainActivity.this);
                tcpClient.run();

            });
        }


        public void cancelTask() {
            if (taskFuture != null && !taskFuture.isDone()) {
                taskFuture.cancel(true);
            }
        }

        // Don't forget to properly handle the clean-up and shutdown of the executor service when finished
        public void shutdownExecutor() {
            executor.shutdown();
        }
    }





}
