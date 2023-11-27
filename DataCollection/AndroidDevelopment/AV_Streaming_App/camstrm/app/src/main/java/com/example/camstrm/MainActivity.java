package com.example.camstrm;

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
import java.nio.file.attribute.FileTime;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MainActivity extends AppCompatActivity implements ImageViewCallback{
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};
    private static final int CAMERA_REQUEST_CODE = 10;
    protected static final String TAG = "cam_stream";
    public static TcpClient mTcpClient;
    public static FeedbackClient mFeedbackClient;

    String serverip = "172.27.164.148";
    String socketioPort ;
    int audioPort;
    int videoPort;
    int feedbackPort;
    private Context mContext;

    ImageView imageView;
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

        Log.i(TAG,"in main");
        mContext = this;

        this.serverip = getString(R.string.server_ip);
        this.socketioPort = getString(R.string.socketio_port);
        this.audioPort = Integer.parseInt(getString(R.string.audio_server_port));
        this.videoPort = Integer.parseInt(getString(R.string.video_server_port));
        this.feedbackPort = Integer.parseInt(getString(R.string.feedback_server_port));

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

        //button to close+exit app
        Button startbtn = (Button) findViewById(R.id.startbtn);
//        startbtn.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View view) {
//                Log.d("startup", "execute Feedback client: ");
//
//
//                new ConnectTask().execute();
//
//                // AsyncTasks are executed in order. ConnectTask takes time and thus FeedbackTask taking longer.
//                // Soln - execute parallely using a thread pool
////                FeedbackTask feedbackTask = new FeedbackTask();
////                feedbackTask.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);
//
//                //chack of the user has given permission for this app to use camera
//                checkPermissionsOrRequest();
//
//                if (hasCameraPermission()) {
//                    Intent cameraServiceIntent = new Intent(MainActivity.this, Camera2Service.class);
//                    Log.i(TAG,"starting cam..");
//
//                    // camera apis expect the cameraId to be a string
//                    // from testing, regular lens = 0, wide angle = 1
//                    String idString = Integer.toString(1);
//                    cameraServiceIntent.putExtra("cameraId", idString);
//                    Log.i(TAG,"starting service...");
//                    startService(cameraServiceIntent);
//                    //start service which access the camera and the stream of camera image frames
//                    //see the class Camera2Service.java class
//                    ContextCompat.startForegroundService(mContext, cameraServiceIntent);
//
//
//                } else {
//                    //if the user has not granted permission, request it
//                    requestPermission();
//                }
//
////
//                AudioStreamService audioStreamService = new AudioStreamService(mContext, serverip, audioPort);
//                audioStreamService.startStreaming();
//            }
//        });


        startCameraStream();

    }

    private void startCameraStream(){
        ConnectTask cameraStream = new ConnectTask();
        cameraStream.execute();
        // AsyncTasks are executed in order. ConnectTask takes time and thus FeedbackTask taking longer.
        // Soln - execute parallely using a thread pool
//                FeedbackTask feedbackTask = new FeedbackTask();
//                feedbackTask.executeOnExecutor(AsyncTask.THREAD_POOL_EXECUTOR);

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

            AudioStreamService audioStreamService = new AudioStreamService(mContext, serverip, audioPort);
            audioStreamService.startStreaming();

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

    @Override
    public void updateImageView(byte[] bytes) {

        runOnUiThread(new Runnable() {

            @Override
            public void run() {

                // Stuff that updates the UI
                Bitmap bitmapImage = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null);
//
                ImageView simpleImageView=(ImageView)  findViewById(R.id.image_view);
                simpleImageView.setImageBitmap(bitmapImage);//set the source in java class



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

//    public class ConnectTask extends AsyncTask<byte[], byte[], TcpClient> {
//
//        @Override
//        protected TcpClient doInBackground(byte[]... message) {
//
//
//            //we create a TCPClient object
//            mTcpClient = new TcpClient(new TcpClient.OnMessageReceived()  {
//                @Override
//                //here the messageReceived method is implemented
//                public void messageReceived(byte[] message) {
//                    //this method calls the onProgressUpdate
//                    Log.d("main", "publish progress is being called - should call onProgessUpdate");
//                    publishProgress(message);
//                }
//            }, serverip, videoPort, MainActivity.this,MainActivity.this);
//            mTcpClient.run();
//
//            return null;
//        }
//
//    }


    public class FeedbackTask extends AsyncTask<byte[], byte[], FeedbackClient> {

        @Override
        protected FeedbackClient doInBackground(byte[]... message) {

            // Create a TCPClient object
            mFeedbackClient = new FeedbackClient(new FeedbackClient.OnMessageReceived() {

                @Override
                // Implementation of messageReceived method
                public void messageReceived(byte[] message) {
                    publishProgress(message); // calls the onProgressUpdate method
                }

            }, serverip, feedbackPort);
            Log.d("feedback", "running Feedback client: ");

            mFeedbackClient.run();
            return null;
        }

        @Override
        protected void onProgressUpdate(byte[]... values) {
            super.onProgressUpdate(values);

            byte b[] = values[0];

            // update edittext
            String str = new String(b, Charset.forName("UTF-8"));
            Log.d("feedback", "Feedback Data received: " + str);
            if(str.startsWith("Protocol")){
                TextView serverPEditText = (TextView) findViewById(R.id.protocol_output);
                str = str.replace(':', '\n');
                serverPEditText.setText(str);
            }
            if(str.startsWith("Intervention")){
                TextView serverIEditText = (TextView) findViewById(R.id.intervention_output);
                str = str.replace(':', '\n');
                str = str.replace('|', '\n');
                serverIEditText.setText(str);
            }
            if(str.startsWith("Concept")){
//                TextView serverCEditText = (TextView) findViewById(R.id.concept_output);
//                serverCEditText.setText(str);
            }


        }
    }
}
