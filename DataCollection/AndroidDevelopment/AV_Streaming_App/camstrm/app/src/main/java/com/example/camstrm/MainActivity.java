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
import java.nio.charset.Charset;
import java.nio.file.attribute.FileTime;

public class MainActivity extends AppCompatActivity {
    private static final String[] CAMERA_PERMISSION = new String[]{Manifest.permission.CAMERA};
    private static final int CAMERA_REQUEST_CODE = 10;
    protected static final String TAG = "cam_stream";
    public static TcpClient mTcpClient;
    public static FeedbackClient mFeedbackClient;

    String serverip = "172.27.164.148";
    private Context mContext;

    ImageView imageView;
    ActivityResultLauncher<Intent> activityResultLauncher;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LayoutInflater inflater = LayoutInflater.from(MainActivity.this); // or (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
        View viewMyLayout = inflater.inflate(R.layout.activity_main, null);
        setContentView(viewMyLayout);
//        setContentView(R.layout.activity_main);
        getWindow(). addFlags (WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Log.i(TAG,"in main");
        mContext = this;

//        Intent intent = getIntent();
//        Bundle extras = getIntent().getExtras();
//        String userName;
//
//        if (extras != null) {
////            userName = extras.getString("name");
//            Uri myUri = Uri.parse(extras.getString("uri"));
////            ImageView img = new ImageView(this);
//            ImageView imageView= (ImageView) findViewById(R.id.image_view);
//            imageView.setImageURI(null);
//            imageView.setImageURI(myUri);
//        }

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
//                EditText serverEditText = (EditText) findViewById(R.id.serverip_input);
//                serverip = serverEditText.getText().toString();

                // start server that sends frames to computer over ADB
                //        Server server=new Server();
                //        server.startServer();

                new ConnectTask().execute();
                new FeedbackTask().execute();

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

    private Uri saveImage(Bitmap image, MainActivity context) {

        File imagefolder = new File(context.getCacheDir(), "images");
        Uri uri = null;
        try{
            imagefolder.mkdirs();
            File file = new File(imagefolder, "captured_image.jpg");
            FileOutputStream stream = new FileOutputStream(file);
            image.compress(Bitmap.CompressFormat.JPEG, 100, stream);
            stream.flush();
            stream.close();
            uri = FileProvider.getUriForFile(context.getApplicationContext(), "com.allcodingtutorial.camerafull1"+".provider", file);
        }
        catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return uri ;
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

    public class ConnectTask extends AsyncTask<byte[], byte[], TcpClient> {

        @Override
        protected TcpClient doInBackground(byte[]... message) {


            //we create a TCPClient object
            mTcpClient = new TcpClient(new TcpClient.OnMessageReceived()  {
                @Override
                //here the messageReceived method is implemented
                public void messageReceived(byte[] message) {
                    //this method calls the onProgressUpdate
                    Log.d("main", "publish progress is being called - should call onProgessUpdate");
                    publishProgress(message);
                }
            }, serverip, Integer.parseInt(getString(R.string.video_server_port)), MainActivity.this);
            mTcpClient.run();

            return null;
        }

//        @Override
        protected void onProgressUpdate(byte... values) {
            super.onProgressUpdate(values);
            //response received from server
//            Log.d("test", "response " + values[0]);

            //if receiving bytes instead:
            Log.d("main", "on progress update is being called in main ");
            Bitmap bmp = BitmapFactory.decodeByteArray(values, 0, values.length);
            ImageView image = (ImageView) findViewById(R.id.image_view);
            image.setImageBitmap(Bitmap.createScaledBitmap(bmp, image.getWidth(), image.getHeight(), false));


            //process server response here....

        }
    }


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

            });

            mFeedbackClient.run();
            Log.d("feedback", "running Feedback client: ");
            return null;
        }

        @Override
        protected void onProgressUpdate(byte[]... values) {
            super.onProgressUpdate(values);

            byte b[] = values[0];

            // update edittext
            String str = new String(b, Charset.forName("UTF-8"));
            Log.d("feedback", "Feedback Data received: " + str);

            TextView serverEditText = (TextView) findViewById(R.id.output);
            serverEditText.setText(str);


        }
    }
}
