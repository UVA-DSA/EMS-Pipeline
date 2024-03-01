package com.example.cognitive_ems;



import android.Manifest;
import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.IBinder;
import android.util.Log;
import android.util.Range;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.ImageView;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;
import androidx.core.app.NotificationCompat;

public class Camera2Service extends Service {


    protected static final int CAMERA_CALIBRATION_DELAY = 5000;
    protected static final String TAG = "camera2Service";
    protected static final int CAMERACHOICE = CameraCharacteristics.LENS_FACING_FRONT;
    protected static long cameraCaptureStartTime;
    protected CameraDevice cameraDevice;
    protected CameraCaptureSession session;
    protected ImageReader imageReader;
    protected CameraCharacteristics camCharacteristics;
    protected int seq=0;
    int ONGOING_NOTIFICATION_ID=1;
    private static final int ID_SERVICE = 101;
    public static volatile ArrayList<ImageData> img_list=new ArrayList<ImageData>();
    private final boolean isTaskStarted = false;

//    LayoutInflater inflater = LayoutInflater.from(MainActivity.this); // or (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);
//    View viewMyLayout = inflater.inflate(R.layout.activity_main, null);
//    ImageView imageView1 = (ImageView)viewMyLayout.findViewById(R.id.image_view);
    protected CameraDevice.StateCallback cameraStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            Log.d(TAG, "CameraDevice.StateCallback onOpened");
            cameraDevice = camera;
            actOnReadyCameraDevice();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            Log.w(TAG, "CameraDevice.StateCallback onDisconnected");
            cameraDevice = camera;
            closeCameraDevice();
        }

        @Override
        public void onError(@NonNull CameraDevice camera,
                            int error) {
            Log.e(TAG, "CameraDevice.StateCallback onError " + error);
        }
    };

    protected CameraCaptureSession.StateCallback sessionStateCallback = new CameraCaptureSession.StateCallback() {

        @Override
        public void onReady(CameraCaptureSession session) {
            Log.d(TAG, "sessionstatecallback onReady() ");
            Camera2Service.this.session = session;
            try {
                session.setRepeatingRequest(createCaptureRequest(), null, null);
//                session.capture(createCaptureRequest(), null, null);
                cameraCaptureStartTime = System.currentTimeMillis();
            } catch (CameraAccessException e) {
                Log.e(TAG, e.getMessage());
            }
        }


        @Override
        public void onConfigured(CameraCaptureSession session) {

        }

        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession session) {

        }
    };

    protected ImageReader.OnImageAvailableListener onImageAvailableListener = new ImageReader.OnImageAvailableListener() {
        //when images are available from camera they appear in this method
        @Override
        public void onImageAvailable(ImageReader reader) {
            Image img = null;
//            Intent img_intent = new Intent(Camera2Service.this.getApplicationContext(), MainActivity.class);

            try{

                Long timestamp = System.currentTimeMillis();
                img = reader.acquireLatestImage();

                if(img.getPlanes().length == 0){
                    return;
                }
                ByteBuffer buf = img.getPlanes()[0].getBuffer();

                byte[] bytes = new byte[buf.remaining()];
                buf.get(bytes);

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    img.setTimestamp(timestamp);
                }

                Bitmap originalBitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length);

// Compress the Bitmap with a quality factor (0-100)
                int quality = 75; // Adjust the quality as needed
                ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                originalBitmap.compress(Bitmap.CompressFormat.JPEG, quality, outputStream);

                bytes = outputStream.toByteArray();

                if(img_list.size() < 1){
                    ImageData imageData=new ImageData(seq,img.getHeight(),img.getWidth(),bytes.length,bytes, img.getTimestamp());
                    img_list.add(imageData);
                }

            }catch (Exception e){

            }finally {
                if(img != null){
                    img.close();
                }
            }


        }
    };

    public void readyCamera(String cameraId) {
        CameraManager cameraManager = (CameraManager) getSystemService(CAMERA_SERVICE);
        Log.d(TAG, "readyCamera created");

        try {

            String pickedCamera = cameraManager.getCameraIdList()[0]; // Choose the camera you want to use. Here, 0 is typically the rear camera

//            String pickedCamera = getCamera(manager);
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }
            cameraManager.openCamera(pickedCamera, cameraStateCallback, null);
            imageReader = ImageReader.newInstance(640, 480, ImageFormat.JPEG, 2 /* images buffered */);
            imageReader.setOnImageAvailableListener(onImageAvailableListener, null);
            Log.d(TAG, "imageReader created");
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
        }
    }


    @Override
    public int onStartCommand(Intent intent,
                              int flags,
                              int startId) {
        String cameraId = intent.getStringExtra("cameraId");
        Log.d(TAG, "onStartCommand service");

        // Create a notification channel (for Android O and above)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel("YourServiceChannel",
                    "Your Service Channel",
                    NotificationManager.IMPORTANCE_DEFAULT);
            NotificationManager manager = getSystemService(NotificationManager.class);
            if (manager != null) {
                manager.createNotificationChannel(channel);
            }
        }

        // Create a notification
        NotificationCompat.Builder notificationBuilder = new NotificationCompat.Builder(this, "YourServiceChannel")
                .setContentTitle("Your Service Running")
                .setContentText("This is your service running in the foreground")
                .setSmallIcon(R.drawable.ic_launcher_background);

        Notification notification = notificationBuilder.build();

        // Start the service in the foreground
        startForeground(1, notification);

        readyCamera(cameraId);
        return START_NOT_STICKY;
    }

    @Override
    public void onCreate() {
        Log.d(TAG, "onCreate service");
        super.onCreate();
    }

    public void actOnReadyCameraDevice() {
        Log.d(TAG, "actOnReadyCameraDevice");
        try {
            cameraDevice.createCaptureSession(Collections.singletonList(imageReader.getSurface()), sessionStateCallback, null);
        } catch (CameraAccessException e) {
            Log.d(TAG, "actOnReadyCameraDevice");
            Log.e(TAG, e.getMessage());
        }
    }

    @Override
    public void onDestroy() {
        closeCameraDevice();
    }

    void closeCameraDevice() {
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    protected CaptureRequest createCaptureRequest() {
        try {
            CaptureRequest.Builder builder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_RECORD);
            builder.addTarget(imageReader.getSurface());
//            WindowManager windowManager = (WindowManager)getSystemService(WINDOW_SERVICE);
//            int rotation = windowManager.getDefaultDisplay().getRotation();
//            int jpegRotation = getJpegOrientation(camCharacteristics, rotation);
            Range<Integer> fpsRange = new Range<>(15,30);
//            builder.set(CaptureRequest.JPEG_ORIENTATION, 180); // hardcoding orientation for the tomy camera
            builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,fpsRange);
            return builder.build();
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
            return null;
        }
    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }
}