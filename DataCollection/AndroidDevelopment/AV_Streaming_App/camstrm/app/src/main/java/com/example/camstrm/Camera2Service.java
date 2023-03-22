package com.example.camstrm;


import static com.example.camstrm.MainActivity.mTcpClient;

import android.Manifest;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
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

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;

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
    private boolean isTaskStarted = false;
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
    /*
    Frames per seconds when
    no load = 29
    ByteBuffer buffer = img.getPlanes()[0].getBuffer(); = 31
    byte[] bytes = new byte[buffer.remaining()];
    buffer.get(bytes); = 40 !
    Bitmap bitmapImage = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null); = 30
    height=bitmapImage.getHeight();
    width=bitmapImage.getWidth();
    int[] data = new int[width * height];
    bitmapImage.getPixels(data, 0, width, 0, 0, width, height); = 30
    *****bottleneck***********
     String str_img=Arrays.toString(data);  = 4 :O

     */

    protected ImageReader.OnImageAvailableListener onImageAvailableListener = new ImageReader.OnImageAvailableListener() {
        //when images are available from camera they appear in this method
        @Override
        public void onImageAvailable(ImageReader reader) {
            Image img = null;

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

                if(img_list.size() < 10){

                    ImageData imageData=new ImageData(seq,img.getHeight(),img.getWidth(),bytes.length,bytes, img.getTimestamp());
                    img_list.add(imageData);
//                    mTcpClient.sendImage(imageData);


                }



//            if( Server.ready){
//                Long start = System.currentTimeMillis();
//                Server.seq+=1;
//                ImageData data=new ImageData(seq,img.getHeight(),img.getWidth(),bytes.length,bytes, img.getTimestamp());
//
//                if(Server.img_list.size() < 100){
//                    Log.d(TAG, "added image to list");
//                    Server.img_list.add(data);
//                }
//
//            }

            }catch (Exception e){

            }finally {
                if(img != null){
                    img.close();
                }

            }


        }
    };

    public void readyCamera(String cameraId) {
        CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
//            String pickedCamera = getCamera(manager);
            String pickedCamera = cameraId; // using the hardcoded cameraId instead of picking with code
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }
            manager.openCamera(pickedCamera, cameraStateCallback, null);
            imageReader = ImageReader.newInstance(640, 480, ImageFormat.JPEG, 2 /* images buffered */);
            imageReader.setOnImageAvailableListener(onImageAvailableListener, null);
            Log.d(TAG, "imageReader created");
        } catch (CameraAccessException e) {
            Log.e(TAG, e.getMessage());
        }
    }


    @RequiresApi(api = Build.VERSION_CODES.O)
    @Override
    public int onStartCommand(Intent intent,
                              int flags,
                              int startId) {
        String input = intent.getStringExtra("inputExtra");
        Intent notificationIntent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.S) {
            pendingIntent = PendingIntent.getActivity
                    (this, 0, notificationIntent, PendingIntent.FLAG_MUTABLE);
        }
        else
        {
            pendingIntent = PendingIntent.getActivity
                    (this, 0, notificationIntent, PendingIntent.FLAG_ONE_SHOT);
        }

        Notification notification = new NotificationCompat.Builder(this, MyApp.CHANNEL_ID)
                .setContentTitle("Auto Start Service")
                .setContentText(input)
                .setContentIntent(pendingIntent)
                .build();
        NotificationManager mNotificationManager = (NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            NotificationChannel channel = new NotificationChannel( MyApp.CHANNEL_ID, MyApp.CHANNEL_NAME, NotificationManager.IMPORTANCE_DEFAULT);
            mNotificationManager.createNotificationChannel(channel);
            new NotificationCompat.Builder(this, MyApp.CHANNEL_ID);
        }

        startForeground(1, notification);
        Log.d(TAG, "onStartCommand flags " + flags + " startId " + startId);
        //String cameraId = intent.getStringExtra("cameraId");
        readyCamera("0");

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
            cameraDevice.createCaptureSession(Arrays.asList(imageReader.getSurface()), sessionStateCallback, null);
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
            builder.set(CaptureRequest.JPEG_ORIENTATION, 180); // hardcoding orientation for the tomy camera
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