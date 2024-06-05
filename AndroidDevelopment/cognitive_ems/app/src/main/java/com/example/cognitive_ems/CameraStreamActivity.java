package com.example.cognitive_ems;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PointF;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.Collections;
import java.util.Comparator;

public class CameraStreamActivity extends AppCompatActivity implements TextureView.SurfaceTextureListener, FeedbackCallback {

    protected static final String TAG = "CameraStreamActivity";
    // Inside your activity or fragment:
    private static final int REQUEST_CAMERA_PERMISSION = 1;
    protected Size imageDimension;
    private String cameraId;
    private final String SERVER_IP = "192.168.0.13"; // Replace with your server IP
    private final int image_seq = 0;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder previewRequestBuilder;
    private Handler backgroundHandler;
    private TextureView textureView;

    private SocketIOService socketIoService;
    private boolean isBound = false;

    private HandlerThread backgroundThread;

    private CustomView customView;
    private TextView protocolBox; //Protocol text box

    private Bitmap bitmap;

    private WebSocketClient webSocketClient;
    private TextDisplayService tds_instance;




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Initialization code here
        setContentView(R.layout.activity_camera_stream); // Set the layout for the activity



        customView = findViewById(R.id.overlayView);

        CustomViewManager.getInstance().setOverlayView(customView);

//        // Example: Set a custom location and size for the rectangle
        Rect customRect = new Rect(500, 200, 800, 500); // Left, Top, Right, Bottom
        String object = "hands: 1.00";
        CustomViewManager.getInstance().updateRectangle(customRect, object);

        //dummy object to test with, take out eventually


        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            // Request permissions if needed
            return;
        }

        textureView = findViewById(R.id.textureView); // Make sure you have a SurfaceView in your layout
        textureView.setSurfaceTextureListener(this);

        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        this.protocolBox = (TextView)findViewById(R.id.protocolTextBox);

        this.tds_instance = TextDisplayService.getInstance();
//        this.tds_instance.setProtocolBox(protocolBox);
        //this.tds_instance.feedbackParser("{\"type\":\"Protocol\",\"protocol\":\"medical - knee pain - MCL suspected (protocol 2 - 1)\",\"protocol_confidence\":0.0209748435020447}");

        SocketStream.getInstance().setFeedbackCallback(this);

    }

    @Override
    protected void onStart() {
        super.onStart();

        Log.d(TAG, "On Start!");
//
        Intent intent = new Intent(this, SocketIOService.class);
        bindService(intent, connection, Context.BIND_AUTO_CREATE);
    }

    @Override
    protected void onResume() {
        super.onResume();

        startBackgroundThread();

        if(textureView.isAvailable()) {
            openCamera();
        } else {
            Log.d(TAG, "onResume setting texture listener");
//            textureView.setSurfaceTextureListener(this);
        }
    }


    // Don't forget to release the camera and stop the background thread in your activity's onPause() or onStop() method.
    @Override
    protected void onPause() {

        if(textureView != null) {
            Log.d(TAG, "onPause setting textureview to null");
            textureView.setSurfaceTextureListener(null);
            textureView = null;
        }

        if(bitmap != null) {
            bitmap.recycle();
            bitmap = null;
        }

        closeCamera();

        stopBackgroundThread();
        super.onPause();


    }

    // Initialize the background handler in your activity's onResume() method or after getting permissions


    @Override
    protected void onDestroy() {

        super.onDestroy();

        // Release the camera and its resources

        // Remove the SurfaceTextureListener to prevent memory leaks
        if(textureView != null) {
            Log.d(TAG, "onPause setting textureview to null");
            textureView.setSurfaceTextureListener(null);
            textureView = null;
        }

        if(bitmap != null) {
            bitmap.recycle();
            bitmap = null;
        }

        closeCamera();


        // Unbind from the service to prevent memory leaks
        if (isBound) {
            unbindService(connection);
            isBound = false;
            socketIoService = null; // Nullify the reference to help GC
        }

        // Shutdown the background handler thread
        if (backgroundHandler != null) {
            backgroundHandler.getLooper().quitSafely();
            backgroundHandler = null; // Help GC by nullifying the reference
        }

    }

    private void startBackgroundThread(){
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    private void stopBackgroundThread(){
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private final ServiceConnection connection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className, IBinder service) {
            Log.d(TAG, "Service bound!");

            SocketIOService.SocketIOServiceBinder binder = (SocketIOService.SocketIOServiceBinder) service;
            socketIoService = binder.getService();
            isBound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            isBound = false;
        }
    };


    private final CameraDevice.StateCallback stateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(CameraDevice camera) {
            cameraDevice = camera;
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(CameraDevice camera) {
            camera.close();
            cameraDevice = null;
        }

        @Override
        public void onError(CameraDevice camera, int error) {
            camera.close();
            cameraDevice = null;
        }
    };



    public void openCamera() {
        CameraManager manager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);


        try {
            cameraId = manager.getCameraIdList()[0]; // Choose the camera you want to use. Here, 0 is typically the rear camera.


            CameraCharacteristics characteristics = manager.getCameraCharacteristics(cameraId);
            StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            assert map != null;
            imageDimension = map.getOutputSizes(SurfaceTexture.class)[0];

            setTextureTransform(characteristics);
            // Check if permissions are granted
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
                return;
            }

            manager.openCamera(cameraId, stateCallback, backgroundHandler);

        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    void setTextureTransform(CameraCharacteristics characteristics) {
        Size previewSize = getPreviewSize(characteristics);
        int width = previewSize.getWidth();
        int height = previewSize.getHeight();
        int sensorOrientation = getCameraSensorOrientation(characteristics);
        // Indicate the size of the buffer the texture should expect
        textureView.getSurfaceTexture().setDefaultBufferSize(width, height);
        // Save the texture dimensions in a rectangle
        RectF viewRect = new RectF(0, 0, textureView.getWidth(), textureView.getHeight());
        // Determine the rotation of the display
        float rotationDegrees = 0;
        try {
            rotationDegrees = (float) getDisplayRotation();
        } catch (Exception ignored) {
        }
        float w, h;
        if ((sensorOrientation - rotationDegrees) % 180 == 0) {
            w = width;
            h = height;
        } else {
            // Swap the width and height if the sensor orientation and display rotation don't match
            w = height;
            h = width;
        }
        float viewAspectRatio = viewRect.width() / viewRect.height();
        float imageAspectRatio = w / h;
        final PointF scale;
        // This will make the camera frame fill the texture view, if you'd like to fit it into the view swap the "<" sign for ">"
        if (viewAspectRatio < imageAspectRatio) {
            // If the view is "thinner" than the image constrain the height and calculate the scale for the texture width
            scale = new PointF((viewRect.height() / viewRect.width()) * ((float) height / (float) width), 1f);
        } else {
            scale = new PointF(1f, (viewRect.width() / viewRect.height()) * ((float) width / (float) height));
        }
        if (rotationDegrees % 180 != 0) {
            // If we need to rotate the texture 90º we need to adjust the scale
            float multiplier = viewAspectRatio < imageAspectRatio ? w / h : h / w;
            scale.x *= multiplier;
            scale.y *= multiplier;
        }

        Matrix matrix = new Matrix();
        // Set the scale
        matrix.setScale(scale.x, scale.y, viewRect.centerX(), viewRect.centerY());
        if (rotationDegrees != 0) {
            // Set rotation of the device isn't upright
            matrix.postRotate(0 - rotationDegrees, viewRect.centerX(), viewRect.centerY());
        }
        // Transform the texture
        textureView.setTransform(matrix);
    }

    int getDisplayRotation() {
        switch (textureView.getDisplay().getRotation()) {
            case Surface.ROTATION_0:
            default:
                return 0;
            case Surface.ROTATION_90:
                return 90;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_270:
                return 270;
        }
    }

    Size getPreviewSize(CameraCharacteristics characteristics) {
        StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
        Size[] previewSizes = map.getOutputSizes(SurfaceTexture.class);
        // TODO: decide on which size fits your view size the best
        return previewSizes[0];
    }

    int getCameraSensorOrientation(CameraCharacteristics characteristics) {
        Integer cameraOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
        return (360 - (cameraOrientation != null ? cameraOrientation : 0)) % 360;
    }

    private void createCameraPreviewSession() {
        try {
            SurfaceTexture texture = textureView.getSurfaceTexture();
            assert texture != null;
            //set a custom texture change listener callback
            texture.setDefaultBufferSize(imageDimension.getWidth(), imageDimension.getHeight());
            Surface surface = new Surface(texture);

            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);

            previewRequestBuilder.addTarget(surface);

            cameraDevice.createCaptureSession(Collections.singletonList(surface),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            if (cameraDevice == null) {
                                return;
                            }

                            captureSession = session;
                            try {
                                // Auto focus should be continuous for camera preview.
                                previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                // Flash is automatically enabled when necessary.
                                previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE,
                                        CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                                previewRequestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<Integer>(10, 20));

                                // Finally, we start displaying the camera preview.
                                CaptureRequest previewRequest = previewRequestBuilder.build();
                                captureSession.setRepeatingRequest(previewRequest, null, backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            // Handle configuration failure
                        }
                    }, backgroundHandler
            );
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == REQUEST_CAMERA_PERMISSION) {
            if (grantResults.length != 1 || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                // Handle the case where the user denies the camera permission
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }


    private void closeCamera(){
        if (null != captureSession) {
            captureSession.close();
            captureSession = null;
        }
        if (null != cameraDevice) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    @Override
    public void onConfigurationChanged(Configuration newConfig) {
        super.onConfigurationChanged(newConfig);

    }

    @Override
    public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {
        openCamera();
    }

    @Override
    public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {

    }

    @Override
    public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surfaceTexture) {
        closeCamera();

        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surfaceTexture) {

        // calculate time taken for next two instructions
        long startTime = System.currentTimeMillis();
        bitmap = textureView.getBitmap(640, 480);
        long endTime = System.currentTimeMillis();
        Log.d(TAG, "Time taken to get bitmap: " + (endTime - startTime) + "ms");

        socketIoService.sendImage(bitmap);
//        bitmap.recycle();

    }

    @Override
    public void onFeedbackReceived(String feedback) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                tds_instance.setProtocolBox(protocolBox);
                tds_instance.feedbackParser(feedback);

            }
        });
    }

    /**
     * Comparator based on the area of camera preview sizes.
     */
    private static class CompareSizesByArea implements Comparator<Size> {
        @Override
        public int compare(Size lhs, Size rhs) {
            // We cast here to ensure the multiplications won't overflow
            return Long.signum((long) lhs.getWidth() * lhs.getHeight() -
                    (long) rhs.getWidth() * rhs.getHeight());
        }
    }

}
