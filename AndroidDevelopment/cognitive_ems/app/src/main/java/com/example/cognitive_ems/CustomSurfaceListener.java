package com.example.cognitive_ems;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.util.Log;
import android.util.Size;
import android.view.Gravity;
import android.view.TextureView;
import android.widget.FrameLayout;

import androidx.annotation.NonNull;

public class CustomSurfaceListener implements TextureView.SurfaceTextureListener {

    protected CameraStreamActivity cameraHandler;
    protected TextureView textureView;
    protected boolean wait = false;
    protected int interval = 1000;

    private Bitmap bitmap;
    private SocketIOService socketIOService;

    private String TAG = "CustomSurfaceListener";


    public void destroy() {
        this.cameraHandler = null;
        this.textureView = null;
    }

    private static CustomSurfaceListener instance;

    public static CustomSurfaceListener getInstance(CameraStreamActivity cameraHandler, TextureView textureView) {
        if (instance == null) {
            instance = new CustomSurfaceListener(cameraHandler, textureView);
        }
        return instance;
    }

    private CustomSurfaceListener(CameraStreamActivity cameraHandler, TextureView textureView) {
        this.cameraHandler = cameraHandler;
        this.textureView = textureView;

        Log.d(TAG, "CustomSurfaceListener created!");
    }

    public void setService(SocketIOService service) {
        this.socketIOService = service;
    }
    @Override
    public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {
        if(cameraHandler == null) {
            Log.d(TAG, "CameraHandler is null");
            return;
        }
        cameraHandler.openCamera();
    }

    @Override
    public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {

    }

    @Override
    public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surfaceTexture) {
        this.cameraHandler = null;
        this.textureView = null;
        return true;
    }

    @Override
    public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surfaceTexture) {

        // generate bitmap and byte array stream for the compress
//        Log.d(TAG, "Got image to surfacetexture updated");
        // Adjust the TextureView's transformation here

        if(socketIOService == null) {
            Log.d(TAG, "SocketIOService is null");
            return;
        }

        if(textureView == null) {
            Log.d(TAG, "TextureView is null");
            return;
        }

//        bitmap = this.textureView.getBitmap();
//        this.socketIOService.sendImage(bitmap);
//        bitmap.recycle();

    }
}
