package com.example.cognitive_ems;

import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.util.Base64OutputStream;
import android.util.Log;
import android.view.TextureView;

import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class ImageCompressor {
    private final SocketStream socketIOAdapter;
    private final ExecutorService executorService;
    private Bitmap bmp;

    private byte[] byteArray;
    private ByteArrayOutputStream stream;


    private static ImageCompressor instance;

    public static synchronized ImageCompressor getInstance( SocketStream socketIOAdapter) {
        if (instance == null) {
            instance = new ImageCompressor( socketIOAdapter);
        }
        return instance;
    }


    private ImageCompressor(SocketStream socketIOAdapter) {
        this.socketIOAdapter = socketIOAdapter;
        this.executorService = Executors.newSingleThreadExecutor();

        this.stream = new ByteArrayOutputStream();

    }

    public void compressAndSend(Bitmap bmp) {

//        try {
//
//            if(bmp == null) {
//                Log.e("ImageCompressor", "Bitmap is null");
//                return;
//            }
//            bmp.compress(Bitmap.CompressFormat.JPEG, 25, stream); // Adjust the format and quality as needed
//
//            byteArray = stream.toByteArray();
//
////                String base64String = Base64.encodeToString(byteArray, Base64.DEFAULT);
//
//            if (byteArray != null) {
//                socketIOAdapter.sendBytes(byteArray);
//                byteArray = null;
//            } else {
//                Log.e("ImageCompressor", "Error while sending image");
//            }
//        } catch (Exception e) {
//            e.printStackTrace();
//        } finally {
//            stream.reset();
//
//        }

        executorService.submit(() -> {
            try {

                if(bmp == null) {
                    Log.e("ImageCompressor", "Bitmap is null");
                    return;
                }
                bmp.compress(Bitmap.CompressFormat.JPEG, 40, stream); // Adjust the format and quality as needed

                byteArray = stream.toByteArray();
                stream.reset();

//                String base64String = Base64.encodeToString(byteArray, Base64.DEFAULT);

                new Handler(Looper.getMainLooper()).post(() -> {
                    if (byteArray != null) {
                        socketIOAdapter.sendBytes(byteArray);

                        byteArray = null;
                    } else {
                        Log.e("ImageCompressor", "Error while sending image");
                    }

                });
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                bmp.recycle();
            }
        });
    }

    // Method to shut down the ExecutorService
    public void shutdownExecutor() {
        if (!executorService.isShutdown()) {
            executorService.shutdown();
        }
    }
}
