package com.example.camstrm;

import java.nio.ByteBuffer;

public interface ImageViewCallback {

    public void updateImageView(byte[] bytes);


    public void updateTextMainActivity(String message);

}
