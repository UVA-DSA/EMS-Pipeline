package com.example.cognitive_ems;

import java.nio.ByteBuffer;

public interface ImageViewCallback {

    void updateImageView(byte[] bytes);


    void updateTextMainActivity(String message);

}
