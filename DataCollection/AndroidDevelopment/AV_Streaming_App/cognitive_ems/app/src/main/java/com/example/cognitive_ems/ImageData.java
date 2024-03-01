package com.example.cognitive_ems;

public class ImageData {
    int seq,height,width,byte_len;

    private static int instance=0;
    long timestamp;
    byte[] data;
    public ImageData(int seq,int height,int width, int byte_len, byte[] data, long timestamp){
        this.seq=++instance;
        this.height=height;
        this.width=width;
        this.byte_len=byte_len;
        this.data=data;
        this.timestamp=timestamp;
    }
}
