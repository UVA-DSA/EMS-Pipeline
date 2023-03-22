package com.example.camstrm;

public class ImageData {
    int seq,height,width,byte_len;
    long timestamp;
    byte[] data;
    public ImageData(int seq,int height,int width, int byte_len, byte[] data, long timestamp){
        this.seq=seq;
        this.height=height;
        this.width=width;
        this.byte_len=byte_len;
        this.data=data;
        this.timestamp=timestamp;
    }
}
