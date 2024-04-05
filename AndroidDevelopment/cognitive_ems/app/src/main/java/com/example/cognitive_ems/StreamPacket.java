package com.example.cognitive_ems;


import java.util.Arrays;

public class StreamPacket{

    //size of the header:
    final static int HEADER_SIZE = 12;

    //Fields that compose the header
    private final byte FIRST_BYTE = (byte)0x80;
    private byte payloadType;
    private short sequenceNumber;
    private int timeStamp;
    private int samplingFrequency;

    //Bitstream of the header
    private byte[] header;

    //size of the payload
    private int payloadSize;
    //Bitstream of the payload
    private byte[] payload;

    //--------------------------
    //Constructor of an RTPpacket object from header fields and payload bitstream
    //--------------------------
    public StreamPacket(byte payloadType, short sequenceNumber, int timeStamp,int samplingFrequency, byte[] data, int dataSize){
        payload = Arrays.copyOf(data, dataSize);
        payloadSize = dataSize;

        //build the header bistream:
        header = new byte[HEADER_SIZE];

        //fill the header array of byte with RTP header fields
        header[0] = FIRST_BYTE;
        header[1] = payloadType;
        header[2] = (byte)(sequenceNumber >> 8);
        header[3] = (byte)(sequenceNumber & 0xFF);
        header[4] = (byte)(timeStamp >> 24);
        header[5] = (byte)(timeStamp >> 16);
        header[6] = (byte)(timeStamp >> 8);
        header[7] = (byte)(timeStamp & 0xFF);
        header[8] = (byte)(samplingFrequency >> 24);
        header[9] = (byte)(samplingFrequency >> 16);
        header[10] = (byte)(samplingFrequency >> 8);
        header[11] = (byte)(samplingFrequency & 0xFF);

    }
    //--------------------------
    //getlength: return the total length of the RTP packet
    //--------------------------
    public int getPacketLength() {
        return(payloadSize + HEADER_SIZE);
    }

    //--------------------------
    //getpacket: returns the packet bitstream and its length
    //--------------------------
    public int getPacket(byte[] packet)
    {
        //construct the packet = header + payload
        for (int i=0; i < HEADER_SIZE; i++)
            packet[i] = header[i];
        for (int i=0; i < payloadSize; i++)
            packet[i+HEADER_SIZE] = payload[i];

        //return total size of the packet
        return(payloadSize + HEADER_SIZE);
    }
}