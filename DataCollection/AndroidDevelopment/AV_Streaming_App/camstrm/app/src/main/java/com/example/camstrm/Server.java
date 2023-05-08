package com.example.camstrm;


import android.graphics.Bitmap;
import android.media.Image;
import android.util.Log;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.logging.Handler;


public class Server {
    protected static final String TAG = "cam_stream";
    public static volatile  Socket client = null;
    public static volatile  DataOutputStream out;
    public static volatile DataInputStream in;

    public static volatile int byte_len=0;
    public static volatile int height=12;
    public static volatile int width=188;
    public static volatile int seq=0;
    public static volatile ArrayList<ImageData> img_list=new ArrayList<ImageData>();
    public static volatile ArrayList<Image> raw_img_list=new ArrayList<Image>();
    public static volatile boolean ready=false;




    public void startServer(){
        Log.i(TAG,"Starting server...");
        new Thread(new ServerStart()).start();
    }

    static class ServerStart implements Runnable {
        public ServerSocket mSocketServer = null;
        public static final String SERVER_IP = "172.27.164.148"; //server IP address
        public static final int SERVER_PORT = 8899;
        public Socket tcpSocket = null;

        private PrintWriter mBufferOut;
        // used to read messages from the server
        private BufferedReader mBufferIn;

        //here you must put your computer's IP address.
        public InetAddress serverAddr = null;

        // while this is true, the server will continue running
        private boolean mRun = false;

        private String mServerMessage;





        @Override
        public void run() {


            try {
                serverAddr = InetAddress.getByName(SERVER_IP);

                Log.d("TCP Client", "C: Connecting...");


                Log.i(TAG, "Created data stream");

                //create a socket to make the connection with the server
                tcpSocket = new Socket(serverAddr, SERVER_PORT);

                //sends the message to the server
                out = new DataOutputStream(tcpSocket.getOutputStream());

                in = new DataInputStream(tcpSocket.getInputStream());

                ready = true;

//                Log.i(TAG, "Server Connection: "+ tcpSocket.isConnected() + in.readUTF());

                while (true) {
//                    Log.i(TAG, "Server Connection: "+ tcpSocket.isConnected());

//                    Log.i(TAG, "list size = " + String.valueOf(img_list.size()));
                    if (img_list.size() > 0) {
                        try {
                            Long time = System.currentTimeMillis();
                            Log.i(TAG, "Image List " + img_list.toArray().length);


                            if(!img_list.isEmpty()){
                                ImageData id = img_list.get(0);



                                Log.i(TAG, "list size = " + String.valueOf(img_list.size()));
                                Log.i(TAG, "Image Streamed ");


                                out.write(22);
                                out.writeInt(id.seq);
                                out.writeLong(id.timestamp);
                                out.writeInt(id.width);
                                out.writeInt(id.byte_len);
                                out.write(id.data);
                                out.flush();


                                Log.i(TAG, String.valueOf(id.byte_len));
                                Log.i(TAG, "done writing");

                                img_list.clear();

                            }


                        } catch (NullPointerException e) {
                            e.printStackTrace();
                        }
                    }
                }

            } catch (Exception e) {
                Log.e("TCP", "S: Error", e);
            } finally {
                //the socket must be closed. It is not possible to reconnect to this socket
                // after it is closed, which means a new socket instance has to be created.
//                try {
////                    tcpSocket.close();
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
            }

//            try{
//                mSocketServer = new ServerSocket(SERVER_PORT);
//            } catch (IOException e) {
//                Log.i(TAG, e.getMessage());
//            }
//
//            try {
//                Log.i(TAG, "connecting...");
//                client = mSocketServer.accept();
//                Log.i(TAG, "Connected! local port = " + client.getLocalPort());
//                Log.i(TAG,String.valueOf(client.isBound()));
//                out = new DataOutputStream(client.getOutputStream());
//                in = new DataInputStream(new BufferedInputStream(client.getInputStream()));
//                Log.i(TAG, "Created data stream");
//                out.write(100);
//            } catch (IOException e) {
//                Log.i(TAG, e.getMessage());
//            }
//
//
//            //wait till the client is ready to receive
//            Log.i(TAG, "Waiting till client is ready....");
//            try{
//                int begin = in.read();
//                Log.i(TAG,String.valueOf(begin));
//                while(begin!=23){
//                    begin=in.read();
//                    Log.i(TAG,String.valueOf(begin));
//                    ready=true;
//                }
//            }catch (IOException e) {
//                Log.i(TAG, e.getMessage());
//            }
//
//            while(true) {
//                Log.i(TAG,"list size = " + String.valueOf(img_list.size()));
//                if (img_list.size()>0) {
//                    try {
//                        ImageData id=img_list.remove(0);
//                        Log.i(TAG,"list size = " + String.valueOf(img_list.size()));
//                        out.write(22);
//                        out.writeInt(id.seq);
//                        out.writeInt(id.height);
//                        out.writeInt(id.width);
//                        out.writeInt(id.byte_len);
//                        Log.i(TAG,String.valueOf(id.byte_len));
//                        out.write(id.data);
//                        out.flush();
//                        Log.i(TAG,"done writing");
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }catch (NullPointerException e){
//                        e.printStackTrace();
//                    }
//                }
//            }
        }
    }
}
