package com.example.camstrm;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.media.Image;
import android.util.Log;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectOutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.util.Timer;
import java.util.TimerTask;

public class TcpClient {

    ImageViewCallback imageViewCallback = null;


    public static final String TAG = TcpClient.class.getSimpleName();
    private Context context = null;
    public  String SERVER_IP; //server IP address
    public  int SERVER_PORT;
    // message to send to the server
    private String mServerMessage;
    // sends message received notifications
    private OnMessageReceived mMessageListener = null;
    // while this is true, the server will continue running
    private boolean mRun = false;
    // used to send messages
    private PrintWriter mBufferOut;
    // used to read messages from the server
    private BufferedReader mBufferIn;

    private DataOutputStream mBufferImageOut;
    private Timer timer;
    private ObjectOutputStream oos;

    /**
     * Constructor of the class. OnMessagedReceived listens for the messages received from server
     */
    public TcpClient(OnMessageReceived listener, String SERVER_IP, int SERVER_PORT, Context context, ImageViewCallback imageViewCallback) {
        mMessageListener = listener;
        this.SERVER_IP = SERVER_IP;
        this.SERVER_PORT = SERVER_PORT;
        this.context = context;
        this.imageViewCallback = imageViewCallback;
    }


    /**
     * Sends the message entered by client to the server
     *
     * @param message text entered by client
     */
    public void sendMessage(final String message) {
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                if (mBufferOut != null) {
                    Log.d(TAG, "Sending: " + message);
                    mBufferOut.println(message);
//                    mBufferOut.write(message,0,message.length());
                    mBufferOut.flush();
                }
            }
        };
        Thread thread = new Thread(runnable);
        thread.start();
    }

    private TimerTaskk timerTask = new TimerTaskk(context);
        public class TimerTaskk extends TimerTask {
            Context ctxObject = null;

            public TimerTaskk(Context ctx) {
                ctxObject = ctx;
            }

            @Override
            public void run() {
                if (mBufferImageOut != null) {

                    try {


                        if (Camera2Service.img_list.size() > 0) {
                            Log.d(TAG, "Sending Image");


                            ImageData image = Camera2Service.img_list.remove(0);


                            mBufferImageOut.write(22);

                            mBufferImageOut.writeInt(image.seq);
                            mBufferImageOut.writeLong(image.timestamp);
                            mBufferImageOut.writeInt(image.width);
                            mBufferImageOut.writeInt(image.byte_len);
                            mBufferImageOut.write(image.data);
                            mBufferImageOut.flush();

                        }


                    } catch (IOException e) {
                        e.printStackTrace();

                    }

                }
            }
        }

    public void start() {
        if(timer != null) {
            return;
        }
        timer = new Timer();
        timer.scheduleAtFixedRate(timerTask, 5000, 50);
    }

    public void stop() {
        timer.cancel();
        timer = null;
    }


    public int sendImage(final ImageData image) {
//        Runnable runnable = new Runnable() {
//            @Override
//            public void run() {
                if (mBufferImageOut != null && image != null) {
                    Log.d(TAG, "Sending Image");
                    if(imageViewCallback != null){
                        imageViewCallback.updateImageView(image.data);
                    }


                    try {
                        mBufferImageOut.write(22);
//                        mBufferImageOut.writeInt(image.seq);
                        mBufferImageOut.writeLong(image.timestamp);
//                        mBufferImageOut.writeInt(image.width);
                        mBufferImageOut.writeLong(image.byte_len);
                        mBufferImageOut.write(image.data);
                        mBufferImageOut.flush();
                        return 0;

                    } catch (IOException e) {
                        e.printStackTrace();
                        return -1;
                    }

                }
                return 0;

    }


    /**
     * Close the connection and release the members
     */
    public void stopClient() {

        mRun = false;

        if (mBufferOut != null) {
            mBufferOut.flush();
            mBufferOut.close();
        }

        mMessageListener = null;
        mBufferIn = null;
        mBufferOut = null;
        mServerMessage = null;
        mBufferImageOut = null;
    }

    public void run() {
        Log.e("Video TCP Client", "Executing run()");

        mRun = true;

        try {
            //here you must put your computer's IP address.

            InetAddress serverAddr = InetAddress.getByName(SERVER_IP);
            Socket socket = null;
            Log.d("Video TCP Client", "C: Connecting...");

            while (true){
                try{
                    socket = new Socket(serverAddr, SERVER_PORT);
                    Log.d("Video TCP Client", "C: Connected!");
                    break;
                }
                catch(Exception e){
                    Thread.sleep(100);
                    Log.d("Video TCP Client", "C: Retrying!");
                }
            }



//
//            while(true){
//
//                try{
//                    //create a socket to make the connection with the server
//                    socket = new Socket(serverAddr, SERVER_PORT);
//                    Log.d("TCP Client", "C: Connected!");
//                    break;
//                }catch (IOException e){
//                    Thread.sleep(1000);
//                    Log.d("TCP Client", "C: Retrying...");
//
//                }
//            }



            try {

                //sends the message to the server
//                mBufferOut = new PrintWriter(new BufferedWriter(new OutputStreamWriter(socket.getOutputStream())), true);
//                mBufferImageOut = new DataOutputStream(socket.getOutputStream());
//                  oos = new ObjectOutputStream(socket.getOutputStream());
                 mBufferImageOut = new DataOutputStream(
                        new BufferedOutputStream(
                                socket.getOutputStream()));
                //receives the message which the server sends back
                mBufferIn = new BufferedReader(new InputStreamReader(socket.getInputStream()));

                int status = 0;
                //in this while the client listens for the messages sent by the server
                while (mRun) {

                    if(Camera2Service.img_list.size() > 0){

                        ImageData img = Camera2Service.img_list.remove(Camera2Service.img_list.size()-1);



                        status = sendImage(img);


                        Camera2Service.img_list.clear();

                        if(status == -1){
                            socket.close();

                        }

                    }
                }

                Log.d("Video TCP Client", "S: Received Message: '" + mServerMessage + "'");

            } catch (Exception e) {
                Log.e("Video TCP Client", "S: Error", e);
            } finally {
                //the socket must be closed. It is not possible to reconnect to this socket
                // after it is closed, which means a new socket instance has to be created.
                socket.close();
            }

        } catch (Exception e) {
            Log.e("Video TCP Client", "C: Error", e);
        }

    }

    //Declare the interface. The method messageReceived(String message) will must be implemented in the Activity
    //class at on AsyncTask doInBackground
    public interface OnMessageReceived {
        public void messageReceived(byte[] message);
    }



}