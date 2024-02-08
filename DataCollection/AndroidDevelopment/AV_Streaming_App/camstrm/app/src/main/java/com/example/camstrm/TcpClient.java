package com.example.camstrm;

import android.app.Activity;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.media.Image;
import android.os.IBinder;
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

import io.socket.emitter.Emitter;
import io.socket.engineio.parser.Base64;

public class TcpClient {

    ImageViewCallback imageViewCallback = null;


    public static final String TAG = TcpClient.class.getSimpleName();
    private Context context = null;
    public  String SERVER_IP; //server IP address
    public  int SERVER_PORT;
    // message to send to the server
    private String mServerMessage;
    // sends message received notifications
    // while this is true, the server will continue running
    private boolean mRun = false;
    private boolean start = false;
    // used to send messages
    private PrintWriter mBufferOut;
    // used to read messages from the server
    private BufferedReader mBufferIn;

    private DataOutputStream mBufferImageOut;
    private Timer timer;
    private ObjectOutputStream oos;

    private SocketIOService socketIOService;
    private SocketIOAdapter socketIOAdapter;
    private boolean mBound = false;

    /**
     * Constructor of the class. OnMessagedReceived listens for the messages received from server
     */
    public TcpClient(String SERVER_IP, int SERVER_PORT, Context context, ImageViewCallback imageViewCallback) {
        this.SERVER_IP = SERVER_IP;
        this.SERVER_PORT = SERVER_PORT;
        this.context = context;
        this.imageViewCallback = imageViewCallback;

        // Bind to LocalService.
        Intent intent = new Intent(this.context, SocketIOService.class);
        this.context.bindService(intent, connection, Context.BIND_AUTO_CREATE);

    }

//    public int sendImage(final ImageData image) {
//                if (mBufferImageOut != null && image != null) {
//                    Log.d(TAG, "Sending Image");
//                    if(imageViewCallback != null){
//                        imageViewCallback.updateImageView(image.data);
//                    }
//
//
//                    try {
//                        mBufferImageOut.write(22);
////                        mBufferImageOut.writeInt(image.seq);
//                        mBufferImageOut.writeLong(image.timestamp);
////                        mBufferImageOut.writeInt(image.width);
//                        mBufferImageOut.writeLong(image.byte_len);
//                        mBufferImageOut.write(image.data);
//                        mBufferImageOut.flush();
//                        return 0;
//
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                        return -1;
//                    }
//
//                }
//                return 0;
//
//    }


    public int sendImage(final ImageData image, SocketIOService client) {
        if (image != null) {
            Log.d(TAG, "Sending Image");
            try {

            String base64String = Base64.encodeToString(image.data, Base64.DEFAULT);
            client.sendVideo(base64String);
            Thread.sleep(33);

            } catch (Exception e) {
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

        mBufferIn = null;
        mBufferOut = null;
        mServerMessage = null;
        mBufferImageOut = null;
    }

    /** Defines callbacks for service binding, passed to bindService(). */
    private ServiceConnection connection = new ServiceConnection() {

        @Override
        public void onServiceConnected(ComponentName className,
                                       IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance.
            SocketIOService.SocketIOServiceBinder binder = (SocketIOService.SocketIOServiceBinder) service;
            socketIOService = binder.getService();
            Log.d("Video Stream Client", "Bounded to service!");
            mBound = true;
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mBound = false;
        }
    };


    public void run() {
        Log.e("Video TCP Client", "Executing run()");

        mRun = true;

        try {
            //here you must put your computer's IP address.




            // Usage example
//            SocketIOAdapter client = new SocketIOAdapter("http://172.27.150.146:3000");
//            client.sendMessage("Hello from Java!");



            try {

                int status = 0;
                String message = "";

                //in this while the client listens for the messages sent by the server
                while (mRun) {

                    if(mBound){



                    String command = socketIOService.getCommand();
                    Log.d("Video TCP Client", "C: Command "+ command );

                    String display ;
                    if(command.equals("start")){
                        start = true;
                        display = "Recording in Progress ...";
                    }else{
                        start = false;
                        display = "Recording Stopped";
                    }

                    if(!message.equals(command)) {
                        imageViewCallback.updateTextMainActivity(display);
                        message = command;
                    }

//                    Log.d("Video Client", "C: Start "+ start );
                    if(start){

                        if(Camera2Service.img_list.size() > 0){

                            ImageData img = Camera2Service.img_list.remove(Camera2Service.img_list.size()-1);

//                            if(imageViewCallback != null && img != null){
//                                imageViewCallback.updateImageView(img.data);
//                            }

                            status = sendImage(img, socketIOService);

                            Camera2Service.img_list.clear();

                        }

                    }

                }


                }


            } catch (Exception e) {
                Log.e("Video TCP Client", "S: Error", e);
            } finally {
                //the socket must be closed. It is not possible to reconnect to this socket
                // after it is closed, which means a new socket instance has to be created.
//                socket.close();
                // Optionally, you can add more messages or events here
//                socketIOService.disconnect();

            }

        } catch (Exception e) {
            Log.e("Video TCP Client", "C: Error", e);
        }

    }





}