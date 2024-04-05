package com.example.cognitive_ems;

import android.util.Log;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import java.util.concurrent.TimeUnit;

public class WebSocketClient {

    private String serverUrl;
    private WebSocket webSocket;
    private OkHttpClient wsClient;
    private WebSocketListener listener;
    private static WebSocketClient instance;

    private String TAG = "WebSocketClient";

    public static synchronized WebSocketClient getInstance(String serverUrl, WebSocketListener listener) {
        if (instance == null) {
            instance = new WebSocketClient(serverUrl, listener);
        }
        return instance;
    }

    private WebSocketClient(String serverUrl, WebSocketListener listener) {
        this.serverUrl = serverUrl;
        this.listener = listener;
        this.wsClient = new OkHttpClient.Builder()
                .retryOnConnectionFailure(true) // Automatically retry on connection failure
                .readTimeout(0, TimeUnit.MILLISECONDS) // Disable read timeout
                .build();
        connect(); // Auto-connect on instance creation
    }

    public void connect() {
        Request request = new Request.Builder().url(serverUrl).build();
        webSocket = wsClient.newWebSocket(request, new WebSocketListener() {
            @Override
            public void onOpen(WebSocket webSocket, okhttp3.Response response) {
                super.onOpen(webSocket, response);
                // Notify original listener

                listener.onOpen(webSocket, response);
            }

            @Override
            public void onMessage(WebSocket webSocket, String text) {
                super.onMessage(webSocket, text);
                // Notify original listener
                listener.onMessage(webSocket, text);
            }

            @Override
            public void onClosing(WebSocket webSocket, int code, String reason) {
                super.onClosing(webSocket, code, reason);
                // Notify original listener
                listener.onClosing(webSocket, code, reason);
                // Attempt to reconnect
                reconnect();
            }

            @Override
            public void onFailure(WebSocket webSocket, Throwable t, okhttp3.Response response) {
                super.onFailure(webSocket, t, response);
                // Notify original listener
                listener.onFailure(webSocket, t, response);
                // Attempt to reconnect
                reconnect();
            }
        });
    }

    private void reconnect() {
        try {
            // Wait for a certain duration before reconnecting, to avoid rapid reconnect attempts
            Thread.sleep(5000); // Wait for 5 seconds before retrying
            Log.d(TAG, "Reconnecting to " + serverUrl);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        connect(); // Reconnect
    }

    public void sendMessage(String message) {
        if (webSocket != null) {
            webSocket.send(message);
            Log.d(TAG, "Sent message: ");
        }
    }

    public void close() {
        if (webSocket != null) {
            webSocket.close(1000, "Closing Connection");
        }
        if (!wsClient.dispatcher().executorService().isShutdown()) {
            wsClient.dispatcher().executorService().shutdown();
        }
    }
}
