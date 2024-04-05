package com.example.cognitive_ems;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.provider.Settings;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_ALL = 1;
    private String[] permissions = {
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.WAKE_LOCK,
            Manifest.permission.ACCESS_NETWORK_STATE
    };

    private boolean permissionsGranted = false;

    protected static final String TAG = "MainActivity";
    public static String serverip;
    public static int audioport;
    public static String socketio_url;
    private Context mContext;

    private AudioStreamService audioStreamService;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        LayoutInflater inflater = LayoutInflater.from(MainActivity.this); // or (LayoutInflater) getSystemService(Context.LAYOUT_INFLATER_SERVICE);

        View viewMyLayout = inflater.inflate(R.layout.activity_main, null);
        setContentView(viewMyLayout);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        Log.i(TAG, "in main");
        mContext = this;

        SharedPreferences preferences = getSharedPreferences("CognitiveEMSConfig", Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = preferences.edit();
        editor.putString("serverip", getString(R.string.server_ip));
        String socketio_uri = "http://" + getString(R.string.server_ip) + ":" + getString(R.string.socketio_port);
        editor.putString("socketio_uri", socketio_uri);
        editor.putInt("audioport", Integer.parseInt(getString(R.string.audio_server_port)));
        editor.apply();

        serverip = getString(R.string.server_ip);
        audioport = Integer.parseInt(getString(R.string.audio_server_port));

        Button btn1 = (Button) findViewById(R.id.btn1);
        btn1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                finish();
                System.exit(0);
            }
        });

        if (!hasPermissions(this, permissions)) {
            ActivityCompat.requestPermissions(this, permissions, PERMISSION_ALL);
        } else {
            permissionsGranted = true;
            Log.d(TAG, "Permission Granted");
            startCameraStreamActivity();
            startAudioStreamActivity();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == PERMISSION_ALL) {
            boolean allPermissionsGranted = true;
            for (int grantResult : grantResults) {
                if (grantResult != PackageManager.PERMISSION_GRANTED) {
                    allPermissionsGranted = false;
                    break;
                }
            }

            if (allPermissionsGranted) {
                permissionsGranted = true;
                Log.d(TAG, "Permission Granted");
            } else {
                showPermissionRationaleDialog();
            }
        }
    }

    private void startCameraStreamActivity() {
        if (permissionsGranted) {
            // New Handler to start the SecondActivity and close this MainActivity after some seconds.
            new Handler().postDelayed(new Runnable(){
                @Override
                public void run() {
                    // Create Intent to start the SecondActivity
                    Intent intent = new Intent(MainActivity.this, CameraStreamActivity.class);
                    MainActivity.this.startActivity(intent);
                }
            }, 1000);

        }
    }

    private void startAudioStreamActivity() {
        if (permissionsGranted) {
            audioStreamService = new AudioStreamService(mContext, serverip, audioport);
            audioStreamService.startStreaming();
        }
    }

    private void stopAudioStreamActivity() {
        if (audioStreamService != null) {
            audioStreamService.stopStreaming();
        }
    }

    public boolean hasPermissions(Context context, String... permissions) {
        if (context != null && permissions != null) {
            for (String permission : permissions) {
                if (ContextCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    Log.d(TAG, "hasPermissions: no permission for " + permission);
                    return false;
                }
            }
        }
        return true;
    }

    private void showPermissionRationaleDialog() {
        new AlertDialog.Builder(this)
                .setTitle("Permissions Required")
                .setMessage("This app requires camera and audio permissions to function properly. Please grant them in the app settings.")
                .setPositiveButton("App Settings", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        Intent intent = new Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                Uri.fromParts("package", getPackageName(), null));
                        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                        startActivity(intent);
                    }
                })
                .setNegativeButton(android.R.string.cancel, new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int which) {
                        dialog.dismiss();
                    }
                })
                .setIcon(android.R.drawable.ic_dialog_alert)
                .show();
    }
}
