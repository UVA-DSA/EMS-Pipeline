package com.example.cognitive_ems;

import android.graphics.Rect;
import android.util.Log;
import android.widget.TextView;

import androidx.constraintlayout.solver.widgets.Rectangle;

import java.util.List;

public class CustomViewManager {
    private static CustomViewManager instance;
    private CustomView customView;

    private String TAG = "CustomViewManager";

    private CustomViewManager() {
        // Private constructor to prevent instantiation
    }

    public static CustomViewManager getInstance() {
        if (instance == null) {
            instance = new CustomViewManager();
        }
        return instance;
    }

    public void setOverlayView(CustomView overlayView) {
        this.customView = overlayView;
    }

    public void updateRectangle(Rect customRect, String object) {
        if (customView != null) {
            System.out.println("Updating rectangle in CVM!");
            customView.setCustomRect(customRect, object);
        }
    }

    public void updateRectangleList(List<CustomRectangle> customRects) {
        if (customView != null) {
            System.out.println("Updating rectangle in CVM!");
            customView.setCustomRectangleList(customRects);
        }
    }
    public void updateProtocolBox(String str, TextView protocolBox) {
        Log.d(TAG, "updating protocol box with: " + str);
        customView.setProtocolBox(str,protocolBox);
    }

    public void updateActionLogBox(String str, TextView actionLogBox){
        System.out.println("I am in update Action Log Box!");
        customView.setActionLogBox(str, actionLogBox);
    }
    //Removes all current displayed rectangles
    public void clearRectangles() {
        customView.clearCustomRects();
    }

}
