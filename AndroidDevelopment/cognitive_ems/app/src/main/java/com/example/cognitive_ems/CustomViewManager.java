package com.example.cognitive_ems;

import android.graphics.Rect;
import android.widget.TextView;

public class CustomViewManager {
    private static CustomViewManager instance;
    private CustomView customView;

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
            customView.setCustomRect(customRect, object);
        }
    }
    public void updateProtocolBox(String str, TextView protocolBox) {
        System.out.println("I am in update ProtocolBox!");
        System.out.println("updating protocol box with: " + str);
        customView.setProtocolBox(str,protocolBox);
    }
    public void clearRectangle() {
        customView.clearCustomRect();
    }
}
