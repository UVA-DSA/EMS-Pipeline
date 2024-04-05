package com.example.cognitive_ems;

import android.graphics.Rect;

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

    public void updateRectangle(Rect customRect) {
        if (customView != null) {
            customView.setCustomRect(customRect);
        }
    }

    public void clearRectangle() {
        customView.clearCustomRect();
    }
}
