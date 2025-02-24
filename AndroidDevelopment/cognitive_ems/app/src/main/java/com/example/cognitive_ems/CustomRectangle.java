package com.example.cognitive_ems;
import static java.lang.Float.parseFloat;

import android.graphics.Paint;
import android.graphics.Rect;

import androidx.constraintlayout.solver.widgets.Rectangle;

/*
    CustomRectangle class is used to store the rectangle object, its name and confidence in object detection. (Name and confidence are in objectStr in the format "hand - 0.99"). 
    It serves as a data structure for the CustomView class
    */


public class CustomRectangle {

    private Rect rectangle;
    private String objectStr;

    private Paint rectanglePaint;
    

    public CustomRectangle(Rect rectangle, String objectStr, Paint rectanglePaint) {
        this.rectangle = rectangle;
        this.objectStr = objectStr;
        this.rectanglePaint = rectanglePaint;
    }

    public Rect getRectangle() {
        return rectangle;
    }
    public Paint getRectanglePaint() {
        return rectanglePaint;
    }

    public String getObjectStr() {
        return objectStr;
    }


//    public boolean equals(CustomRectangle other) {
//        return this.rectangle.equals(other.rectangle) && this.objectStr.equals(other.objectStr);
//    }

    public String getObjectName() { return objectStr.substring(0, objectStr.indexOf(": "));}

    public Float getConfidence() { return parseFloat(objectStr.substring(objectStr.indexOf(": ") + 2));}



}