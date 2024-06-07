package com.example.cognitive_ems;
import android.graphics.Rect;
/*
    CustomRectangle class is used to store the rectangle object, its name and confidence in object detection. (Name and confidence are in objectStr in the format "hand - 0.99"). 
    It serves as a data structure for the CustomView class
    */


public class CustomRectangle {

    private Rectangle rectangle;
    private String objectStr;
    

    public CustomRectangle(Rectangle rectangle, String objectStr) {
        this.rectangle = rectangle;
        this.objectStr = objectStr;
    }

    public Rectangle getRectangle() {
        return rectangle;
    }

    public String getObjectStr() {
        return objectStr;
    }
    
    public boolean eqauls(CustomRectangle other) {
        return this.rectangle.equals(other.rectangle) && this.objectStr.equals(other.objectStr);
    } 



}