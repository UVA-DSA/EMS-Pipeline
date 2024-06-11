package com.example.cognitive_ems;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.os.Looper;
import android.util.AttributeSet;
import android.view.TextureView;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;

import androidx.constraintlayout.solver.widgets.Rectangle;

import java.net.SocketOption;
import java.util.ArrayList;
import java.util.List;

public class CustomView extends View {
    private Paint rectanglePaint;
    private Paint objectPaint;
    private Paint objectStrRectPaint;
    private List<CustomRectangle> customRects; //list of custom rectangles to be displayed onscreen
    private Rect objectStrRect;//filled rectangle for object identification text
    private Integer maxRectangles;//maximum rectangles allowed on screen at a time

    public CustomView(Context context) {
        super(context);
        init();
    }

    public CustomView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public CustomView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }

    private void init() {
        // Initialize the Paint for drawing the rectangle
        rectanglePaint = new Paint();
        rectanglePaint.setColor(Color.RED); // Set rectangle color to red
        rectanglePaint.setStyle(Paint.Style.STROKE); // Set style to stroke
        rectanglePaint.setStrokeWidth(5);// Set stroke width
        objectPaint = new Paint();
        objectPaint.setColor(Color.WHITE);//Set text color to black
        objectPaint.setStrokeWidth(2);//Set stroke width
        objectPaint.setTextSize(24);
        objectStrRectPaint = new Paint();
        objectStrRectPaint.setColor(Color.RED);
        objectStrRectPaint.setStyle(Paint.Style.FILL);
        customRects = new ArrayList<>();
        maxRectangles = 4; //arbitrary maximum number of rectangles allowed on screen
    }
    //MAY NEED TO EDIT THESE WITH ADDITION OF CUSTOM RECTANGLE CLASS
    public void clearCustomRects() {
        //clears all current rectangles
        customRects = new ArrayList<>();

        invalidate(); // Trigger a redraw to remove the rectangle
    }

    public void setCustomRect(Rect rect, String object) {
        //check if this exact rectangle already exists in the list of custom rectangles, so as not to trigger too many redraws
        if (customRects.size() > maxRectangles){System.out.println("hitting limit of rectangles on screen! cannot add"); return; }
        for (CustomRectangle rectangle:customRects) {
            if (rectangle.getRectangle().equals(rect)) {
                System.out.println("This rectangle already exists!");
                return;
            }
        }
            CustomRectangle newRect = new CustomRectangle(rect, object); //create a new custom rectangle from incoming feedback data
            customRects.add(newRect); //add the new custom rectangle to the list of custom rectangles to be displayed
            System.out.println("customRects list now has " + customRects.size() + "rectangles");
            invalidate(); // Trigger a redraw when customRect is updated
    }

    public void setProtocolBox(String str, TextView protocolBox){
        //System.out.println("Made it to setProtocolBox!");
        protocolBox.setText(str);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        System.out.println("Starting new redraw in onDraw");
        super.onDraw(canvas);
            // Draw a rectangle on the TextureView
            for (CustomRectangle rect:customRects) {
                canvas.drawRect(rect.getRectangle(), rectanglePaint);
                //This creates the background for the object name and confidence level to be displayed on. For better graphics, this should be resized relative to the text size
                objectStrRect = new Rect(rect.getRectangle().left - 4, rect.getRectangle().top - 20, rect.getRectangle().left + 200, rect.getRectangle().top);
                //Draw object name and confidence level on top-left corner of rectangle
                canvas.drawRect(objectStrRect, objectStrRectPaint);
                canvas.drawText(rect.getObjectStr(), rect.getRectangle().left, rect.getRectangle().top, objectPaint);
            }
    }
}