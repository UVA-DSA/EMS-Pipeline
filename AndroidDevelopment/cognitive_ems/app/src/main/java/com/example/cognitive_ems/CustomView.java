package com.example.cognitive_ems;
import static java.lang.Float.parseFloat;

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
    private Integer numRectangles; //tracks number of rectangles currently displayed, because using .remove() will affect accurate .size() measurement

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
        objectPaint.setTextSize(28);
        objectStrRectPaint = new Paint();
        objectStrRectPaint.setColor(Color.RED);
        objectStrRectPaint.setStyle(Paint.Style.FILL);
        customRects = new ArrayList<>();
        maxRectangles = 3; //arbitrary maximum number of rectangles allowed on screen
        numRectangles = 0;
    }
    //MAY NEED TO EDIT THESE WITH ADDITION OF CUSTOM RECTANGLE CLASS
    public void clearCustomRects() {
        //clears all current rectangles
        customRects = new ArrayList<>();

        invalidate(); // Trigger a redraw to remove the rectangle

    }

    public void setCustomRectangleList(List<CustomRectangle> customRects) {

        this.customRects = new ArrayList<>();;
        numRectangles = 0;
        for (CustomRectangle rectangle : customRects) {
            if(numRectangles < maxRectangles)
                this.customRects.add(rectangle);
            numRectangles++;
        }
        invalidate(); // Trigger a redraw to remove the rectangle

    }

    public void setCustomRect(Rect rect, String object) {
        System.out.println("Calling setCustomRect. Current state of customRects: " + customRects.toString());

        String objectName = object.substring(0, object.indexOf(": "));
        float newConfidence = Float.parseFloat(object.substring(object.indexOf(": ") + 2));

        CustomRectangle existingRectangle = null;
        for (CustomRectangle rectangle : customRects) {
            System.out.println("Comparing rectangle " + objectName + " to " + rectangle.getObjectName());
            if (rectangle.getObjectName().equals(objectName)) {
                System.out.println("Same type of object!");
                existingRectangle = rectangle;
                break;
            }
        }

        if (existingRectangle != null) {
            if (existingRectangle.getConfidence() > newConfidence) {
                return; // Don't add new rectangle if same type as another rectangle, keep only the highest confidence one
            } else {
                System.out.println("This object already exists with lower confidence, removing existing one.");
                customRects.remove(existingRectangle);
                numRectangles--;
            }
        }

        CustomRectangle newRect = new CustomRectangle(rect, object); // Create a new custom rectangle from incoming feedback data
        customRects.add(newRect); // Add the new custom rectangle to the list of custom rectangles to be displayed
        numRectangles++;
        System.out.println("customRects list now is " + customRects.toString());

        while (numRectangles > maxRectangles) {
            customRects.remove(0);
            numRectangles--;
        }

        System.out.println("Adding rectangle: " + object);
        invalidate(); // Trigger a redraw when customRect is updated
    }

    public void setProtocolBox(String str, TextView protocolBox){
        //System.out.println("Made it to setProtocolBox!");
        protocolBox.setText(str);
    }

    public void setActionLogBox(String str, TextView actionLogBox){
        System.out.println("Made it to set Action Log Box in CV");
        actionLogBox.setText(str);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        System.out.println("Starting new redraw in onDraw, with custom rects as follows" + customRects.toString());
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
