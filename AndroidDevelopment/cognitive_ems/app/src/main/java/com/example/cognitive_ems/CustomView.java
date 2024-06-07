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

public class CustomView extends View {
    private Paint rectanglePaint;
    private Paint objectPaint;
    private Paint objectStrRectPaint;
    private List<CustomRectangle> customRects; //list of custom rectangles to be displayed onscreen
    private Rect objectStrRect;//filled rectangle for object identification text

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
    }
    //MAY NEED TO MOVE THESE WITH ADDITION OF CUSTOM RECTANGLE CLASS
    public void clearCustomRect() {
        this.customRect = null; // Clear the custom rectangle
        invalidate(); // Trigger a redraw to remove the rectangle
    }

    public void setCustomRect(Rect rect, String object) {
        //check if this exact rectangle already exists in the list of custom rectangles, so as not to trigger too many redraws
        for rect in customRects:
            if (rect.getRectangle().equals(rect)){
                return;
        }
        CustomRectangle customRect = new CustomRectangle(rect, object); //create a new custom rectangle from incoming feedback data
        customRects.add(customRect); //add the new custom rectangle to the list of custom rectangles to be displayed
        
        invalidate(); // Trigger a redraw when customRect is updated
    }

    public void setProtocolBox(String str, TextView protocolBox){
        //System.out.println("Made it to setProtocolBox!");
        protocolBox.setText(str);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (this.customRect != null) {
            // Draw a rectangle on the TextureView
            for rect in customRects:
                canvas.drawRect(rect.getRectangle, rectanglePaint);
                //This creates the background for the object name and confidence level to be displayed on. For better graphics, this should be resized relative to the text size
                objectStrRect = new Rect(customRect.left - 4, customRect.top - 20, customRect.left + 200, customRect.top);
                //Draw object name and confidence level on top-left corner of rectangle
                canvas.drawRect(objectStrRect, objectStrRectPaint);
                canvas.drawText(rect.getObjectStr, customRect.getRectangle.left, customRect.getRectangle.top, objectPaint);
        }
    }
}
