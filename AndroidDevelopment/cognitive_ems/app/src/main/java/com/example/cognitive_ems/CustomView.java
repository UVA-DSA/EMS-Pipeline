package com.example.cognitive_ems;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.view.TextureView;
import android.view.View;

public class CustomView extends View {
    private Paint rectanglePaint;
    private Rect customRect; // Store the custom location and size

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
        rectanglePaint.setStrokeWidth(5); // Set stroke width
    }

    public void clearCustomRect() {
        this.customRect = null; // Clear the custom rectangle
        invalidate(); // Trigger a redraw to remove the rectangle
    }

    public void setCustomRect(Rect rect) {
        this.customRect = rect;
        invalidate(); // Trigger a redraw when customRect is updated
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if(this.customRect != null)
            // Draw a rectangle on the TextureView
            canvas.drawRect(this.customRect, rectanglePaint);
    }
}
