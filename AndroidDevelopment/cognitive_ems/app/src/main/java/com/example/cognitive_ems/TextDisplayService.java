package com.example.cognitive_ems;


import static java.lang.Integer.parseInt;

import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;


import org.json.JSONArray;
import org.json.JSONObject;
import org.w3c.dom.Text;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class TextDisplayService {

    private String TAG = "TextDisplayService";
    private static TextDisplayService instance;
    private CustomViewManager cvm;
    private TextView protocolBox, actionLogBox;

    private String action1 = " ", action2 = " ", action3 = " ";


    public  TextDisplayService(){
    }

    public static TextDisplayService getInstance(){
        if (instance == null) {
            instance = new TextDisplayService();
        }
        return instance;
    }

    public void setProtocolBox(TextView pbox){
        if (pbox != null) {
            this.protocolBox = pbox;
        }
    }

    public void setActionLogBox(TextView abox){
        if (abox != null){
            this.actionLogBox = abox;
        }
    }

/*
Assuming protocol feedback comes in the form: {\"type\":\"Protocol\",\"protocol\":\"medical - knee pain - MCL suspected (protocol 2 - 1)\",\"protocol_confidence\":0.0209748435020447}
Assuming Object Detection feedback comes in the form: {"type":"detection","box_coords":[[70,0],[512,511]],"obj_name":"person","confidence":"0.96"}
 */

    protected void objectFeedbackParser(String args, Size size) {
        try {
            Log.d(TAG, "Object feedback: " + args);

//            cvm.clearRectangles();

            JSONArray jsonArray = new JSONArray(args);
            List<String> listObjects = new ArrayList<>();
            List<CustomRectangle> listOfRectangles = new ArrayList<>();

            for (int i = 0; i < jsonArray.length(); i++) {
                JSONObject jsonObject = jsonArray.getJSONObject(i);
                listObjects.add(jsonObject.toString());
            }


            for (String feedback : listObjects) {
                JSONObject jsonObject = new JSONObject(feedback);

                String objName = jsonObject.getString("obj_name");
                float confidence = Float.parseFloat(jsonObject.getString("confidence"));
                JSONArray boxCoords = jsonObject.getJSONArray("box_coords");
                JSONArray minCoords = boxCoords.getJSONArray(0);
                JSONArray maxCoords = boxCoords.getJSONArray(1);

                int minX = minCoords.getInt(0);
                int minY = minCoords.getInt(1);
                int maxX = maxCoords.getInt(0);
                int maxY = maxCoords.getInt(1);

                // Image dimensions
                int imageWidth = 640;
                int imageHeight = 480;

                // Rotate coordinates 90 degrees counterclockwise
//                minX = minY;
//                minY = imageWidth - maxX;
//                maxX = maxY;
//                maxY = imageWidth - minX;

                // Scaling factors
                float scaleX = size.getWidth() / imageWidth;
                float scaleY = size.getHeight() / imageHeight;

                // Apply scaling
                minX = Math.round(minX * scaleX / 4.5f);
                minY = Math.round(minY * scaleY / 6);
                maxX = Math.round(maxX * scaleX / 4.5f);
                maxY = Math.round(maxY * scaleY / 6);

                Log.d("ObjectFeedback", "minX: " + minX + " minY: " + minY + " maxX: " + maxX + " maxY: " + maxY);

                if (minY < 20) {
                    minY = 40;
                }

                Rect rect = new Rect(minX, minY, maxX, maxY);
                String objectString = objName + ": " + confidence;
                Paint paint = new Paint();
                switch (objName){
                    case "hands":
                        paint.setColor(Color.BLUE);
                        break;
                    case "bvm" :
                        paint.setColor(Color.RED);
                        break;
                    case "defib pads":
                        paint.setColor(Color.MAGENTA);
                        break;
                    case "dummy":
                        paint.setColor(Color.GREEN);
                        break;
                    default:
                        paint.setColor(Color.WHITE);
                        break;
                }
                paint.setStyle(Paint.Style.STROKE); // Set style to stroke
                paint.setStrokeWidth(5);// Set stroke width

                listOfRectangles.add(new CustomRectangle(rect, objectString, paint));

//                cvm.getInstance().updateRectangle(rect, objectString);

            }

            // send the list to get updated
            CustomViewManager.getInstance().updateRectangleList(listOfRectangles);

        } catch (Exception e) {
            System.out.println("Could not understand feedback : " + e);
        }
    }
    protected void actionParser(String action){
        try {
            action3 = action2;
            action2 = action1;
            action1 = action;
            String actionString = action3 + "\n" + action2 + "\n" + action1;
            cvm.getInstance().updateActionLogBox(actionString, actionLogBox);
        } catch (Exception e) {
            System.out.println("Could not understand action : " + e);
        }
    }

    protected void protocolFeedbackParser(String feedback){
        try{
            String protocolDisplayStr = feedback.substring(feedback.indexOf(":", 10) + 2, feedback.indexOf("(") + 16) + " - " + feedback.substring(feedback.indexOf("confidence") + 12, feedback.indexOf("confidence") + 16);
            cvm.getInstance().updateProtocolBox(protocolDisplayStr, protocolBox);
        } catch (Exception e) {
            System.out.println("Could not understand Protocol feedback : " + e);
        }
    }

}
