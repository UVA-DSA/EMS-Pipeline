package com.example.cognitive_ems;


import static java.lang.Integer.parseInt;

import android.graphics.Rect;
import android.util.Log;
import android.util.Size;
import android.widget.TextView;


import org.w3c.dom.Text;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class TextDisplayService {

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
    protected void objectFeedbackParser(Object args, Size size) {
        try {
            args = args.substring(1, args.length()-1);
            String[] objects = args.split(Pattern.quote("},"));
            List<String> listObjects = new ArrayList<>();
            for (String object : objects) {
                listObjects.add(object);
            }
            System.out.println(listObjects.toString());
            for (String feedback : listObjects) {
                System.out.println("Looping through objects: " + feedback);
                //CURRENTLY just truncating confidence, should perhaps round if more than 2 decimal places?
                Float confidence = Float.parseFloat(feedback.substring(feedback.indexOf("confidence") + 13, feedback.indexOf("confidence") + 17));
                System.out.println("Confidence is: " + confidence.toString());

                String objectString = feedback.substring(feedback.indexOf("name") + 7, feedback.indexOf("\"", feedback.indexOf("name") + 8)) + ":  " + confidence;


                Integer minX = Integer.parseInt(feedback.substring(feedback.indexOf("[[") + 2, feedback.indexOf(",", feedback.indexOf("[["))));
                Integer minY = Integer.parseInt(feedback.substring(feedback.indexOf(",", feedback.indexOf("[[")) + 1, feedback.indexOf("]", feedback.indexOf("[["))));
                Integer maxX = Integer.parseInt(feedback.substring(feedback.indexOf("],[") + 3, feedback.indexOf(",", feedback.indexOf("],[") + 3)));
                Integer maxY = Integer.parseInt(feedback.substring(feedback.indexOf(",",feedback.indexOf("],[")+4) + 1, feedback.indexOf("]]")));

                // Scaling factors
                float scaleX = size.getWidth() / 640.0f;
                float scaleY = size.getHeight() / 480.0f;

                // Apply scaling
                minX = Math.round(minX * scaleX/4);
                minY = Math.round(minY * scaleY/4);
                maxX = Math.round(maxX * scaleX/4);
                maxY = Math.round(maxY * scaleY/4);


                Log.d("ObjectFeedback", "minX: " + minX + " minY: " +  minY + " maxX: " +  maxX + " maxY:  " + maxY);
                if (minY < 20){
                    minY = 40;
                }
                Rect rect = new Rect(minX, minY, maxX, maxY);
                cvm.getInstance().updateRectangle(rect, objectString);
            }

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
