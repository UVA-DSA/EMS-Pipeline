package com.example.cognitive_ems;


import android.graphics.Rect;
import android.widget.TextView;


import org.w3c.dom.Text;

public class TextDisplayService {

    private static TextDisplayService instance;
    private CustomViewManager cvm;
    private TextView protocolBox;

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

/*
Assuming protocol feedback comes in the form: {\"type\":\"Protocol\",\"protocol\":\"medical - knee pain - MCL suspected (protocol 2 - 1)\",\"protocol_confidence\":0.0209748435020447}
Assuming Object Detection feedback comes in the form: {\"type\":\"detection\",\"box_coords\":{\"center_point\":\"(500, 500)\", \"width\":\"200\",\"height\":\"300\"}, \"obj_name\":\"hand\", \"confidence\":\"0.897346927837\"}
IMPORTANT NOTE!!!!!!!!! THERE MUST BE A SPACE BETWEEN THE COORDINATE POINTS FOR THE BOX COORDINATE, OTHERWISE EDIT BELOW
 */
    protected void feedbackParser(Object args) {
        try {
            String feedback = args.toString(); //Sent as python dict, unable to parse in current form, so identify as string
            if (feedback.contains("\"type\":\"Protocol")) { //determining if feedback is Protocol type
                String protocolDisplayStr = feedback.substring(feedback.indexOf(":", 10) + 2, feedback.indexOf("(") + 16) + " - " + feedback.substring(feedback.indexOf("confidence") + 12, feedback.indexOf("confidence") + 16);
                cvm.getInstance().updateProtocolBox(protocolDisplayStr, protocolBox);
            } else if (feedback.contains("\"type\":\"detection")) {
                System.out.println(feedback);
                Float confidence = Float.parseFloat(feedback.substring(feedback.indexOf("confidence") + 13, feedback.indexOf("confidence") + 17));
                System.out.println("Confidence is: " + confidence.toString());
                String objectString = feedback.substring(feedback.indexOf("name") + 7, feedback.indexOf("\"", feedback.indexOf("name") + 8)) + ":  " + confidence;
                Integer width = Integer.parseInt(feedback.substring(feedback.indexOf("width") + 8, feedback.indexOf("\"", feedback.indexOf("width") + 9)));
                Integer height = Integer.parseInt(feedback.substring(feedback.indexOf("height") + 9, feedback.indexOf("\"", feedback.indexOf("height")+10)));
                Integer centerX = Integer.parseInt(feedback.substring(feedback.indexOf("point") + 9, feedback.indexOf(",", feedback.indexOf("point") + 9)));
                Integer centerY = Integer.parseInt(feedback.substring(feedback.indexOf(",",feedback.indexOf("point"))+2, feedback.indexOf(")", feedback.indexOf("point"))));
                System.out.println("width: " + width + " height: " + height + " centerX: " +  centerX + " centerY: " +  centerY);
                Rect rect = new Rect(centerX-width/2, centerY-height/2, centerX + width/2, centerY + height/2);
                cvm.getInstance().updateRectangle(rect, objectString);
            }
        } catch (Exception e) {
            System.out.println("Could not understand feedback : " + e);
        }
    }

}
