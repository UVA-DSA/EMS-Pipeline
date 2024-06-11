package com.example.cognitive_ems;


import android.graphics.Rect;
import android.widget.TextView;


import org.w3c.dom.Text;

public class TextDisplayService {

    private static TextDisplayService instance;
    private CustomViewManager cvm;
    private TextView protocolBox, actionLogBox;

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
    protected void objectFeedbackParser(Object args) {
        try {
            String feedback = args.toString();
//            System.out.println(args.toString());//Sent as python dict, unable to parse in current form, so identify as string
//            if (feedback.contains("\"type\":\"Protocol")) { //determining if feedback is Protocol type
//                String protocolDisplayStr = feedback.substring(feedback.indexOf(":", 10) + 2, feedback.indexOf("(") + 16) + " - " + feedback.substring(feedback.indexOf("confidence") + 12, feedback.indexOf("confidence") + 16);
//                cvm.getInstance().updateProtocolBox(protocolDisplayStr, protocolBox);
//            } else if (feedback.contains("\"type\":\"detection")) {
                //CURRENTLY just truncating confidence, should perhaps round if more than 2 decimal places?
                Float confidence = Float.parseFloat(feedback.substring(feedback.indexOf("confidence") + 13, feedback.indexOf("confidence") + 17));
                System.out.println("Confidence is: " + confidence.toString());

                String objectString = feedback.substring(feedback.indexOf("name") + 7, feedback.indexOf("\"", feedback.indexOf("name") + 8)) + ":  " + confidence;

                Integer minX = Integer.parseInt(feedback.substring(feedback.indexOf("[[") + 2, feedback.indexOf(",", feedback.indexOf("[["))));
                Integer minY = Integer.parseInt(feedback.substring(feedback.indexOf(",", feedback.indexOf("[[")) + 1, feedback.indexOf("]", feedback.indexOf("[["))));
                Integer maxX = Integer.parseInt(feedback.substring(feedback.indexOf("],[") + 3, feedback.indexOf(",", feedback.indexOf("],[") + 3)));
                Integer maxY = Integer.parseInt(feedback.substring(feedback.indexOf(",",feedback.indexOf("],["))+4, feedback.indexOf("]]")));

                System.out.println("minX: " + minX + " minY: " +  minY + " maxX: " +  maxX + " maxY:  " + maxY);
                Rect rect = new Rect(minX, minY, maxX, maxY);
                cvm.getInstance().updateRectangle(rect, objectString);

        } catch (Exception e) {
            System.out.println("Could not understand feedback : " + e);
        }
    }

    protected void actionParser(String action){

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
