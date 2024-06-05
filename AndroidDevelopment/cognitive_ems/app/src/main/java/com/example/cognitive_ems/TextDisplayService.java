package com.example.cognitive_ems;


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


    protected String feedbackParser(Object args) {
        try{
            String feedback = args.toString(); //Sent as python dict, unable to parse in current form, so identify as string
            if (feedback.contains("\"type\":\"Protocol")){ //determining if feedback is Protocol type
                String protocolDisplayStr =  feedback.substring(feedback.indexOf(":", 10) + 2, feedback.indexOf("(")+16) + " - " + feedback.substring(feedback.indexOf("confidence")+12,feedback.indexOf("confidence")+16);
                System.out.println("This is the cvm instance!: " + cvm.getInstance());
                cvm.getInstance().updateProtocolBox(protocolDisplayStr, protocolBox);
                return protocolDisplayStr;
            } //TODO:Put else-if here for display feedback, routing to box display
        } catch (Exception e) {
            System.out.println("Could not understand feedback : " + e);
        }
        return null;
    }

}
