package com.example.cognitive_ems;

public interface FeedbackCallback {

    void onObjectFeedbackReceived(String feedback);
    void onProtocolFeedbackReceived(String feedback);

    void onActionReceived(String action);

    void onResetReceived();

}

