package com.example.cognitive_ems;

public interface FeedbackCallback {

    void onFeedbackReceived(String feedback);

    void onActionReceived(String action);

}
