package com.example.cognitive_ems;

import android.media.AudioFormat;

public enum PayloadType {
    RAW_16BIT(127, 2, Compression.NONE),
    RAW_8BIT(126, 1, Compression.NONE),
    ZIP_16BIT(125, 2, Compression.ZIP),
    ZIP_8BIT(124, 1, Compression.ZIP);

    public enum Compression { NONE, ZIP}

    public final int payloadTypeId;
    public final int sampleByteSize;
    public final Compression compression;

    PayloadType(int payloadTypeId, int sampleByteSize, Compression compression) {
        this.payloadTypeId = payloadTypeId;
        this.sampleByteSize = sampleByteSize;
        this.compression = compression;
    }

    public int getAudioFormat() {
        switch (sampleByteSize) {
            case 1:
                return AudioFormat.ENCODING_PCM_8BIT;
            case 2:
                return AudioFormat.ENCODING_PCM_16BIT;
            default:
                return AudioFormat.ENCODING_INVALID;
        }
    }
}
