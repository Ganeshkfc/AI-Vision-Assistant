package org.kivy.camerax;

import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.annotation.NonNull;

public class ImageAnalysisAnalyzer implements ImageAnalysis.Analyzer {
    private CallbackWrapper callback_wrapper;

    public ImageAnalysisAnalyzer(CallbackWrapper callback_wrapper) {    
        this.callback_wrapper = callback_wrapper;
    }

    @Override
    public void analyze(@NonNull ImageProxy image) {
        this.callback_wrapper.callback_image(image); 
    }
}
