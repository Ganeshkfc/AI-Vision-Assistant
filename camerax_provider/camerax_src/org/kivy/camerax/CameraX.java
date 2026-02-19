package org.kivy.camerax;

import java.io.File;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.ExecutionException;
import com.google.common.util.concurrent.ListenableFuture;
import org.kivy.android.PythonActivity;
import android.app.Activity;
import android.util.Size;
import android.util.Rational;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.content.Context;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.net.Uri;
import android.provider.MediaStore;
import androidx.core.content.ContextCompat;
import androidx.camera.core.Camera;
import androidx.camera.core.Preview;
import androidx.camera.core.AspectRatio;
import androidx.camera.core.ImageCapture; 
import androidx.camera.core.ImageAnalysis;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.core.UseCaseGroup; 
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ViewPort;
import androidx.lifecycle.ProcessLifecycleOwner;
import androidx.lifecycle.LifecycleOwner;

// Note: VideoCapture is removed here because the older core.VideoCapture 
// is incompatible with API 33/AndroidX. 
// We focus on Photo + Analysis for your Vision Assistant.

public class CameraX {
    private boolean photo;
    private boolean video; // Keep variable for logic compatibility
    private boolean analysis;
    private int lensFacing;
    private int [] cameraResolution;  
    private int aspectRatio;
    private CallbackWrapper callbackClass;
    private int flashMode;
    private int imageOptimize;
    private float zoomScaleFront = 0.0f;
    private float zoomScaleBack = 0.0f;
    private int dataFormat;

    public static Executor executor = Executors.newSingleThreadExecutor();
    private ProcessCameraProvider cameraProvider = null;
    private Preview preview = null;
    private ImageCapture imageCapture = null;
    private Camera camera = null;
    private KivySurfaceProvider kivySurfaceProvider = null;
    private UseCaseGroup useCaseGroup = null;
    private boolean imageIsReady = false;
    private int viewPortWidth;
    private int viewPortHeight;

    public CameraX(boolean photo, boolean video, boolean analysis, String facing,
                   int[] resolution, String aspect_ratio, CallbackWrapper callback_class,
                   String flash, String optimize, float zoom_scale, String data_format) {
        this.photo = photo;
        this.video = false; // Forced false to prevent VideoCapture compile errors
        this.analysis = analysis;
        this.lensFacing = facing.equals("front") ? CameraSelector.LENS_FACING_FRONT : CameraSelector.LENS_FACING_BACK;
        this.cameraResolution = resolution;
        this.aspectRatio = aspect_ratio.equals("16:9") ? AspectRatio.RATIO_16_9 : AspectRatio.RATIO_4_3;
        this.callbackClass = callback_class;
        this.dataFormat = data_format.equals("rgba") ? ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888 : ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888;
    }

    public void startCamera() {
        Context context = PythonActivity.mActivity;
        final ListenableFuture<ProcessCameraProvider> cameraProviderFuture = ProcessCameraProvider.getInstance(context);
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                configureCamera();
            } catch (Exception e) { e.printStackTrace(); }
        }, ContextCompat.getMainExecutor(context));
    }

    private void configureCamera() {
        int rotation = PythonActivity.mActivity.getWindowManager().getDefaultDisplay().getRotation();
        
        // Preview
        preview = new Preview.Builder().setTargetAspectRatio(aspectRatio).build();

        // Image Analysis
        ImageAnalysis imageAnalysis = null;
        if (analysis) {
            imageAnalysis = new ImageAnalysis.Builder()
                .setTargetAspectRatio(aspectRatio)
                .setOutputImageFormat(dataFormat)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
            imageAnalysis.setAnalyzer(executor, new ImageAnalysisAnalyzer(callbackClass));
        }

        // Image Capture
        if (photo) {
            imageCapture = new ImageCapture.Builder()
                .setTargetAspectRatio(aspectRatio)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build();
        }

        UseCaseGroup.Builder ucgb = new UseCaseGroup.Builder().addUseCase(preview);
        if (analysis) ucgb.addUseCase(imageAnalysis);
        if (photo) ucgb.addUseCase(imageCapture);
        
        useCaseGroup = ucgb.build();
        bindPreview();
    }

    private void bindPreview() {
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(lensFacing).build();
        cameraProvider.unbindAll();
        camera = cameraProvider.bindToLifecycle((LifecycleOwner)PythonActivity.mActivity, cameraSelector, useCaseGroup);
        
        callbackClass.callback_config(new Rect(0,0,viewPortWidth,viewPortHeight), new Size(viewPortWidth, viewPortHeight), 0);
    }

    public void setTexture(int texture_id, int[] size) {
        this.viewPortWidth = size[0];
        this.viewPortHeight = size[1];
        kivySurfaceProvider = new KivySurfaceProvider(texture_id, ContextCompat.getMainExecutor(PythonActivity.mActivity), size[0], size[1]);
        preview.setSurfaceProvider(kivySurfaceProvider);
    }
    
    public boolean imageReady() {
        if (imageIsReady) {
            if (kivySurfaceProvider != null) kivySurfaceProvider.KivySurfaceTextureUpdate();
            return true;
        }
        return false;
    }
}
