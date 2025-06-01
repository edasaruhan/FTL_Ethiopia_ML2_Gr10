package com.example.wastesorting

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.firebase.analytics.FirebaseAnalytics
import com.google.firebase.analytics.ktx.analytics
import com.google.firebase.analytics.ktx.logEvent
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import java.nio.MappedByteBuffer
import java.io.FileInputStream
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var previewView: PreviewView
    private lateinit var classifyButton: Button
    private lateinit var resultText: TextView
    private lateinit var disposalTipText: TextView
    private lateinit var tflite: Interpreter
    private lateinit var firebaseAnalytics: FirebaseAnalytics
    private val db = Firebase.firestore
    private val categories = listOf("paper", "plastic", "metal", "glass", "cardboard", "trash")
    private val CAMERA_REQUEST_CODE = 100

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.camera_preview)
        classifyButton = findViewById(R.id.classify_button)
        resultText = findViewById(R.id.result_text)
        disposalTipText = findViewById(R.id.disposal_tip_text)

        // Initialize Firebase Analytics
        firebaseAnalytics = Firebase.analytics

        // Initialize TensorFlow Lite
        try {
            tflite = Interpreter(loadModelFile())
        } catch (e: Exception) {
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }

        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()

        // Request camera permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_REQUEST_CODE
            )
        }

        // Set up classify button
        classifyButton.setOnClickListener {
            // Classification handled in ImageAnalysis
        }

        // Check for model updates (example: store model version in Firestore)
        checkModelUpdate()
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("waste_classifier.tflite")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            // Image analysis
            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                processImage(imageProxy)
            }

            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
            } catch (exc: Exception) {
                Toast.makeText(this, "Failed to start camera: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toBitmap()
        if (bitmap != null) {
            val result = classifyImage(bitmap)
            runOnUiThread {
                resultText.text = getString(R.string.classification_result, result)
                updateDisposalTip(result)
                logClassificationEvent(result)
            }
        }
        imageProxy.close()
    }

    private fun ImageProxy.toBitmap(): Bitmap? {
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = android.graphics.YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = java.io.ByteArrayOutputStream()
        yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
        val imageBytes = out.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun classifyImage(bitmap: Bitmap): String {
        // Resize and preprocess image
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val inputBuffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4)
        inputBuffer.order(ByteOrder.nativeOrder())
        
        val pixels = IntArray(224 * 224)
        resizedBitmap.getPixels(pixels, 0, 224, 0, 0, 224, 224)
        for (pixel in pixels) {
            inputBuffer.putFloat(((pixel shr 16) and 0xFF) / 255.0f) // R
            inputBuffer.putFloat(((pixel shr 8) and 0xFF) / 255.0f)  // G
            inputBuffer.putFloat((pixel and 0xFF) / 255.0f)          // B
        }

        // Run inference
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        tflite.run(inputBuffer, output)

        // Get predicted class
        val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: 0
        return categories[maxIndex]
    }

    private fun updateDisposalTip(category: String) {
        val tipResId = when (category) {
            "paper" -> R.string.disposal_tip_paper
            "plastic" -> R.string.disposal_tip_plastic
            "metal" -> R.string.disposal_tip_metal
            "glass" -> R.string.disposal_tip_glass
            "cardboard" -> R.string.disposal_tip_cardboard
            "trash" -> R.string.disposal_tip_trash
            else -> R.string.disposal_tip_trash
        }
        disposalTipText.text = getString(tipResId)
    }

    private fun logClassificationEvent(category: String) {
        firebaseAnalytics.logEvent(FirebaseAnalytics.Event.SELECT_ITEM) {
            param(FirebaseAnalytics.Param.ITEM_ID, category)
            param(FirebaseAnalytics.Param.ITEM_NAME, "waste_classification")
            param(FirebaseAnalytics.Param.CONTENT_TYPE, "classification_result")
        }
    }

    private fun checkModelUpdate() {
        // Example: Check Firestore for model version
        db.collection("model_updates").document("current_version")
            .get()
            .addOnSuccessListener { document ->
                if (document.exists()) {
                    val version = document.getString("version") ?: "1.0"
                    // In a real app, compare version and download new model if needed
                    Toast.makeText(this, "Model version: $version", Toast.LENGTH_SHORT).show()
                }
            }
            .addOnFailureListener { e ->
                Toast.makeText(this, "Failed to check model updates: ${e.message}", Toast.LENGTH_SHORT).show()
            }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_REQUEST_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(this, R.string.camera_permission_denied, Toast.LENGTH_LONG).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tflite.close()
    }

    companion object {
        private const val NUM_CLASSES = 6
    }
}