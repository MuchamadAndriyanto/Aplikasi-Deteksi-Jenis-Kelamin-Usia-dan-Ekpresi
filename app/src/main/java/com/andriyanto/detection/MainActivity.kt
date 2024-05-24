package com.andriyanto.detection

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.andriyanto.detection.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var agenderInterpreter: Interpreter
    private lateinit var expressionInterpreter: Interpreter

    companion object {
        private const val SELECT_PHOTO_REQUEST_CODE = 1
        private const val OPEN_CAMERA_REQUEST_CODE = 2
        private const val CAMERA_PERMISSION_REQUEST_CODE = 3
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        try {
            agenderInterpreter = Interpreter(loadModelFile("agender_cnn_model (4).tflite"))
            expressionInterpreter = Interpreter(loadModelFile("expression_cnn_model (4).tflite"))
        } catch (e: IOException) {
            e.printStackTrace()
        }

        binding.selectImageButton.setOnClickListener {
            selectImageFromGallery()
        }

        binding.openCameraButton.setOnClickListener {
            openCamera()
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun selectImageFromGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, SELECT_PHOTO_REQUEST_CODE)
    }

    private fun openCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_REQUEST_CODE)
        } else {
            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(intent, OPEN_CAMERA_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST_CODE -> {
                if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                    openCamera()
                } else {
                    Toast.makeText(this, "Camera permission is required to use the camera", Toast.LENGTH_SHORT).show()
                }
                return
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        when (requestCode) {
            SELECT_PHOTO_REQUEST_CODE -> {
                if (resultCode == Activity.RESULT_OK) {
                    val imageUri: Uri? = data?.data
                    if (imageUri != null) {
                        val inputStream: InputStream? = contentResolver.openInputStream(imageUri)
                        val bitmap = BitmapFactory.decodeStream(inputStream)
                        binding.imageView.setImageBitmap(bitmap)

                        // Perform face detection
                        detectFace(bitmap)
                    }
                }
            }
            OPEN_CAMERA_REQUEST_CODE -> {
                if (resultCode == Activity.RESULT_OK) {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    binding.imageView.setImageBitmap(imageBitmap)

                    // Perform face detection
                    detectFace(imageBitmap)
                }
            }
        }
    }

    private fun detectFace(bitmap: Bitmap) {
        val image = InputImage.fromBitmap(bitmap, 0)
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_NONE)
            .build()
        val detector = FaceDetection.getClient(options)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    binding.ageGenderTextView.text = "No face detected"
                    binding.expressionTextView.text = ""
                } else {
                    // Perform predictions
                    predictAgeGender(bitmap)
                    predictExpression(bitmap)
                }
            }
            .addOnFailureListener { e ->
                e.printStackTrace()
                binding.ageGenderTextView.text = "Face detection failed"
                binding.expressionTextView.text = ""
            }
    }

    private fun predictAgeGender(bitmap: Bitmap) {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 56, 56, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * 56 * 56 * 1)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap to grayscale and normalize
        for (y in 0 until 56) {
            for (x in 0 until 56) {
                val px = resizedBitmap.getPixel(x, y)
                val r = (px shr 16 and 0xFF).toFloat()
                inputBuffer.putFloat(r / 255.0f)
            }
        }

        inputBuffer.rewind()

        val genderOutput = Array(1) { FloatArray(1) }
        val ageOutput = Array(1) { FloatArray(1) }

        agenderInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), mapOf(0 to genderOutput, 1 to ageOutput))

        val gender = if (genderOutput[0][0] < 0.5) "Male" else "Female"
        val age = when (Math.round(ageOutput[0][0])) {
            0 -> "<18"
            1 -> "18-30"
            2 -> "30-50"
            3 -> "50+"
            else -> "Unknown"
        }

        binding.ageGenderTextView.text = "Age: $age\nGender: $gender"
    }

    private fun predictExpression(bitmap: Bitmap) {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 56, 56, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * 56 * 56 * 1)
        inputBuffer.order(ByteOrder.nativeOrder())

        // Convert bitmap to grayscale and normalize
        for (y in 0 until 56) {
            for (x in 0 until 56) {
                val px = resizedBitmap.getPixel(x, y)
                val r = (px shr 16 and 0xFF).toFloat()
                inputBuffer.putFloat(r / 255.0f)
            }
        }

        inputBuffer.rewind()

        val expressionOutput = Array(1) { FloatArray(7) }
        expressionInterpreter.run(inputBuffer, expressionOutput)

        val expressionLabels = arrayOf("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
        val maxIndex = expressionOutput[0].indices.maxByOrNull { expressionOutput[0][it] } ?: -1

        binding.expressionTextView.text = "Expression: ${expressionLabels[maxIndex]}"
    }
}
