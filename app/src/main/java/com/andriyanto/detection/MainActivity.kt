package com.andriyanto.detection

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.andriyanto.detection.databinding.ActivityMainBinding
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
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

        // Initialize TensorFlow Lite interpreters
        try {
            agenderInterpreter = Interpreter(loadModelFile("agender_cnn_model (6).tflite"))
            expressionInterpreter = Interpreter(loadModelFile("expression_cnn_model (4).tflite"))
        } catch (e: IOException) {
            e.printStackTrace()
        }

        // Set listeners for buttons
        binding.selectImageButton.setOnClickListener {
            selectImageFromGallery()
        }

        binding.openCameraButton.setOnClickListener {
            openCamera()
        }
    }

    // Load the TensorFlow Lite model file from assets
    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Select an image from the gallery
    private fun selectImageFromGallery() {
        val intent = Intent(Intent.ACTION_PICK)
        intent.type = "image/*"
        startActivityForResult(intent, SELECT_PHOTO_REQUEST_CODE)
    }

    // Open the camera to take a picture
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

    // Handle permission request results
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

    // Handle activity results for selecting an image or taking a picture
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

                        // Detect faces in the selected image
                        detectFace(bitmap)
                    }
                }
            }
            OPEN_CAMERA_REQUEST_CODE -> {
                if (resultCode == Activity.RESULT_OK) {
                    val imageBitmap = data?.extras?.get("data") as Bitmap
                    binding.imageView.setImageBitmap(imageBitmap)

                    // Detect faces in the taken picture
                    detectFace(imageBitmap)
                }
            }
        }
    }

    // Detect faces and draw rectangles and points
    private fun detectFace(bitmap: Bitmap) {
        val image = InputImage.fromBitmap(bitmap, 0)
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
            .build()
        val detector = FaceDetection.getClient(options)

        detector.process(image)
            .addOnSuccessListener { faces ->
                if (faces.isEmpty()) {
                    binding.ageGenderTextView.text = "No face detected"
                    binding.expressionTextView.text = ""
                } else {
                    // Draw rectangles and points on detected faces
                    val bitmapWithRectanglesAndPoints = drawRectanglesAndPointsOnBitmap(bitmap, faces)
                    binding.imageView.setImageBitmap(bitmapWithRectanglesAndPoints)

                    // Predict age, gender, and expression
                    predictAgeGender(bitmapWithRectanglesAndPoints)
                    predictExpression(bitmapWithRectanglesAndPoints)
                }
            }
            .addOnFailureListener { e ->
                e.printStackTrace()
                binding.ageGenderTextView.text = "Face detection failed"
                binding.expressionTextView.text = ""
            }
    }

    // Draw rectangles and points on detected faces
    private fun drawRectanglesAndPointsOnBitmap(bitmap: Bitmap, faces: List<Face>): Bitmap {
        val resultBitmap = bitmap.copy(bitmap.config, true)
        val canvas = Canvas(resultBitmap)
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }

        // Different colors for different face parts
        val pointPaints = mapOf(
            FaceContour.FACE to Paint().apply { color = Color.BLUE; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.LEFT_EYEBROW_TOP to Paint().apply { color = Color.GREEN; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.LEFT_EYEBROW_BOTTOM to Paint().apply { color = Color.GREEN; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.RIGHT_EYEBROW_TOP to Paint().apply { color = Color.YELLOW; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.RIGHT_EYEBROW_BOTTOM to Paint().apply { color = Color.YELLOW; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.LEFT_EYE to Paint().apply { color = Color.CYAN; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.RIGHT_EYE to Paint().apply { color = Color.CYAN; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.UPPER_LIP_TOP to Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.UPPER_LIP_BOTTOM to Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.LOWER_LIP_TOP to Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.LOWER_LIP_BOTTOM to Paint().apply { color = Color.MAGENTA; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.NOSE_BRIDGE to Paint().apply { color = Color.RED; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.NOSE_BOTTOM to Paint().apply { color = Color.RED; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.LEFT_CHEEK to Paint().apply { color = Color.LTGRAY; style = Paint.Style.FILL; strokeWidth = 4f },
            FaceContour.RIGHT_CHEEK to Paint().apply { color = Color.LTGRAY; style = Paint.Style.FILL; strokeWidth = 4f }
        )

        // Iterate over detected faces
        for (face in faces) {
            val bounds = face.boundingBox
            canvas.drawRect(bounds, paint)

            // Draw points for facial landmarks with different colors
            face.allContours.forEach { contour ->
                val contourPaint = pointPaints[contour.faceContourType] ?: Paint().apply {
                    color = Color.BLUE
                    style = Paint.Style.FILL
                    strokeWidth = 4f
                }
                contour.points.forEach { point ->
                    canvas.drawCircle(point.x, point.y, 8f, contourPaint)
                }
            }

            // Draw an arrow on the nose
            val noseBottom = face.getContour(FaceContour.NOSE_BOTTOM)?.points
            if (noseBottom != null && noseBottom.isNotEmpty()) {
                val noseTip = noseBottom.first()
                val noseBase = noseBottom.last()
                drawArrow(canvas, noseTip.x, noseTip.y, noseBase.x, noseBase.y)
            }
        }

        return resultBitmap
    }

    // Draw an arrow
    private fun drawArrow(canvas: Canvas, startX: Float, startY: Float, endX: Float, endY: Float) {
        val paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.STROKE
            strokeWidth = 8f
        }

        // Draw line
        canvas.drawLine(startX, startY, endX, endY, paint)

        // Draw arrow head
        val arrowHeadLength = 20f
        val angle = Math.atan2((startY - endY).toDouble(), (startX - endX).toDouble())
        val angle1 = angle + Math.PI / 6
        val angle2 = angle - Math.PI / 6

        val arrowX1 = (endX + arrowHeadLength * Math.cos(angle1)).toFloat()
        val arrowY1 = (endY + arrowHeadLength * Math.sin(angle1)).toFloat()
        val arrowX2 = (endX + arrowHeadLength * Math.cos(angle2)).toFloat()
        val arrowY2 = (endY + arrowHeadLength * Math.sin(angle2)).toFloat()

        canvas.drawLine(endX, endY, arrowX1, arrowY1, paint)
        canvas.drawLine(endX, endY, arrowX2, arrowY2, paint)
    }

    // Predict age and gender
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

    // Predict facial expression
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
