package com.rsahel.deboggler

import android.Manifest
import android.os.Bundle
import android.os.Handler
import android.util.Log
import android.view.*
import android.widget.ProgressBar
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.os.bundleOf
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.rsahel.deboggler.databinding.FragmentCameraBinding
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import java.io.File
import java.io.FileOutputStream

class CameraFragment : Fragment(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private lateinit var progressBar: ProgressBar
    private var cvCameraView: CameraBridgeViewBase? = null

    private var previousResult = "";
    private var isResultFound = false
    private var backgroundThread: Thread? = null
    private val processOnBackgroundThread = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        binding.fab.setOnClickListener {
            findNavController().navigate(R.id.action_CameraFragment_to_SolutionFragment)
        }

        if (!LibraryLoaded) {
            val neuralNetworkFile = File(requireActivity().filesDir, "neuralNetwork.bin");
//            if (!neuralNetworkFile.exists()) {
                requireActivity().assets.open("neuralNetwork.bin").use { input ->
                    val outputStream = FileOutputStream(neuralNetworkFile)
                    Log.e(TAG, "Writing to " + neuralNetworkFile.absolutePath)
                    outputStream.use { output ->
                        val buffer = ByteArray(4 * 1024) // buffer size
                        while (true) {
                            val byteCount = input.read(buffer)
                            if (byteCount < 0) break
                            output.write(buffer, 0, byteCount)
                        }
                        output.flush()
                    }
                }
//            }

            if (!OpenCVLoader.initDebug()) {
                Log.d(TAG,"Internal OpenCV library not found. Using OpenCV Manager for initialization")
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, context, cvLoaderCallback)
            } else {
                Log.d(TAG, "OpenCV library found inside package. Using it!")
                cvLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
            }
        }
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        progressBar = binding.progressbar
        progressBar.setProgress(0, true)
        progressBar.max = 5

        val openCvCameraView = binding.mainSurface
        cvCameraView = openCvCameraView
        openCvCameraView.visibility = SurfaceView.VISIBLE
        openCvCameraView.setCvCameraViewListener(this)
        val permissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                cvCameraView?.setCameraPermissionGranted()
            }
        }

        permissionLauncher.launch(Manifest.permission.CAMERA)
        requireActivity().window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onResume() {
        super.onResume()
        cvCameraView?.enableView()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        requireActivity().window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
    }

    override fun onCameraFrame(frame: CameraBridgeViewBase.CvCameraViewFrame): Mat {
        val src = frame.rgba()
//        val lastFrame = File(requireActivity().filesDir, "0.jpg")
//        org.opencv.imgcodecs.Imgcodecs.imwrite(lastFrame.absolutePath, src)

        if (processOnBackgroundThread) {
            if (!isResultFound && (backgroundThread == null || !backgroundThread!!.isAlive)) {
                val processed = src.clone()
                backgroundThread = Thread {
                    processImage(processed)
                    processed.release()
                }
                backgroundThread!!.start()
            }
            return src
        } else {
            if (!isResultFound) {
                processImage(src)
            }
            return src
        }
    }

    private fun processImage(src: Mat) {
        val result = deboggle(src.nativeObjAddr)
        if (result == previousResult && !result.isNullOrEmpty()) {
            Handler(requireContext().mainLooper).post {
                progressBar.setProgress(progressBar.progress + 1, true);
            }
        } else {
            previousResult = result.orEmpty()
            Handler(requireContext().mainLooper).post {
                progressBar.setProgress(0, true);
            }
        }
        if (progressBar.progress >= progressBar.max - 1) {
            isResultFound = true
            Handler(requireContext().mainLooper).postAtFrontOfQueue {
                findNavController().navigate(R.id.action_CameraFragment_to_SolutionFragment, bundleOf("result" to previousResult))
            }
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
    }

    override fun onCameraViewStopped() {
    }

    private external fun deboggle(srcAddr: Long): String?
    private external fun configureNeuralNetwork(path: String)

    private val cvLoaderCallback = object : BaseLoaderCallback(context) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    System.loadLibrary("native-lib")
                    LibraryLoaded = true
                    val neuralNetworkFile = File(activity!!.filesDir, "neuralNetwork.bin");
                    configureNeuralNetwork(neuralNetworkFile.absolutePath);
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    companion object {
        private const val TAG = "Deboggler_SecondFragment"
        private var LibraryLoaded = false
    }
}