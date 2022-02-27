package com.rsahel.deboggler

import android.Manifest
import android.content.res.ColorStateList
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

enum class ProcessResult {
    DicesNotFound,
    BlobsNotMerged,
    FrameNotFound,
    IndividualDicesNotFound,
    PROCESS_FAILURE,
    BoardIsolated,
    DicesFound,
    BlobsMerged,
    FrameFound,
    CornersFound,
    Warped,
    WarpedAndIsolated,
    WarpedAndIsolatedAndCleaned,
    IndividualDicesFound,
    IndividualDicesFoundAndMerged,
    PROCESS_SUCCESS,
    PROCESS_SUCCESS_INDECISIVE;

    companion object {
        fun fromInt(value: Int) = values().first { it.ordinal == value }
    }
};

class CameraFragment : Fragment(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var _binding: FragmentCameraBinding? = null
    private val binding get() = _binding!!

    private var cvCameraView: CameraBridgeViewBase? = null

    private val processOnBackgroundThread = true

    private var guessedChars = CharArray(16)
    private var previousResult = "";
    private var isResultFound = false
    private var backgroundThread: Thread? = null
    private var screenshotIndex = 0
    private var screenshotNext = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? {
        _binding = FragmentCameraBinding.inflate(inflater, container, false)
        binding.fab.setOnClickListener {
            findNavController().navigate(R.id.action_CameraFragment_to_SolutionFragment)
        }
        binding.screenshotFab.setOnClickListener {
            screenshotNext = true
        }

        if (!LibraryLoaded) {
            val neuralNetworkFile = File(requireActivity().filesDir, "neuralNetwork.bin");
            if (!neuralNetworkFile.exists()) {
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
            }

            if (!OpenCVLoader.initDebug()) {
                Log.d(TAG,
                    "Internal OpenCV library not found. Using OpenCV Manager for initialization")
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
        binding.progressbar.setProgress(0, true)
        binding.progressbar.max = 3

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
        if (_binding == null) {
            return src
        }
        if (screenshotNext) {
            val lastFrame = File(requireActivity().filesDir, "$screenshotIndex.jpg")
            var dst = Mat()
            org.opencv.imgproc.Imgproc.cvtColor(src, dst, org.opencv.imgproc.Imgproc.COLOR_RGB2BGR)
            org.opencv.imgcodecs.Imgcodecs.imwrite(lastFrame.absolutePath, dst)
            screenshotIndex++
            screenshotNext = false
        }

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
        val intStatus = deboggle(src.nativeObjAddr, guessedChars)
        val status = ProcessResult.fromInt(intStatus)
        val color = when (status) {
            ProcessResult.DicesNotFound -> R.color.grid_idle_color
            ProcessResult.BlobsNotMerged -> R.color.grid_idle_color
            ProcessResult.FrameNotFound -> R.color.grid_idle_color
            ProcessResult.IndividualDicesNotFound -> R.color.red
            ProcessResult.PROCESS_FAILURE -> R.color.red
            ProcessResult.BoardIsolated -> R.color.grid_idle_color
            ProcessResult.DicesFound -> R.color.grid_idle_color
            ProcessResult.BlobsMerged -> R.color.grid_idle_color
            ProcessResult.FrameFound -> R.color.grid_idle_color
            ProcessResult.CornersFound -> R.color.grid_idle_color
            ProcessResult.Warped -> R.color.grid_idle_color
            ProcessResult.WarpedAndIsolated -> R.color.grid_idle_color
            ProcessResult.WarpedAndIsolatedAndCleaned -> R.color.grid_idle_color
            ProcessResult.IndividualDicesFound -> R.color.grid_idle_color
            ProcessResult.IndividualDicesFoundAndMerged -> R.color.grid_idle_color
            ProcessResult.PROCESS_SUCCESS -> R.color.on_color
            ProcessResult.PROCESS_SUCCESS_INDECISIVE -> R.color.red
        }
        Handler(requireContext().mainLooper).post {
            _binding?.grid?.backgroundTintList =
                ColorStateList.valueOf(resources.getColor(color, null))
        }

        var resetProgress = false
        if (status == ProcessResult.PROCESS_SUCCESS) {
            val result = String(guessedChars)
            if (result == previousResult) {
                Handler(requireContext().mainLooper).post {
                    _binding?.progressbar?.setProgress(binding.progressbar.progress + 1, true)
                }
            } else {
                resetProgress = true
                previousResult = result
            }
            if (_binding != null && binding.progressbar.progress >= binding.progressbar.max - 1) {
                isResultFound = true
                Handler(requireContext().mainLooper).postAtFrontOfQueue {
                    findNavController().navigate(R.id.action_CameraFragment_to_SolutionFragment,
                        bundleOf("result" to previousResult))
                }
            }
        }

        if (resetProgress) {
            Handler(requireContext().mainLooper).post {
                _binding?.progressbar?.setProgress(0, true);
            }
        }

    }

    override fun onCameraViewStarted(width: Int, height: Int) {
    }

    override fun onCameraViewStopped() {
    }

    private external fun deboggle(srcAddr: Long, result: CharArray): Int
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