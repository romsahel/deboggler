package com.rsahel.deboggler

import android.animation.ArgbEvaluator
import android.animation.ValueAnimator
import android.content.Context.MODE_PRIVATE
import android.content.Intent
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.activity.viewModels
import androidx.core.animation.doOnRepeat
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.google.android.material.math.MathUtils
import com.rsahel.deboggler.databinding.FragmentSolutionBinding


class SolutionFragment : Fragment() {

    private var _binding: FragmentSolutionBinding? = null
    private val binding get() = _binding!!

    private lateinit var letters: Array<TextView>
    private lateinit var currentSolution: List<SolutionItem>
    private lateinit var viewModel: SolutionViewModel
    private var animationIndex = -1
    private var selectedSolution: SolutionItem? = null
    private var onColor: Int = 0
    private var offColor: Int = 0
    private lateinit var onAnimator: ValueAnimator

    private var expanded = false
    private val listExpandedSize = 0
    private var listNonExpandedSize = -1

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?,
    ): View? {
        _binding = FragmentSolutionBinding.inflate(inflater, container, false)
        binding.fab.setOnClickListener {
            findNavController().navigate(R.id.action_SolutionFragment_to_CameraFragment)
        }
        binding.fabExpand.setOnClickListener {
            expandBoard(!expanded)
        }

        letters = Array(16) {
            when (it) {
                12 -> binding.boggle0
                8 -> binding.boggle1
                4 -> binding.boggle2
                0 -> binding.boggle3

                13 -> binding.boggle4
                9 -> binding.boggle5
                5 -> binding.boggle6
                1 -> binding.boggle7

                14 -> binding.boggle8
                10 -> binding.boggle9
                6 -> binding.boggle10
                2 -> binding.boggle11

                15 -> binding.boggle12
                11 -> binding.boggle13
                7 -> binding.boggle14
                3 -> binding.boggle15
                else -> null as TextView
            }
        }
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val mainActivity = requireActivity() as MainActivity
        viewModel = mainActivity.viewModels<SolutionViewModel>().value

        val recyclerView = binding.recyclerView
        val adapter = SolutionListAdapter(
            { solution -> onSelectedSolutionChanged(solution) },
            { solution, clickType -> onDictionaryClicked(solution, clickType) }
        )
        recyclerView.adapter = adapter
        viewModel.getSolutions().observe(mainActivity) { solutions ->
            adapter.submitList(solutions)
        }
        initializeAnimator()
        selectedSolution = null
        onAnimator.pause()

        Thread {
            val result =
                if (arguments != null) requireArguments().getString("result")!! else savedResult
            if (arguments != null) savedResult = result

            currentSolution = mainActivity.solutioner.findSolutions(result.lowercase())
            Handler(requireContext().mainLooper).post {
                for (i in letters.indices) {
                    letters[i].text = result[i].toString()
                }
                viewModel.loadSolutions(currentSolution)
            }
        }.start()
    }

    private fun initializeAnimator() {
        offColor = resources.getColor(R.color.white, null)
        onColor = resources.getColor(R.color.on_color, null)
        onAnimator = ValueAnimator.ofObject(ArgbEvaluator(), offColor, onColor)
        onAnimator.duration = 350
        onAnimator.addUpdateListener { updatedAnimation ->
            if (animationIndex >= 0 && selectedSolution != null) {
                letters[selectedSolution!!.indices[animationIndex]].backgroundTintList =
                    ColorStateList.valueOf(updatedAnimation.animatedValue as Int)
            }
        }
        onAnimator.doOnRepeat { _ ->
            animationIndex++
            if (selectedSolution == null || animationIndex >= selectedSolution!!.indices.size) {
                animationIndex = -1
            }
        }
        onAnimator.repeatMode = ValueAnimator.RESTART
        onAnimator.repeatCount = -1
    }

    private fun expandBoard(expand: Boolean) {
        expanded = expand

        if (listNonExpandedSize <= 0) {
            listNonExpandedSize = binding.recyclerView.measuredHeight
        }

        val boardStartSize = binding.boggleContainer.scaleX
        val boardEndSize = if (expanded) 1.5f else 1.0f

        val listStartSize = binding.recyclerView.measuredHeight
        val listEndSize = if (expanded) listExpandedSize else listNonExpandedSize

        val valueAnimator = ValueAnimator.ofFloat(0.0f, 1.0f)
        valueAnimator.duration = 250L
        valueAnimator.addUpdateListener {
            var params = binding.recyclerView.layoutParams
            val size = MathUtils.lerp(listStartSize.toFloat(),
                listEndSize.toFloat(),
                valueAnimator.animatedValue as Float).toInt()
            params.height = size
            binding.recyclerView.layoutParams = params

            val scale =
                MathUtils.lerp(boardStartSize, boardEndSize, valueAnimator.animatedValue as Float)
            binding.boggleContainer.scaleX = scale
            binding.boggleContainer.scaleY = scale
        }
        valueAnimator.start()
    }

    private var savedResult: String
        get() {
            val settings = requireActivity().getSharedPreferences("Solution", MODE_PRIVATE)
            return settings.getString("SavedResult", "LRFTAGLENETAITSA")!!
        }
        set(value) {
            val settings = requireActivity().getSharedPreferences("Solution", MODE_PRIVATE)
            settings.edit().putString("SavedResult", value).commit()
        }

    private fun onDictionaryClicked(solution: SolutionItem, clickType: ClickType) {
        if (clickType == ClickType.Regular)
        {
            startActivity(Intent(Intent.ACTION_VIEW,
                Uri.parse("https://www.larousse.fr//dictionnaires//francais//${solution.value}")))
        }
        else
        {
            startActivity(Intent(Intent.ACTION_VIEW,
                Uri.parse("https://www.cnrtl.fr/definition/${solution.value}")))
        }
    }

    private fun onSelectedSolutionChanged(solution: SolutionItem) {
        if (solution == selectedSolution) return

        for (i in letters.indices) {
            letters[i].backgroundTintList =
                if (solution.indices.contains(i)) ColorStateList.valueOf(onColor)
                else ColorStateList.valueOf(offColor)
        }

        selectedSolution = solution
        animationIndex = -1
        onAnimator.start()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
        viewModel.clearSolutions()
    }

    companion object {
        private const val TAG = "Deboggler_FirstFragment"
    }
}