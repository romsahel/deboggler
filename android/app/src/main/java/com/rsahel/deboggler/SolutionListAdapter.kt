package com.rsahel.deboggler

import android.content.Intent
import android.net.Uri
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.view.animation.Animation
import android.view.animation.AnimationUtils
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import java.util.*

data class SolutionItem(
    val value: String,
    val indices: List<Int>,

    ) {
    override fun hashCode(): Int {
        return value.hashCode()
    }

    override fun equals(other: Any?): Boolean {
        return other != null && value == (other as SolutionItem).value
    }
}

public enum class ClickType
{
    Regular,
    Long
}

class SolutionListAdapter(
    private val onClick: (SolutionItem) -> Unit,
    private val onDictionaryClick: (SolutionItem, ClickType) -> Unit,
) :
    ListAdapter<SolutionItem, SolutionListAdapter.SolutionViewHolder>(SolutionItemDiffCallback) {

    var selected: SolutionItem? = null
    var selectedView: SolutionViewHolder? = null

    class SolutionViewHolder(
        itemView: View,
        val onClick: (SolutionViewHolder) -> Unit,
        val onDictionaryClick: (SolutionViewHolder, ClickType) -> Unit,
    ) :
        RecyclerView.ViewHolder(itemView) {

        lateinit var solution: SolutionItem

        val valueView: TextView = itemView.findViewById(R.id.solution_value)
        val scoreView: TextView = itemView.findViewById(R.id.solution_score)
        val dictionaryView: ImageView = itemView.findViewById(R.id.solution_dictionary)

        fun bind(item: SolutionItem) = with(itemView) {
            solution = item
            valueView.text = solution.value.replaceFirstChar {
                if (it.isLowerCase()) it.titlecase(Locale.getDefault()) else it.toString()
            }
            scoreView.text = (when (solution.value.length) {
                3, 4 -> 1
                5 -> 2
                6 -> 3
                7 -> 5
                else -> 11
            }).toString()
        }

        init {
            itemView.setOnClickListener {
                onClick(this)
            }
            dictionaryView.setOnClickListener {
                onDictionaryClick(this, ClickType.Regular)
            }
            dictionaryView.setOnLongClickListener {
                onDictionaryClick(this, ClickType.Long)
                return@setOnLongClickListener true
            }
        }

        fun select(select: Boolean) {
            val offColor = itemView.resources.getColorStateList(R.color.white, null)
            val onColor = itemView.resources.getColorStateList(R.color.on_color, null)
            if (select) {
                valueView.setTextColor(onColor)
                scoreView.backgroundTintList = onColor
                dictionaryView.visibility = View.VISIBLE
            } else {
                valueView.setTextColor(offColor)
                scoreView.backgroundTintList = offColor
                dictionaryView.visibility = View.GONE
            }
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): SolutionViewHolder {
        val view =
            LayoutInflater.from(parent.context).inflate(R.layout.solution_item, parent, false)
        return SolutionViewHolder(view, {
            if (selectedView != null) {
                selectedView!!.select(false)
            }

            selected = it.solution
            selectedView = it

            it.select(true)
            onClick(it.solution)
        }, { it: SolutionViewHolder, clickType: ClickType ->
            onDictionaryClick(it.solution, clickType)
        })
    }

    override fun onBindViewHolder(holder: SolutionViewHolder, position: Int) {
        holder.bind(getItem(position))
        val isSelected = selected != null && selected!!.value == holder.solution.value
        holder.select(isSelected)
        if (isSelected) {
            selectedView = holder
        } else if (selectedView == holder) {
            selectedView = null
        }
    }
}

object SolutionItemDiffCallback : DiffUtil.ItemCallback<SolutionItem>() {
    override fun areItemsTheSame(oldItem: SolutionItem, newItem: SolutionItem): Boolean {
        return oldItem == newItem
    }

    override fun areContentsTheSame(oldItem: SolutionItem, newItem: SolutionItem): Boolean {
        return oldItem.value == newItem.value
    }
}