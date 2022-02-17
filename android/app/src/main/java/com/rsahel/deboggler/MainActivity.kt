package com.rsahel.deboggler

import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.findNavController
import androidx.navigation.ui.AppBarConfiguration
import androidx.navigation.ui.navigateUp
import androidx.navigation.ui.setupActionBarWithNavController
import android.view.Menu
import android.view.MenuItem
import android.view.WindowManager
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import com.rsahel.deboggler.databinding.ActivityMainBinding

class SolutionViewModel : ViewModel() {
    private val solutions: MutableLiveData<List<SolutionItem>> by lazy {
        MutableLiveData<List<SolutionItem>>()
    }

    fun getSolutions(): LiveData<List<SolutionItem>> {
        return solutions
    }

    fun loadSolutions(currentSolutions: List<SolutionItem>) {
        var hashset = currentSolutions.toHashSet()
        var sorted = hashset.sortedByDescending { it.value.length }
        solutions.value = sorted;
    }

    fun clearSolutions() {
        solutions.value = emptyList()
    }
}

class MainActivity : AppCompatActivity() {

    private lateinit var appBarConfiguration: AppBarConfiguration
    private lateinit var binding: ActivityMainBinding

    lateinit var solutioner: Solutioner

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        assets.open("french.txt").use {
            solutioner = Solutioner()
            Log.e(TAG, "Loading dictionary")
            solutioner.loadDictionary(it)
            Log.e(TAG, "Dictionary loaded")
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        // Inflate the menu; this adds items to the action bar if it is present.
        menuInflater.inflate(R.menu.menu_main, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        return when (item.itemId) {
            R.id.action_settings -> true
            else -> super.onOptionsItemSelected(item)
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        val navController = findNavController(R.id.nav_host_fragment_content_main)
        return navController.navigateUp(appBarConfiguration)
                || super.onSupportNavigateUp()
    }

    companion object {
        private const val TAG = "Deboggler_MainActivity"
    }
}