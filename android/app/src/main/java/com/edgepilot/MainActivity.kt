package com.edgepilot

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgepilot.ui.screens.HomeScreen
import com.edgepilot.ui.screens.MetricsScreen
import com.edgepilot.ui.theme.EdgePilotTheme
import com.edgepilot.viewmodel.BenchmarkViewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            EdgePilotTheme {
                MainApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainApp(viewModel: BenchmarkViewModel = viewModel()) {
    var selectedTab by remember { mutableStateOf(0) }

    LaunchedEffect(Unit) {
        viewModel.detectHardware()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("EdgePilot") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    titleContentColor = MaterialTheme.colorScheme.onPrimary
                )
            )
        },
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Icon(Icons.Default.Home, "Benchmark") },
                    label = { Text("测试") }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Icon(Icons.Default.Info, "Metrics") },
                    label = { Text("指标") }
                )
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            when (selectedTab) {
                0 -> HomeScreen(
                    uiState = viewModel.uiState,
                    onDetectHardware = { viewModel.detectHardware() },
                    onLoadModel = { viewModel.loadModel(it) },
                    onRunBenchmark = { viewModel.runBenchmark(it) }
                )
                1 -> MetricsScreen(
                    result = viewModel.uiState.result,
                    metricsJson = ""
                )
            }
        }
    }
}
