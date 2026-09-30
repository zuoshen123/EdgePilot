package com.edgepilot

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Home
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.List
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgepilot.harness.Recovery
import com.edgepilot.ui.screens.BaselineScreen
import com.edgepilot.ui.screens.HomeScreen
import com.edgepilot.ui.screens.MetricsScreen
import com.edgepilot.ui.theme.EdgePilotTheme
import com.edgepilot.viewmodel.BenchmarkViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    // v0.4 §2.3：intent 恢复通道与 UI 共用一个引擎所有权 → activity 级 VM（P-reco：通道走 lifecycleScope，不经 Compose 层）
    private val viewModel: BenchmarkViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            EdgePilotTheme {
                MainApp(viewModel)
            }
        }
        handleEpIntent(intent)  // 冷启动路
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent); setIntent(intent); handleEpIntent(intent)
    }

    private fun handleEpIntent(i: Intent) {
        val save = i.getStringExtra("ep_session_save"); val load = i.getStringExtra("ep_session_load")
        i.removeExtra("ep_session_save"); i.removeExtra("ep_session_load")   // 一次性消费：旋转/热启动不重放（spec §8.6）
        val base = save ?: load ?: return
        lifecycleScope.launch {
            val res = withContext(Dispatchers.Default) {
                Recovery.run(applicationContext, base, save != null) { Log.i("EdgePilotRecovery", it) }
            }
            // Recovery.run 以 release() 收尾=会话终结 → UI 旗标如实同步（裁定 §5.3；plan 文外增补，控制器裁决）
            viewModel.onEngineReleased()
            Log.i("EdgePilotRecovery", "channel done: $res")
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
                NavigationBarItem(
                    selected = selectedTab == 2,
                    onClick = { selectedTab = 2 },
                    icon = { Icon(Icons.Default.List, "基线") },
                    label = { Text("基线") }
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
                    onRunBenchmark = { p, cont -> viewModel.runBenchmark(p, 128, cont) },
                    onCancelBenchmark = viewModel::cancelRun,
                    onClearTurns = { viewModel.endSessionDemo() }
                )
                1 -> MetricsScreen(
                    result = viewModel.uiState.result,
                    metricsJson = ""
                )
                2 -> BaselineScreen(onEngineReleased = { viewModel.onEngineReleased() })
            }
        }
    }
}
