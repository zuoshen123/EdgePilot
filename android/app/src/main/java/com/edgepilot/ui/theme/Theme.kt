package com.edgepilot.ui.theme

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// EdgePilot 品牌色
val Primary = Color(0xFF6366F1)        // Indigo
val PrimaryDark = Color(0xFF4F46E5)
val Secondary = Color(0xFF10B981)      // Emerald
val Accent = Color(0xFFF59E0B)         // Amber
val Surface = Color(0xFFF8FAFC)
val SurfaceDark = Color(0xFF1E293B)
val Error = Color(0xFFEF4444)

// 性能指标色
val MetricGood = Color(0xFF10B981)
val MetricWarn = Color(0xFFF59E0B)
val MetricBad = Color(0xFFEF4444)

private val LightColorScheme = lightColorScheme(
    primary = Primary,
    onPrimary = Color.White,
    secondary = Secondary,
    onSecondary = Color.White,
    surface = Surface,
    onSurface = Color(0xFF1E293B),
    background = Color(0xFFF1F5F9),
    error = Error
)

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF818CF8),
    onPrimary = Color.White,
    secondary = Color(0xFF34D399),
    onSecondary = Color.Black,
    surface = SurfaceDark,
    onSurface = Color(0xFFE2E8F0),
    background = Color(0xFF0F172A),
    error = Color(0xFFF87171)
)

@Composable
fun EdgePilotTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = when {
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context)
            else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColorScheme
        else -> LightColorScheme
    }

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content
    )
}
