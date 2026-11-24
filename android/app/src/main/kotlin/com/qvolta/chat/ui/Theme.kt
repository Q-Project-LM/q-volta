package com.qvolta.chat.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

object Pal {
    val bg = Color(0xFF0B0E14); val surface = Color(0xFF121724); val surface2 = Color(0xFF1A2133); val outline = Color(0xFF263049)
    val text = Color(0xFFE6E9F2); val muted = Color(0xFF8A93AB)
    val violet = Color(0xFF7C5CFF); val indigo = Color(0xFF4F46E5); val teal = Color(0xFF2DD4BF); val amber = Color(0xFFFBBF24); val red = Color(0xFFFF6B6B)
    val userBubble = Brush.linearGradient(listOf(violet, indigo))
    val heroGlow = Brush.radialGradient(listOf(violet.copy(alpha = .35f), Color.Transparent))
}

private val scheme = darkColorScheme(
    primary = Pal.violet, onPrimary = Color.White, secondary = Pal.teal, background = Pal.bg, onBackground = Pal.text,
    surface = Pal.surface, onSurface = Pal.text, surfaceVariant = Pal.surface2, onSurfaceVariant = Pal.muted, outline = Pal.outline, error = Pal.red,
)

private val type = Typography(
    bodyLarge = TextStyle(fontSize = 16.sp, lineHeight = 24.sp), bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 21.sp),
    titleLarge = TextStyle(fontSize = 22.sp, fontWeight = FontWeight.SemiBold), titleMedium = TextStyle(fontSize = 16.sp, fontWeight = FontWeight.SemiBold),
    labelSmall = TextStyle(fontSize = 11.sp, fontFamily = FontFamily.Monospace),
)

@Composable
fun QvTheme(content: @Composable () -> Unit) = MaterialTheme(colorScheme = scheme, typography = type, content = content)
