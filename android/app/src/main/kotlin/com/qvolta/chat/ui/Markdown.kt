package com.qvolta.chat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Minimal Markdown: fenced code blocks, `inline code`, **bold**, *italic*, "# headings" and "- bullets". */
@Composable
fun Markdown(text: String, color: Color = Pal.text, modifier: Modifier = Modifier) {
    val parts = text.split("```")
    SelectionContainer {
        Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            parts.forEachIndexed { i, part ->
                if (i % 2 == 1) {
                    val body = part.substringAfter('\n', part).trimEnd()
                    val lang = if (part.contains('\n')) part.substringBefore('\n').trim() else ""
                    Column(Modifier.fillMaxWidth().background(Color(0xFF0A0D14), RoundedCornerShape(12.dp)).padding(12.dp)) {
                        if (lang.isNotEmpty()) Text(lang, color = Pal.muted, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
                        Text(body, color = Color(0xFFD7DCEC), fontFamily = FontFamily.Monospace, fontSize = 13.sp, lineHeight = 19.sp)
                    }
                } else if (part.isNotBlank()) {
                    part.trim('\n').split("\n").forEach { line ->
                        val h = line.takeWhile { it == '#' }.length
                        when {
                            h in 1..4 && line.getOrNull(h) == ' ' -> Text(inline(line.drop(h + 1)), color = color, fontSize = (22 - 2 * h).sp, fontWeight = FontWeight.SemiBold)
                            line.trimStart().startsWith("- ") || line.trimStart().startsWith("* ") ->
                                Row { Text("•  ", color = Pal.violet); Text(inline(line.trimStart().drop(2)), color = color, lineHeight = 22.sp) }
                            line.isBlank() -> Spacer(Modifier.height(2.dp))
                            else -> Text(inline(line), color = color, lineHeight = 23.sp)
                        }
                    }
                }
            }
        }
    }
}

private fun inline(s: String): AnnotatedString = buildAnnotatedString {
    var i = 0
    while (i < s.length) {
        when {
            s.startsWith("**", i) && s.indexOf("**", i + 2) > 0 -> {
                val e = s.indexOf("**", i + 2); withStyle(SpanStyle(fontWeight = FontWeight.Bold)) { append(s.substring(i + 2, e)) }; i = e + 2
            }
            s[i] == '`' && s.indexOf('`', i + 1) > 0 -> {
                val e = s.indexOf('`', i + 1)
                withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = Color(0xFF0A0D14), color = Pal.teal)) { append(s.substring(i + 1, e)) }; i = e + 1
            }
            s[i] == '*' && s.indexOf('*', i + 1) > 0 && s.getOrNull(i + 1) != ' ' -> {
                val e = s.indexOf('*', i + 1); withStyle(SpanStyle(fontStyle = androidx.compose.ui.text.font.FontStyle.Italic)) { append(s.substring(i + 1, e)) }; i = e + 1
            }
            else -> { append(s[i]); i++ }
        }
    }
}
