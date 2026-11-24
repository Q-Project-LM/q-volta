package com.qvolta.chat.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.qvolta.chat.*
import java.util.Locale

private fun f1(v: Double) = String.format(Locale.US, "%.1f", v)

// ------------------------------------------------------------------------------------------------ start / setup
@Composable
fun StartScreen(vm: ChatViewModel, state: ServerState) {
    var showSettings by remember { mutableStateOf(false) }
    val log by ServerController.log.collectAsStateWithLifecycle()
    val loadS by ServerController.loadSeconds.collectAsStateWithLifecycle()
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { it?.let(vm::importModel) }
    val starting = state is ServerState.Starting
    Box(Modifier.fillMaxSize()) {
        Box(Modifier.fillMaxWidth().height(380.dp).background(Pal.heroGlow))
        Column(Modifier.fillMaxSize().padding(horizontal = 20.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Row(Modifier.fillMaxWidth().padding(top = 8.dp), horizontalArrangement = Arrangement.End) {
                IconButton({ showSettings = true }) { Icon(Icons.Default.Tune, null, tint = Pal.muted) }
            }
            Spacer(Modifier.height(24.dp))
            Logo(starting)
            Spacer(Modifier.height(20.dp))
            Text("Q-Volta", fontSize = 34.sp, fontWeight = FontWeight.Bold, color = Pal.text)
            Text("Inkling-2-Mini · fully on-device", color = Pal.muted, fontSize = 14.sp)
            Spacer(Modifier.height(28.dp))

            MemoryCard(vm.mem, vm.models.firstOrNull { it.file.absolutePath == vm.selectedModel }?.sizeGb)
            Spacer(Modifier.height(14.dp))
            ModelCard(vm, onImport = { picker.launch(arrayOf("*/*")) })
            Spacer(Modifier.weight(1f))

            AnimatedVisibility(starting) {
                Column(Modifier.fillMaxWidth().padding(bottom = 12.dp)) {
                    LinearProgressIndicator(Modifier.fillMaxWidth().clip(CircleShape), color = Pal.violet, trackColor = Pal.surface2)
                    Text(log.lastOrNull().orEmpty().take(110), color = Pal.muted, fontSize = 11.sp, fontFamily = FontFamily.Monospace, maxLines = 2, modifier = Modifier.padding(top = 8.dp))
                }
            }
            if (state is ServerState.Failed) Text(state.message, color = Pal.red, fontSize = 13.sp, modifier = Modifier.padding(bottom = 10.dp))
            Button(
                onClick = { if (starting) vm.stopServer() else vm.startServer() },
                enabled = vm.selectedModel.isNotEmpty(),
                modifier = Modifier.fillMaxWidth().height(56.dp), shape = RoundedCornerShape(18.dp),
                colors = ButtonDefaults.buttonColors(containerColor = if (starting) Pal.surface2 else Pal.violet),
            ) {
                Icon(if (starting) Icons.Default.Close else Icons.Default.PlayArrow, null)
                Spacer(Modifier.width(8.dp))
                Text(if (starting) "Cancel" else "Load model", fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
            }
            Text(if (loadS != null) "Loaded in ${f1(loadS!!)} s" else "Weights are memory-mapped; first tokens warm the cache",
                color = Pal.muted, fontSize = 12.sp, modifier = Modifier.padding(top = 10.dp, bottom = 16.dp))
        }
    }
    if (showSettings) SettingsSheet(vm) { showSettings = false }
}

@Composable
private fun Logo(pulse: Boolean) {
    val t = rememberInfiniteTransition(label = "logo")
    val s by t.animateFloat(1f, if (pulse) 1.12f else 1.03f, infiniteRepeatable(tween(1400, easing = FastOutSlowInEasing), RepeatMode.Reverse), label = "s")
    Box(Modifier.size(96.dp).graphicsLayer { scaleX = s; scaleY = s }.clip(RoundedCornerShape(28.dp)).background(Pal.userBubble), contentAlignment = Alignment.Center) {
        Icon(Icons.Default.Bolt, null, tint = Color.White, modifier = Modifier.size(54.dp))
    }
}

@Composable
private fun MemoryCard(m: MemInfo, modelGb: Double?) {
    val need = (modelGb ?: 0.0) * 1024
    val usable = m.availMb + m.cachedMb * 0.9
    val enough = need == 0.0 || usable > need * 0.95
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), colors = CardDefaults.cardColors(containerColor = Pal.surface)) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Memory, null, tint = Pal.teal); Spacer(Modifier.width(8.dp))
                Text("Memory", fontWeight = FontWeight.SemiBold, color = Pal.text)
                Spacer(Modifier.weight(1f))
                Text("${f1(m.availMb / 1024.0)} GB free · ${f1(m.cachedMb / 1024.0)} GB cache", color = Pal.muted, fontSize = 12.sp)
            }
            Spacer(Modifier.height(10.dp))
            val frac = if (m.totalMb > 0) 1f - m.availMb.toFloat() / m.totalMb else 0f
            LinearProgressIndicator({ frac }, Modifier.fillMaxWidth().height(8.dp).clip(CircleShape), color = if (enough) Pal.teal else Pal.amber, trackColor = Pal.surface2)
            if (modelGb != null) Text(
                if (enough) "Model ${f1(modelGb)} GB fits in the page cache — expect full speed."
                else "Model ${f1(modelGb)} GB is larger than the available memory. Close other apps for best speed.",
                color = if (enough) Pal.muted else Pal.amber, fontSize = 12.sp, modifier = Modifier.padding(top = 8.dp))
        }
    }
}

@Composable
private fun ModelCard(vm: ChatViewModel, onImport: () -> Unit) {
    Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), colors = CardDefaults.cardColors(containerColor = Pal.surface)) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Psychology, null, tint = Pal.violet); Spacer(Modifier.width(8.dp))
                Text("Model", fontWeight = FontWeight.SemiBold, color = Pal.text); Spacer(Modifier.weight(1f))
                IconButton({ vm.refreshModels() }, Modifier.size(32.dp)) { Icon(Icons.Default.Refresh, null, tint = Pal.muted) }
                TextButton(onImport, enabled = vm.importProgress == null) { Text("Import .qv") }
            }
            vm.importProgress?.let { LinearProgressIndicator({ it }, Modifier.fillMaxWidth().clip(CircleShape), color = Pal.violet, trackColor = Pal.surface2) }
            vm.importError?.let { Text(it, color = Pal.red, fontSize = 12.sp) }
            if (vm.models.isEmpty()) Text("No model found. Copy a .qv file to:\n${vm.modelDirHint()}\nor import one.", color = Pal.muted, fontSize = 12.sp, lineHeight = 18.sp)
            vm.models.forEach { m ->
                val sel = m.file.absolutePath == vm.selectedModel
                Row(Modifier.fillMaxWidth().padding(top = 6.dp).clip(RoundedCornerShape(12.dp))
                    .background(if (sel) Pal.violet.copy(alpha = .16f) else Color.Transparent)
                    .border(1.dp, if (sel) Pal.violet else Pal.outline, RoundedCornerShape(12.dp))
                    .clickable { vm.selectedModel = m.file.absolutePath; vm.saveSettings() }.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    RadioButton(sel, null, colors = RadioButtonDefaults.colors(selectedColor = Pal.violet)); Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) { Text(m.name, color = Pal.text, maxLines = 1, overflow = TextOverflow.Ellipsis, fontSize = 14.sp); Text("${f1(m.sizeGb)} GB", color = Pal.muted, fontSize = 12.sp) }
                }
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------ chat
private val suggestions = listOf("Explain how a transformer works in simple terms", "Write a Python function to merge two sorted lists", "Что такое энтропия? Объясни коротко", "Give me a 3-day itinerary for Tokyo")

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(vm: ChatViewModel) {
    var showSettings by remember { mutableStateOf(false) }
    val list = rememberLazyListState()
    val last = vm.messages.lastOrNull()
    LaunchedEffect(vm.messages.size, last?.content?.length, last?.reasoning?.length) { if (vm.messages.isNotEmpty()) list.scrollToItem(vm.messages.size - 1) }
    Column(Modifier.fillMaxSize()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(34.dp).clip(RoundedCornerShape(10.dp)).background(Pal.userBubble), contentAlignment = Alignment.Center) { Icon(Icons.Default.Bolt, null, tint = Color.White, modifier = Modifier.size(20.dp)) }
            Spacer(Modifier.width(10.dp))
            Column(Modifier.weight(1f)) {
                Text("Inkling-2-Mini", fontWeight = FontWeight.SemiBold, color = Pal.text, fontSize = 16.sp)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(if (vm.busy) Pal.amber else Pal.teal)); Spacer(Modifier.width(6.dp))
                    Text(if (vm.busy) (if (vm.liveTps > 0) "${f1(vm.liveTps)} tok/s" else "thinking…") else "on-device · ${f1(vm.mem.availMb / 1024.0)} GB free", color = Pal.muted, fontSize = 12.sp)
                }
            }
            IconButton({ vm.clearChat() }) { Icon(Icons.Default.DeleteSweep, null, tint = Pal.muted) }
            IconButton({ showSettings = true }) { Icon(Icons.Default.Tune, null, tint = Pal.muted) }
        }
        HorizontalDivider(color = Pal.outline.copy(alpha = .5f))
        Box(Modifier.weight(1f).fillMaxWidth()) {
            if (vm.messages.isEmpty()) EmptyState { vm.send(it) }
            else LazyColumn(state = list, modifier = Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                items(vm.messages, key = { it.id }) { Bubble(it, vm.showThinking) }
            }
        }
        Composer(vm)
    }
    if (showSettings) SettingsSheet(vm) { showSettings = false }
}

@Composable
private fun EmptyState(onPick: (String) -> Unit) {
    Column(Modifier.fillMaxSize().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Center) {
        Box(Modifier.size(72.dp).clip(RoundedCornerShape(22.dp)).background(Pal.userBubble), contentAlignment = Alignment.Center) { Icon(Icons.Default.Bolt, null, tint = Color.White, modifier = Modifier.size(40.dp)) }
        Spacer(Modifier.height(16.dp))
        Text("How can I help?", fontSize = 24.sp, fontWeight = FontWeight.SemiBold, color = Pal.text)
        Text("Everything runs on this phone. No cloud.", color = Pal.muted, fontSize = 14.sp)
        Spacer(Modifier.height(24.dp))
        suggestions.forEach {
            Box(Modifier.fillMaxWidth().padding(vertical = 4.dp).clip(RoundedCornerShape(14.dp)).background(Pal.surface).border(1.dp, Pal.outline, RoundedCornerShape(14.dp)).clickable { onPick(it) }.padding(14.dp)) {
                Text(it, color = Pal.text, fontSize = 14.sp)
            }
        }
    }
}

@Composable
private fun Bubble(m: UiMessage, showThinking: Boolean) {
    if (m.role == "user") {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            Box(Modifier.widthIn(max = 320.dp).clip(RoundedCornerShape(20.dp, 20.dp, 6.dp, 20.dp)).background(Pal.userBubble).padding(horizontal = 16.dp, vertical = 12.dp)) {
                Text(m.content, color = Color.White, fontSize = 16.sp, lineHeight = 23.sp)
            }
        }
        return
    }
    Column(Modifier.fillMaxWidth()) {
        if (showThinking && m.reasoning.isNotEmpty()) Thinking(m)
        if (m.content.isNotEmpty()) {
            Box(Modifier.clip(RoundedCornerShape(6.dp, 20.dp, 20.dp, 20.dp)).background(Pal.surface).padding(horizontal = 16.dp, vertical = 12.dp)) { Markdown(m.content) }
        } else if (m.streaming && m.reasoning.isEmpty()) TypingDots()
        m.error?.let { Text(it, color = Pal.red, fontSize = 12.sp, modifier = Modifier.padding(top = 6.dp)) }
        if (!m.streaming && m.genTps > 0) Text("${f1(m.genTps)} tok/s · ${m.genTokens} tokens · prompt ${f1(m.promptTps)} tok/s", color = Pal.muted, fontSize = 11.sp, fontFamily = FontFamily.Monospace, modifier = Modifier.padding(start = 6.dp, top = 6.dp))
    }
}

@Composable
private fun Thinking(m: UiMessage) {
    var open by remember(m.id) { mutableStateOf(true) }
    LaunchedEffect(m.content.isNotEmpty()) { if (m.content.isNotEmpty()) open = false }
    Column(Modifier.padding(bottom = 8.dp).clip(RoundedCornerShape(14.dp)).background(Pal.surface2.copy(alpha = .55f)).border(1.dp, Pal.outline, RoundedCornerShape(14.dp)).animateContentSize()) {
        Row(Modifier.fillMaxWidth().clickable { open = !open }.padding(horizontal = 12.dp, vertical = 9.dp), verticalAlignment = Alignment.CenterVertically) {
            Icon(Icons.Default.AutoAwesome, null, tint = Pal.violet, modifier = Modifier.size(16.dp)); Spacer(Modifier.width(8.dp))
            Text(if (m.streaming && m.content.isEmpty()) "Thinking…" else "Thought for ${m.thinkMs / 1000}s", color = Pal.muted, fontSize = 13.sp, modifier = Modifier.weight(1f))
            Icon(if (open) Icons.Default.ExpandLess else Icons.Default.ExpandMore, null, tint = Pal.muted)
        }
        if (open) Text(m.reasoning.trim(), color = Pal.muted, fontSize = 13.sp, lineHeight = 19.sp, modifier = Modifier.padding(start = 12.dp, end = 12.dp, bottom = 12.dp))
    }
}

@Composable
private fun TypingDots() {
    val t = rememberInfiniteTransition(label = "dots")
    Row(Modifier.clip(RoundedCornerShape(20.dp)).background(Pal.surface).padding(horizontal = 18.dp, vertical = 16.dp), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        repeat(3) { i ->
            val a by t.animateFloat(.25f, 1f, infiniteRepeatable(tween(600, delayMillis = i * 150), RepeatMode.Reverse), label = "d$i")
            Box(Modifier.size(8.dp).alpha(a).clip(CircleShape).background(Pal.violet))
        }
    }
}

@Composable
private fun Composer(vm: ChatViewModel) {
    var text by remember { mutableStateOf("") }
    fun go() { if (text.isNotBlank()) { vm.send(text); text = "" } }
    Column {
    Row(Modifier.fillMaxWidth().padding(start = 14.dp, end = 14.dp, top = 8.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Icon(Icons.Default.AutoAwesome, null, tint = if (vm.think == Think.Off) Pal.muted else Pal.violet, modifier = Modifier.size(16.dp))
        Text("Thinking", color = Pal.muted, fontSize = 12.sp, modifier = Modifier.padding(end = 4.dp))
        Think.entries.forEach { t ->
            val sel = vm.think == t
            Box(Modifier.clip(CircleShape).background(if (sel) Pal.violet else Pal.surface).border(1.dp, if (sel) Pal.violet else Pal.outline, CircleShape)
                .clickable(enabled = !vm.busy) { vm.think = t; vm.saveSettings() }.padding(horizontal = 12.dp, vertical = 6.dp)) {
                Text(t.label, color = if (sel) Color.White else Pal.muted, fontSize = 12.sp, fontWeight = if (sel) FontWeight.SemiBold else FontWeight.Normal)
            }
        }
    }
    Row(Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.Bottom) {
        TextField(
            text, { text = it }, Modifier.weight(1f), placeholder = { Text("Message", color = Pal.muted) }, maxLines = 6, shape = RoundedCornerShape(24.dp),
            colors = TextFieldDefaults.colors(focusedContainerColor = Pal.surface, unfocusedContainerColor = Pal.surface, focusedIndicatorColor = Color.Transparent, unfocusedIndicatorColor = Color.Transparent, cursorColor = Pal.violet, focusedTextColor = Pal.text, unfocusedTextColor = Pal.text),
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send), keyboardActions = KeyboardActions(onSend = { go() }),
        )
        Spacer(Modifier.width(10.dp))
        FloatingActionButton(
            onClick = { if (vm.busy) vm.stopGeneration() else go() }, containerColor = if (vm.busy) Pal.surface2 else Pal.violet, contentColor = Color.White,
            shape = CircleShape, modifier = Modifier.size(52.dp), elevation = FloatingActionButtonDefaults.elevation(0.dp),
        ) { Icon(if (vm.busy) Icons.Default.Stop else Icons.AutoMirrored.Filled.Send, null) }
    }
    }
}

// ------------------------------------------------------------------------------------------------ settings
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsSheet(vm: ChatViewModel, onClose: () -> Unit) {
    val state by ServerController.state.collectAsStateWithLifecycle()
    ModalBottomSheet({ vm.saveSettings(); onClose() }, containerColor = Pal.surface, contentColor = Pal.text) {
        Column(Modifier.padding(horizontal = 20.dp).padding(bottom = 28.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Text("Settings", fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
            SliderRow("Temperature", f1(vm.temperature.toDouble()), vm.temperature, 0f..1.5f) { vm.temperature = it }
            SliderRow("Max new tokens", "${vm.maxTokens}", vm.maxTokens.toFloat(), 128f..4096f, 30) { vm.maxTokens = (it / 128).toInt() * 128 }
            Text("Engine (applies after reloading the model)", color = Pal.muted, fontSize = 12.sp)
            SliderRow("CPU threads", "${vm.threads}", vm.threads.toFloat(), 1f..5f, 3) { vm.threads = it.toInt() }
            SliderRow("Context", "${vm.ctx / 1024}K", (vm.ctx / 1024).toFloat(), 2f..32f, 14) { vm.ctx = (it.toInt().coerceAtLeast(2)) * 1024 }
            OutlinedTextField(vm.systemPrompt, { vm.systemPrompt = it }, Modifier.fillMaxWidth(), label = { Text("System prompt") }, maxLines = 4, shape = RoundedCornerShape(14.dp))
            Row(verticalAlignment = Alignment.CenterVertically) { Text("Show thinking", Modifier.weight(1f)); Switch(vm.showThinking, { vm.showThinking = it }) }
            if (state !is ServerState.Stopped) OutlinedButton({ vm.stopServer(); onClose() }, Modifier.fillMaxWidth()) { Text("Unload model & stop server", color = Pal.red) }
        }
    }
}

@Composable
private fun SliderRow(label: String, value: String, v: Float, range: ClosedFloatingPointRange<Float>, steps: Int = 0, onChange: (Float) -> Unit) {
    Column {
        Row { Text(label, Modifier.weight(1f), fontSize = 14.sp); Text(value, color = Pal.teal, fontFamily = FontFamily.Monospace) }
        Slider(v, onChange, valueRange = range, steps = steps, colors = SliderDefaults.colors(thumbColor = Pal.violet, activeTrackColor = Pal.violet, inactiveTrackColor = Pal.surface2))
    }
}
