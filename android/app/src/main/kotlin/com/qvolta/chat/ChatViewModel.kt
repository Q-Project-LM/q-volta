package com.qvolta.chat

import android.app.Application
import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class UiMessage(val id: Long, val role: String, content: String = "", reasoning: String = "") {
    var content by mutableStateOf(content)
    var reasoning by mutableStateOf(reasoning)
    var streaming by mutableStateOf(false)
    var error by mutableStateOf<String?>(null)
    var genTps by mutableStateOf(0.0)
    var genTokens by mutableStateOf(0)
    var promptTps by mutableStateOf(0.0)
    var thinkMs by mutableStateOf(0L)
    var startedAt = 0L
}

data class MemInfo(val totalMb: Int = 0, val availMb: Int = 0, val cachedMb: Int = 0)
data class ModelFile(val file: File, val sizeGb: Double) { val name get() = file.name }

class ChatViewModel(app: Application) : AndroidViewModel(app) {
    private val prefs = app.getSharedPreferences("qv", Context.MODE_PRIVATE)
    val messages = mutableStateListOf<UiMessage>()
    var busy by mutableStateOf(false); private set
    var liveTps by mutableStateOf(0.0); private set
    var mem by mutableStateOf(MemInfo()); private set
    var models by mutableStateOf<List<ModelFile>>(emptyList()); private set
    var importProgress by mutableStateOf<Float?>(null); private set
    var importError by mutableStateOf<String?>(null)

    var selectedModel by mutableStateOf(prefs.getString("model", "") ?: "")
    var threads by mutableStateOf(prefs.getInt("threads", 3))
    var ctx by mutableStateOf(prefs.getInt("ctx", 8192))
    var temperature by mutableStateOf(prefs.getFloat("temp", 0.7f))
    var maxTokens by mutableStateOf(prefs.getInt("maxtok", 1024))
    var systemPrompt by mutableStateOf(prefs.getString("system", "") ?: "")
    var showThinking by mutableStateOf(prefs.getBoolean("showthink", true))
    var think by mutableStateOf(Think.entries.firstOrNull { it.name == prefs.getString("think", "Low") } ?: Think.Low)

    private var genJob: Job? = null
    private var nextId = 1L

    init {
        refreshModels()
        viewModelScope.launch { while (true) { mem = readMem(); delay(2000) } }
    }

    fun saveSettings() = prefs.edit().putString("model", selectedModel).putInt("threads", threads).putInt("ctx", ctx)
        .putFloat("temp", temperature).putInt("maxtok", maxTokens).putString("system", systemPrompt).putBoolean("showthink", showThinking).putString("think", think.name).apply()

    private fun modelDirs(): List<File> {
        val c = getApplication<Application>()
        return listOfNotNull(c.getExternalFilesDir(null), c.filesDir)
    }

    fun modelDirHint(): String = modelDirs().firstOrNull()?.absolutePath ?: ""

    fun refreshModels() {
        models = modelDirs().flatMap { d -> d.listFiles { f -> f.isFile && f.name.endsWith(".qv") }?.toList().orEmpty() }
            .map { ModelFile(it, it.length() / 1073741824.0) }.sortedBy { it.sizeGb }
        if (models.none { it.file.absolutePath == selectedModel }) selectedModel = models.firstOrNull()?.file?.absolutePath ?: ""
    }

    fun startServer() {
        if (selectedModel.isEmpty()) return
        saveSettings()
        ServerService.start(getApplication(), ServerConfig(selectedModel, threads, ctx))
    }

    fun stopServer() { stopGeneration(); ServerService.stop(getApplication()) }

    fun importModel(uri: Uri) {
        val c = getApplication<Application>()
        viewModelScope.launch(Dispatchers.IO) {
            try {
                importError = null
                var name = "model.qv"; var size = 0L
                c.contentResolver.query(uri, null, null, null, null)?.use { q ->
                    if (q.moveToFirst()) {
                        q.getColumnIndex(OpenableColumns.DISPLAY_NAME).takeIf { it >= 0 }?.let { name = q.getString(it) }
                        q.getColumnIndex(OpenableColumns.SIZE).takeIf { it >= 0 }?.let { size = q.getLong(it) }
                    }
                }
                val dst = File(c.filesDir, if (name.endsWith(".qv")) name else "$name.qv")
                importProgress = 0f
                c.contentResolver.openInputStream(uri)!!.use { ins ->
                    dst.outputStream().use { out ->
                        val buf = ByteArray(8 shl 20); var done = 0L
                        while (true) {
                            val n = ins.read(buf); if (n < 0) break
                            out.write(buf, 0, n); done += n
                            if (size > 0) importProgress = done.toFloat() / size
                        }
                    }
                }
                withContext(Dispatchers.Main) { refreshModels(); selectedModel = dst.absolutePath }
            } catch (e: Exception) { importError = e.message } finally { importProgress = null }
        }
    }

    fun send(text: String) {
        if (busy || text.isBlank()) return
        messages.add(UiMessage(nextId++, "user", text.trim()))
        val reply = UiMessage(nextId++, "assistant").also { it.streaming = true; it.startedAt = System.currentTimeMillis() }
        messages.add(reply)
        busy = true; liveTps = 0.0
        val history = buildList {
            if (systemPrompt.isNotBlank()) add(ChatMsg("system", systemPrompt))
            messages.dropLast(1).filter { it.content.isNotBlank() }.forEach { add(ChatMsg(it.role, it.content)) }
        }
        genJob = viewModelScope.launch {
            var tokens = 0; var t0 = 0L
            ChatApi.stream(history, GenParams(temperature, maxTokens, think = think))
                .catch { reply.error = it.message ?: "stream failed" }
                .collect { d ->
                    when (d) {
                        is Delta.Reasoning -> { reply.reasoning += d.text; reply.thinkMs = System.currentTimeMillis() - reply.startedAt; tokens++ }
                        is Delta.Content -> { reply.content += d.text; tokens++ }
                        is Delta.Done -> { reply.genTps = d.genTps; reply.genTokens = d.genTokens; reply.promptTps = d.promptTps }
                    }
                    if (t0 == 0L) t0 = System.currentTimeMillis()
                    val dt = (System.currentTimeMillis() - t0) / 1000.0
                    if (dt > 1.0) liveTps = tokens / dt
                }
            reply.streaming = false; busy = false; liveTps = 0.0
        }
    }

    fun stopGeneration() { genJob?.cancel(); messages.lastOrNull()?.streaming = false; busy = false; liveTps = 0.0 }
    fun clearChat() { stopGeneration(); messages.clear() }

    private fun readMem(): MemInfo = try {
        val m = File("/proc/meminfo").readLines().associate { l -> l.substringBefore(':') to (l.substringAfter(':').trim().substringBefore(' ').toLongOrNull() ?: 0L) }
        MemInfo((m["MemTotal"]!! / 1024).toInt(), (m["MemAvailable"]!! / 1024).toInt(), (m["Cached"]!! / 1024).toInt())
    } catch (_: Exception) { MemInfo() }
}
