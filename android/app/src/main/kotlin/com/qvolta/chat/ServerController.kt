package com.qvolta.chat

import android.content.Context
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

sealed interface ServerState {
    data object Stopped : ServerState
    data object Starting : ServerState
    data object Ready : ServerState
    data class Failed(val message: String) : ServerState
}

data class ServerConfig(
    val modelPath: String,
    val threads: Int = 3,
    val ctx: Int = 8192,
    val cpuList: String = "7,3,4,5,6",   // Snapdragon 8 Gen 2: X3 (7), A715 (3,4), A710 (5,6); A510 (0-2) are skipped
)

/** Runs the bundled `libqv_server.so` (qv_server, CPU backend) as a child process on 127.0.0.1:PORT. */
object ServerController {
    const val PORT = 18765
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private var process: Process? = null
    private var job: Job? = null

    private val _state = MutableStateFlow<ServerState>(ServerState.Stopped)
    val state: StateFlow<ServerState> = _state
    private val _log = MutableStateFlow<List<String>>(emptyList())
    val log: StateFlow<List<String>> = _log
    private val _loadSeconds = MutableStateFlow<Double?>(null)
    val loadSeconds: StateFlow<Double?> = _loadSeconds

    val baseUrl get() = "http://127.0.0.1:$PORT"

    @Synchronized
    fun start(ctx: Context, cfg: ServerConfig) {
        if (_state.value is ServerState.Starting || _state.value is ServerState.Ready) return
        _state.value = ServerState.Starting
        _log.value = emptyList()
        _loadSeconds.value = null
        val exe = File(ctx.applicationInfo.nativeLibraryDir, "libqv_server.so")
        job = scope.launch {
            try {
                if (!exe.exists()) error("libqv_server.so not found in ${exe.parent}")
                val pb = ProcessBuilder(
                    exe.absolutePath, "--model", cfg.modelPath, "--backend", "cpu", "--threads", cfg.threads.toString(),
                    "--host", "127.0.0.1", "--port", PORT.toString(), "-c", cfg.ctx.toString(), "--no-webui",
                ).redirectErrorStream(true)
                pb.environment()["QV_CPU_LIST"] = cfg.cpuList.split(",").take(cfg.threads).joinToString(",")
                _log.update { it + ("$ " + pb.command().joinToString(" ") + "  [QV_CPU_LIST=" + pb.environment()["QV_CPU_LIST"] + ", exists=" + File(cfg.modelPath).exists() + ", size=" + File(cfg.modelPath).length() + "]") }
                val p = pb.start()
                process = p
                launch {
                    p.inputStream.bufferedReader().forEachLine { line ->
                        _log.update { (it + line).takeLast(300) }
                        Regex("model loaded in ([0-9.]+)s").find(line)?.let { m -> _loadSeconds.value = m.groupValues[1].toDoubleOrNull() }
                    }
                }
                launch {
                    val code = p.waitFor()
                    if (process === p) {
                        process = null
                        if (_state.value !is ServerState.Stopped)
                            _state.value = ServerState.Failed("Server exited (code $code).\n" + _log.value.takeLast(6).joinToString("\n"))
                    }
                }
                // wait until /health says ok
                repeat(1800) {
                    if (!isActive || process !== p) return@launch
                    if (healthOk()) { _state.value = ServerState.Ready; return@launch }
                    delay(500)
                }
                _state.value = ServerState.Failed("Timed out while loading the model")
            } catch (e: Exception) {
                _state.value = ServerState.Failed(e.message ?: e.toString())
            }
        }
    }

    @Synchronized
    fun stop() {
        job?.cancel()
        val p = process
        process = null
        _state.value = ServerState.Stopped
        p?.destroy()
        scope.launch { delay(1500); p?.destroyForcibly() }
    }

    private fun healthOk(): Boolean = try {
        val c = URL("$baseUrl/health").openConnection() as HttpURLConnection
        c.connectTimeout = 400; c.readTimeout = 800
        (c.responseCode == 200).also { c.disconnect() }
    } catch (_: Exception) { false }
}
