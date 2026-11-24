package com.qvolta.chat

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

data class ChatMsg(val role: String, val content: String)

enum class Think(val label: String, val effort: String) { Off("Off", "none"), Low("Low", "low"), Medium("Medium", "medium"), High("High", "high") }

data class GenParams(val temperature: Float = 0.7f, val maxTokens: Int = 1024, val topP: Float = 0.95f, val think: Think = Think.Low)

sealed interface Delta {
    data class Content(val text: String) : Delta
    data class Reasoning(val text: String) : Delta
    data class Done(val promptTokens: Int, val genTokens: Int, val promptTps: Double, val genTps: Double) : Delta
}

object ChatApi {
    /** Streams /v1/chat/completions (SSE). Cancelling the collector closes the connection, which stops generation server-side. */
    fun stream(messages: List<ChatMsg>, p: GenParams): Flow<Delta> = channelFlow {
        // Think.Off: render the prompt with effort 0 and pre-open the assistant's TEXT message (skips the thinking message entirely).
        // The OpenAI chat API has no assistant prefill, so this goes through /apply-template + /v1/completions.
        val raw = p.think == Think.Off
        val kw = JSONObject().put("reasoning_effort", p.think.effort)
        val msgs = JSONArray().also { a -> messages.forEach { a.put(JSONObject().put("role", it.role).put("content", it.content)) } }
        val path: String
        val body = if (raw) {
            val pc = URL("${ServerController.baseUrl}/apply-template").openConnection() as HttpURLConnection
            pc.requestMethod = "POST"; pc.doOutput = true; pc.connectTimeout = 3000; pc.readTimeout = 30000
            pc.setRequestProperty("content-type", "application/json")
            pc.outputStream.use { it.write(JSONObject().put("messages", msgs).put("chat_template_kwargs", kw).toString().toByteArray()) }
            val prompt = JSONObject(pc.inputStream.bufferedReader().readText()).getString("prompt")
            path = "/v1/completions"
            JSONObject().put("model", "local").put("stream", true).put("temperature", p.temperature.toDouble()).put("top_p", p.topP.toDouble())
                .put("max_tokens", p.maxTokens).put("prompt", prompt + "<|message_assistant|><|content_text|>")
                .put("special", true).put("stop", JSONArray().put("<|end_message|>")).toString()
        } else {
            path = "/v1/chat/completions"
            JSONObject().put("model", "local").put("stream", true).put("temperature", p.temperature.toDouble()).put("top_p", p.topP.toDouble())
                .put("max_tokens", p.maxTokens).put("messages", msgs).put("chat_template_kwargs", kw).toString()
        }
        val c = URL("${ServerController.baseUrl}$path").openConnection() as HttpURLConnection
        c.requestMethod = "POST"; c.doOutput = true; c.connectTimeout = 3000; c.readTimeout = 0
        c.setRequestProperty("content-type", "application/json")
        launch(Dispatchers.IO) {
            try {
                c.outputStream.use { it.write(body.toByteArray()) }
                if (c.responseCode != 200) error("HTTP ${c.responseCode}: ${c.errorStream?.bufferedReader()?.readText()?.take(300)}")
                var lastTimings: JSONObject? = null
                var usage: JSONObject? = null
                c.inputStream.bufferedReader().useLines { lines ->
                    for (line in lines) {
                        if (!line.startsWith("data:")) continue
                        val data = line.removePrefix("data:").trim()
                        if (data == "[DONE]") break
                        val j = JSONObject(data)
                        j.optJSONObject("timings")?.let { lastTimings = it }
                        j.optJSONObject("usage")?.let { usage = it }
                        val choice = j.optJSONArray("choices")?.optJSONObject(0) ?: continue
                        if (raw) { choice.optString("text").takeIf { it.isNotEmpty() }?.let { trySend(Delta.Content(it)) }; continue }
                        val d = choice.optJSONObject("delta") ?: continue
                        if (!d.isNull("reasoning_content")) d.optString("reasoning_content").takeIf { it.isNotEmpty() }?.let { trySend(Delta.Reasoning(it)) }
                        if (!d.isNull("content")) d.optString("content").takeIf { it.isNotEmpty() }?.let { trySend(Delta.Content(it)) }
                    }
                }
                val t = lastTimings
                trySend(Delta.Done(
                    promptTokens = t?.optInt("prompt_n") ?: usage?.optInt("prompt_tokens") ?: 0,
                    genTokens = t?.optInt("predicted_n") ?: usage?.optInt("completion_tokens") ?: 0,
                    promptTps = t?.optDouble("prompt_per_second") ?: 0.0, genTps = t?.optDouble("predicted_per_second") ?: 0.0))
                close()
            } catch (e: Exception) {
                close(e)
            }
        }
        awaitClose { c.disconnect() }
    }.flowOn(Dispatchers.IO)
}
