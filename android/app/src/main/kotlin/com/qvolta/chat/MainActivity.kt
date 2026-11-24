package com.qvolta.chat

import android.Manifest
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import androidx.compose.foundation.layout.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.foundation.background
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.qvolta.chat.ui.*

class MainActivity : ComponentActivity() {
    private val vm: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 1)
        setContent {
            QvTheme {
                val state by ServerController.state.collectAsStateWithLifecycle()
                Box(Modifier.fillMaxSize().background(Pal.bg).systemBarsPadding().imePadding()) {
                    if (state is ServerState.Ready) ChatScreen(vm) else StartScreen(vm, state)
                }
            }
        }
    }
}
