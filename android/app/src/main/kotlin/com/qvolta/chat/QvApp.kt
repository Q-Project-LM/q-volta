package com.qvolta.chat

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager

class QvApp : Application() {
    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(NotificationChannel("server", "Model server", NotificationManager.IMPORTANCE_LOW))
    }
}
