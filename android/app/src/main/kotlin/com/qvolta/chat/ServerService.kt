package com.qvolta.chat

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager

/** Foreground service so Android keeps the model server alive while the app is in the background. */
class ServerService : Service() {
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(i: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            ServerController.stop(); stopSelf(); return START_NOT_STICKY
        }
        val open = PendingIntent.getActivity(this, 0, Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val n = Notification.Builder(this, "server")
            .setSmallIcon(android.R.drawable.stat_notify_sync).setContentTitle("Q-Volta model is running")
            .setContentText("On-device inference server (localhost)").setContentIntent(open).setOngoing(true).build()
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        wake = (getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "qvolta:server").apply { acquire(6 * 3600_000L) }
        ServerController.start(this, ServerConfig(
            modelPath = intent?.getStringExtra("model") ?: return START_NOT_STICKY,
            threads = intent.getIntExtra("threads", 3), ctx = intent.getIntExtra("ctx", 8192),
            cpuList = intent.getStringExtra("cpus") ?: "7,3,4,5,6"))
        return START_NOT_STICKY
    }

    override fun onDestroy() { wake?.let { if (it.isHeld) it.release() }; super.onDestroy() }

    companion object {
        const val ACTION_STOP = "stop"
        fun start(c: Context, cfg: ServerConfig) = c.startForegroundService(Intent(c, ServerService::class.java)
            .putExtra("model", cfg.modelPath).putExtra("threads", cfg.threads).putExtra("ctx", cfg.ctx).putExtra("cpus", cfg.cpuList))
        fun stop(c: Context) { c.startService(Intent(c, ServerService::class.java).setAction(ACTION_STOP)) }
    }
}
