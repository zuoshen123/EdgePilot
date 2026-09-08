# EdgePilot ProGuard Rules

# 保留 JNI 方法
-keepclasseswithmembernames class com.edgepilot.native.NativeEngine {
    native <methods>;
}

# 保留 Compose
-dontwarn androidx.compose.**
-keep class androidx.compose.** { *; }

# 保留数据类
-keep class com.edgepilot.viewmodel.** { *; }
