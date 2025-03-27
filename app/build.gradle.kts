plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "com.iauto.onnxjava"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.iauto.onnxjava"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"


    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        viewBinding = true
    }
    // 配置支持的 ABI
    splits {
        abi {
            isEnable = true
            reset()
            include("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
            isUniversalApk = false
        }
    }
}

dependencies {

    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
    // 使用 fileTree 引用本地的 onnxruntime-android-1.21.0.aar
    implementation(fileTree(mapOf("dir" to "libs", "include" to listOf("onnxruntime-android-1.21.0.aar"))))
    // https://mvnrepository.com/artifact/ai.djl.huggingface/tokenizers
    implementation("ai.djl.huggingface:tokenizers:0.32.0")
}