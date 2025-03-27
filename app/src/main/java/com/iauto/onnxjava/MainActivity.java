package com.iauto.onnxjava;

import android.os.Bundle;

import com.google.android.material.snackbar.Snackbar;

import androidx.appcompat.app.AppCompatActivity;

import android.view.View;

import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.AppBarConfiguration;
import androidx.navigation.ui.NavigationUI;

import com.iauto.onnxjava.databinding.ActivityMainBinding;

import android.view.Menu;
import android.view.MenuItem;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;

public class MainActivity extends AppCompatActivity {

    private AppBarConfiguration appBarConfiguration;
    private ActivityMainBinding binding;


    private OrtEnvironment env;
    private OrtSession session;
    private BertTokenizer tokenizer;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        setSupportActionBar(binding.toolbar);

        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        appBarConfiguration = new AppBarConfiguration.Builder(navController.getGraph()).build();
        NavigationUI.setupActionBarWithNavController(this, navController, appBarConfiguration);


        binding.fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                try {

                    env = OrtEnvironment.getEnvironment();

                    InputStream inputStream = getAssets().open("torch-model.onnx");
                    File outputFile = new File(getCacheDir(), "torch-model.onnx");
                    try (FileOutputStream outputStream = new FileOutputStream(outputFile)) {
                        byte[] buffer = new byte[1024];
                        int length;
                        while ((length = inputStream.read(buffer)) > 0) {
                            outputStream.write(buffer, 0, length);
                        }
                    }
                    OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
                    session = env.createSession(outputFile.getAbsolutePath(),sessionOptions);

                    inputStream = getAssets().open("vocab.txt");
                    outputFile = new File(getCacheDir(), "vocab.txt");
                    try (FileOutputStream outputStream = new FileOutputStream(outputFile)) {
                        byte[] buffer = new byte[1024];
                        int length;
                        while ((length = inputStream.read(buffer)) > 0) {
                            outputStream.write(buffer, 0, length);
                        }
                    }

                    String text = "帮我打开左车窗";

                    BertTokenizer bertTokenizer = new BertTokenizer(outputFile.getAbsolutePath());
                    List<String> inputs = bertTokenizer.tokenize(text);
                    if (inputs == null || inputs.isEmpty()) {
                        throw new IllegalArgumentException("Tokenize method returned an empty list");
                    }
                    Map<String, OnnxTensor> inputMap = bertTokenizer.tokenizeOnnxTensor(List.of(text));

                    long startTime = System.currentTimeMillis();
                    OrtSession.Result results = session.run(inputMap);
                    long endTime = System.currentTimeMillis();
                    OnnxValue outputTensorValue = results.get(0);

                    float[][] logits = (float[][]) outputTensorValue.getValue();
                    float[] probabilities = softmax(logits[0]);
                    int predictedLabelIndex = argmax(probabilities);

                    // 定义标签字典（请注意其中部分标签存在重复，根据实际情况调整）
                    Map<Integer, String> labelDict = new HashMap<>();
                    labelDict.put(0, "车窗场景");
                    labelDict.put(1, "温度控制场景");
                    labelDict.put(2, "氛围灯场景");
                    labelDict.put(3, "车门场景");
                    labelDict.put(4, "座椅场景");
                    labelDict.put(5, "调光玻璃场景");
                    labelDict.put(6, "车外灯场景");
                    labelDict.put(7, "调光玻璃场景");
                    labelDict.put(8, "雨刷场景");
                    labelDict.put(9, "导航场景");
                    labelDict.put(10, "后视镜场景");
                    labelDict.put(11, "播放音乐场景");
                    labelDict.put(12, "氛围灯场景");
                    labelDict.put(13, "方向盘加热场景");


                    String predictedLabel = labelDict.get(predictedLabelIndex);
                    System.out.println("场景是：" + predictedLabel + ", 转换onnx之后时间: " + (endTime - startTime) + " ms");

                    results.close();
                    session.close();
                    env.close();
                    Snackbar.make(view, "预测："+predictedLabel + "  预测时间：" + (endTime - startTime) + " ms" , Snackbar.LENGTH_LONG)
                            .setAnchorView(R.id.fab)
                            .setAction("Action", null).show();

                } catch (IOException | OrtException e) {
                    e.printStackTrace();
                    Snackbar.make(view, "预测失败", Snackbar.LENGTH_LONG)
                            .setAnchorView(R.id.fab)
                            .setAction("Action", null).show();
                }
            }
        });
    }

    // 计算 softmax
    private static float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        double sum = 0.0;
        float[] exps = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exps[i] = (float) Math.exp(logits[i] - max);
            sum += exps[i];
        }
        float[] softmax = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            softmax[i] = (float) (exps[i] / sum);
        }
        return softmax;
    }


    // 求 argmax
    private static int argmax(float[] array) {
        int index = 0;
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }




    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public boolean onSupportNavigateUp() {
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_content_main);
        return NavigationUI.navigateUp(navController, appBarConfiguration)
                || super.onSupportNavigateUp();
    }
}