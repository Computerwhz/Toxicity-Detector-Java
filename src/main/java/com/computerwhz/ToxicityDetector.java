package com.computerwhz;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.LongBuffer;
import java.nio.file.*;
import java.util.*;

public class ToxicityDetector {

    private final HuggingFaceTokenizer tokenizer;
    private final OrtEnvironment env;
    private final OrtSession session;
    private final String[] labels;

    public ToxicityDetector() throws Exception {
        // Extract model + tokenizer to temp files from resources
        Path modelPath = extractResource("model_quantized.onnx");
        Path tokenizerPath = extractResource("tokenizer.json");

        // Load tokenizer
        tokenizer = HuggingFaceTokenizer.newInstance(tokenizerPath);

        // Create ONNX Runtime environment and session
        env = OrtEnvironment.getEnvironment();
        SessionOptions opts = new SessionOptions();
        opts.setIntraOpNumThreads(Math.max(1, Runtime.getRuntime().availableProcessors() / 2));
        opts.setInterOpNumThreads(1);
        session = env.createSession(modelPath.toString(), opts);

        // Load label names from config.json
        labels = loadLabelsFromConfig();

        System.out.println("Loaded model with " + labels.length + " labels.");
    }

    private Path extractResource(String resourceName) throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(resourceName)) {
            if (is == null) {
                throw new FileNotFoundException(resourceName + " not found in resources folder");
            }

            String fileName = Paths.get(resourceName).getFileName().toString();
            String suffix = "";
            int idx = fileName.lastIndexOf('.');
            if (idx != -1) {
                suffix = fileName.substring(idx); // .onnx
                fileName = fileName.substring(0, idx);
            }

            Path tmp = Files.createTempFile(fileName + "_", suffix);
            tmp.toFile().deleteOnExit();

            // Stream copy with buffer
            try (OutputStream os = Files.newOutputStream(tmp, StandardOpenOption.WRITE)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }
            }

            return tmp;
        }
    }



    private String[] loadLabelsFromConfig() throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream("config.json")) {
            if (is == null)
                throw new FileNotFoundException("config.json not found in resources folder");

            ObjectMapper om = new ObjectMapper();
            Map<String, Object> cfg = om.readValue(is, new TypeReference<>() {
            });
            @SuppressWarnings("unchecked")
            Map<String, String> id2label = (Map<String, String>) cfg.get("id2label");

            return id2label.entrySet().stream()
                    .sorted(Comparator.comparingInt(e -> Integer.parseInt(e.getKey())))
                    .map(Map.Entry::getValue)
                    .toArray(String[]::new);
        }
    }

    /**
     * Analyze a single text string for toxicity and identity-related probabilities.
     *
     * @param text input text
     * @return ToxicityScore (immutable result with all 16 labels)
     */
    public ToxicityScore analyze(String text) throws Exception {
        var enc = tokenizer.encode(text);
        long[] ids = enc.getIds();
        long[] mask = enc.getAttentionMask();
        long[] shape = new long[]{1, ids.length};

        try (OnnxTensor inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(ids), shape);
             OnnxTensor attnMask = OnnxTensor.createTensor(env, LongBuffer.wrap(mask), shape)) {

            Map<String, OnnxTensor> inputs = Map.of(
                    "input_ids", inputIds,
                    "attention_mask", attnMask
            );

            try (OrtSession.Result result = session.run(inputs)) {
                float[][] logits = (float[][]) result.get(0).getValue();
                float[] row = logits[0];

                Map<String, Double> scores = new LinkedHashMap<>();
                int n = Math.min(row.length, labels.length);
                for (int i = 0; i < n; i++) {
                    double prob = 1.0 / (1.0 + Math.exp(-row[i]));
                    scores.put(labels[i], Double.valueOf(prob));
                }

                return ToxicityScore.of(text, scores);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        System.out.println(new ToxicityDetector().analyze("Hello").getScores());
    }
}