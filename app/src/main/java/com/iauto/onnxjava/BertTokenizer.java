package com.iauto.onnxjava;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Bert Tokenizer
 *
 * @author jadepeng
 */
public class BertTokenizer implements Tokenizer {

    private String vocabFile = "vocab.txt";
    private Map<String, Integer> tokenIdMap;
    private Map<Integer, String> idTokenMap;
    private final boolean doLowerCase = true;
    private final boolean doBasicTokenize = true;
    private final List<String> neverSplit = new ArrayList<String>();
    private final String unkToken = "[UNK]";
    private final String sepToken = "[SEP]";
    private final String padToken = "[PAD]";
    private final String clsToken = "[CLS]";
    private final String maskToken = "[MASK]";
    private final boolean tokenizeChineseChars = true;
    private BasicTokenizer basicTokenizer;
    private WordpieceTokenizer wordpieceTokenizer;

    private static final int MAX_LEN = 2048;

    public BertTokenizer(String vocabFile) {
        this.vocabFile = vocabFile;
        init();
    }

    public BertTokenizer() {
        init();
    }

    private void init() {
        try {
            this.tokenIdMap = loadVocab(vocabFile);
        } catch (IOException e) {
            ;
        }
        this.idTokenMap = new HashMap<>(this.tokenIdMap.size());
        for (String key : tokenIdMap.keySet()) {
            this.idTokenMap.put(tokenIdMap.get(key), key);
        }

        if (doBasicTokenize) {
            this.basicTokenizer = new BasicTokenizer(doLowerCase, neverSplit, tokenizeChineseChars);
        }
        this.wordpieceTokenizer = new WordpieceTokenizer(tokenIdMap, unkToken);
    }

    private Map<String, Integer> loadVocab(String vocabFileName) throws IOException {
        return TokenizerUtils.generateTokenIdMap(new FileInputStream(vocabFileName));
    }

    /**
     * Tokenizes a piece of text into its word pieces.
     * <p>
     * This uses a greedy longest-match-first algorithm to perform tokenization
     * using the given vocabulary.
     * <p>
     * For example: input = "unaffable" output = ["un", "##aff", "##able"]
     * <p>
     * Args: text: A single token or whitespace separated tokens. This should have
     * already been passed through `BasicTokenizer`.
     * <p>
     * Returns: A list of wordpiece tokens.
     */
    @Override
    public List<String> tokenize(String text) {
        List<String> splitTokens = new ArrayList<>();
        if (doBasicTokenize) {
            for (String token : basicTokenizer.tokenize(text)) {
                splitTokens.addAll(wordpieceTokenizer.tokenize(token));
            }
        } else {
            splitTokens = wordpieceTokenizer.tokenize(text);
        }
        return splitTokens;
    }

    public String convertTokensToString(List<String> tokens) {
        // Converts a sequence of tokens (string) in a single string.
        return tokens.stream().map(s -> s.replace("##", "")).collect(Collectors.joining(" "));
    }

    public long[][] convertTokensToIds(List<String> tokens) {
        long[][] result = new long[1][tokens.size() + 2]; // 初始化内部数组，大小为 tokens.size() + 2
        int i = 0;
        result[0][i++] = tokenIdMap.get(clsToken); // 添加 [CLS] token
        for (String s : tokens) {
            result[0][i++] = tokenIdMap.get(s); // 添加 tokens
        }
        result[0][i++] = tokenIdMap.get(sepToken); // 添加 [SEP] token
        return result;
    }

    public long[][] generateAttentionMask(long[][] inputIds) {
        long[][] attentionMask = new long[inputIds.length][inputIds[0].length];
        for (int i = 0; i < inputIds.length; i++) {
            for (int j = 0; j < inputIds[i].length; j++) {
                attentionMask[i][j] = inputIds[i][j] != tokenIdMap.get(padToken) ? 1 : 0;
            }
        }
        return attentionMask;
    }

    static long[] paddingZero(long[] array1, int paddingSize) {
        long[] result = new long[array1.length + paddingSize];
        System.arraycopy(array1, 0, result, 0, array1.length);
        for (int i = array1.length; i < result.length; i++) {
            result[i] = 0;
        }
        return result;
    }

    public Map<String, OnnxTensor> tokenizeOnnxTensor(List<String> texts)
            throws OrtException {
        var env = OrtEnvironment.getEnvironment();
        long[][] textTokensIds = new long[texts.size()][]; // 初始化内部数组
        long[][] typeIds = new long[texts.size()][]; // 初始化内部数组
        int rowIndex = 0;
        int maxColumn = 0;
        for (String text : texts) {
            List<String> tokens = tokenize(text);
            long[] tokenIds = new long[tokens.size() + 2];
            int index = 0;
            tokenIds[index++] = tokenIdMap.get(clsToken);
            for (String s : tokens) {
                tokenIds[index++] = tokenIdMap.get(s);
            }
            tokenIds[index++] = tokenIdMap.get(sepToken);
            textTokensIds[rowIndex] = tokenIds;
            typeIds[rowIndex++] = buildTokenTypeArray(index);
            maxColumn = Math.max(maxColumn, index);
        }

        // padding 0
        for (int row = 0; row < texts.size(); row++) {
            if (textTokensIds[row].length < maxColumn) {
                // padding 0
                textTokensIds[row] = paddingZero(textTokensIds[row], maxColumn - textTokensIds[row].length);
                typeIds[row] = paddingZero(typeIds[row], maxColumn - typeIds[row].length);
            }
        }

        long[][] attentionMask = generateAttentionMask(textTokensIds);

        OnnxTensor inputIds = OnnxTensor.createTensor(env, textTokensIds);
        OnnxTensor tokenTypeIds = OnnxTensor.createTensor(env, typeIds);
        OnnxTensor attentionMasks = OnnxTensor.createTensor(env, attentionMask);

        Map<String, OnnxTensor> inputMap = new HashMap<>();
        inputMap.put("input_ids", inputIds);
        inputMap.put("token_type_ids", tokenTypeIds);
        inputMap.put("attention_mask", attentionMasks);

        return inputMap;
    }

    long[] buildTokenTypeArray(int size) {
        long[] typeIds = new long[size];
        for (int i = 0; i < size; i++) {
            typeIds[i] = 0;
        }
        return typeIds;
    }

    public int vocabSize() {
        return tokenIdMap.size();
    }
}
