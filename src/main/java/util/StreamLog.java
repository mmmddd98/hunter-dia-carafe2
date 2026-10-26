package main.java.util;

import java.io.BufferedReader;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.function.Consumer;

public class StreamLog extends Thread {
    InputStream stream;
    String type;
    boolean isOut;
    List<Pattern> filterPatterns;
    private final Consumer<String> lineConsumer;

    public StreamLog(InputStream is, String type, boolean isOut) {
        this(is, type, isOut, new ArrayList<>(), null);
    }

    public StreamLog(InputStream is, String type, boolean isOut, List<String> filterPatterns) {
        this(is, type, isOut, filterPatterns, null);
    }

    public StreamLog(InputStream is, String type, boolean isOut, Consumer<String> lineConsumer) {
        this(is, type, isOut, new ArrayList<>(), lineConsumer);
    }

    private StreamLog(InputStream is, String type, boolean isOut, List<String> filterPatterns, Consumer<String> lineConsumer) {
        this.stream = is;
        this.type = type;
        this.isOut = isOut;
        // Compile regex patterns for better performance
        this.filterPatterns = new ArrayList<>();
        if (filterPatterns != null) {
            for (String pattern : filterPatterns) {
                this.filterPatterns.add(Pattern.compile(pattern));
            }
        }
        this.lineConsumer = lineConsumer;
    }

    private boolean shouldFilter(String line) {
        for (Pattern pattern : filterPatterns) {
            if (pattern.matcher(line).find()) {
                return true;
            }
        }
        return false;
    }

    public void run() {
        try {
            InputStreamReader isr = new InputStreamReader(stream);
            BufferedReader br = new BufferedReader(isr);
            String line;
            boolean previousLineWasPythonWarning = false;
            while ((line = br.readLine()) != null) {
                if (lineConsumer != null) {
                    lineConsumer.accept(line);
                }
                if (this.isOut && !shouldFilter(line)) {
                    boolean isPythonWarning = line.contains("UserWarning:");
                    // Python writes warnings to stderr, but stderr does not imply a failed
                    // process. Keep the warning and its source line visible without
                    // labelling them as an error. Real stderr output remains Error.
                    String lineType = (isPythonWarning ||
                            (previousLineWasPythonWarning && line.startsWith("  ")))
                            ? type.replace("Error:", "Warning:")
                            : type;
                    System.out.println(lineType + " > " + line);
                    previousLineWasPythonWarning = isPythonWarning;
                }
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
            System.exit(1);
        }
    }
}
