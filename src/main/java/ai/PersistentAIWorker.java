package main.java.ai;

import main.java.input.CParameter;
import main.java.util.Cloger;
import main.java.util.StreamLog;
import org.apache.commons.lang3.StringUtils;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

import static main.java.ai.AIGear.get_jar_path;
import static main.java.ai.AIGear.get_py_path;

/** One long-lived, single-visible-GPU Python prediction worker. */
public final class PersistentAIWorker implements Runnable {
    private static final String[] CACHE_FILES = {"ms2_model.pt", "rt_model.pt", "ccs_model.pt", "model_evaluation_metrics.json", "meta.json"};
    public static final class Chunk {
        final String inputFile;
        final String outPrefix;
        final int peptideForms;
        public Chunk(String inputFile, String outPrefix, int peptideForms) {
            this.inputFile = inputFile;
            this.outPrefix = outPrefix;
            this.peptideForms = peptideForms;
        }
    }

    private final String modelDir, outDir, device, instrument, aiMode, aiVersion;
    private final double nce;
    private final int cudaVisibleDevice, workerId;
    private final List<Chunk> chunks;

    public PersistentAIWorker(String modelDir, String outDir, String device, String instrument, double nce,
                              String aiMode, String aiVersion, int cudaVisibleDevice, int workerId,
                              List<Chunk> chunks) {
        this.modelDir = modelDir; this.outDir = outDir; this.device = device; this.instrument = instrument;
        this.nce = nce; this.aiMode = aiMode; this.aiVersion = aiVersion;
        this.cudaVisibleDevice = cudaVisibleDevice; this.workerId = workerId; this.chunks = chunks;
    }

    /**
     * Materialise the fine-tuned model files once on local storage. The directory
     * name is a SHA-256 of the actual file contents, so a completed cache can
     * never be reused for a changed model. Incomplete cache entries have no ready
     * marker and are rebuilt safely.
     */
    public static String stageLocalModelCache(String modelDir) throws IOException {
        if (modelDir == null || modelDir.equalsIgnoreCase("generic")) return modelDir;
        Path source = Path.of(modelDir);
        if (!Files.isRegularFile(source.resolve("ms2_model.pt")) || !Files.isRegularFile(source.resolve("rt_model.pt"))) {
            Cloger.getInstance().logger.warn("Fine-tuned MS2/RT model files are incomplete; keeping original model path: " + modelDir);
            return modelDir;
        }
        String key = modelCacheKey(source);
        Path cacheRoot = Path.of(System.getProperty("java.io.tmpdir"), "carafe2-model-cache");
        Path cache = cacheRoot.resolve(key);
        Path ready = cache.resolve(".complete");
        if (Files.isRegularFile(ready)) {
            Cloger.getInstance().logger.info("Model cache hit: " + cache);
            return cache.toString();
        }
        Files.createDirectories(cacheRoot);
        Path staging = Files.createTempDirectory(cacheRoot, key + ".staging-");
        try {
            for (String name : CACHE_FILES) {
                Path file = source.resolve(name);
                if (Files.isRegularFile(file)) Files.copy(file, staging.resolve(name), StandardCopyOption.COPY_ATTRIBUTES);
            }
            Files.writeString(staging.resolve(".complete"), "source=" + source.toAbsolutePath() + "\nkey=" + key + "\n", StandardCharsets.UTF_8);
            try {
                Files.move(staging, cache, StandardCopyOption.ATOMIC_MOVE);
            } catch (java.nio.file.FileAlreadyExistsException ignored) {
                deleteRecursively(staging);
            } catch (java.nio.file.AtomicMoveNotSupportedException e) {
                try { Files.move(staging, cache); } catch (java.nio.file.FileAlreadyExistsException ignored) { deleteRecursively(staging); }
            }
            if (!Files.isRegularFile(ready)) throw new IOException("Model cache was not completed: " + cache);
            Cloger.getInstance().logger.info("Model cache populated: " + cache);
            return cache.toString();
        } catch (IOException e) {
            deleteRecursively(staging);
            throw e;
        }
    }

    private static String modelCacheKey(Path source) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] buffer = new byte[1024 * 1024];
            for (String name : CACHE_FILES) {
                Path file = source.resolve(name);
                if (!Files.isRegularFile(file)) continue;
                digest.update(name.getBytes(StandardCharsets.UTF_8));
                try (var in = Files.newInputStream(file)) {
                    for (int read; (read = in.read(buffer)) >= 0;) digest.update(buffer, 0, read);
                }
            }
            StringBuilder hex = new StringBuilder();
            for (byte b : digest.digest()) hex.append(String.format("%02x", b));
            return hex.toString();
        } catch (NoSuchAlgorithmException e) { throw new IllegalStateException("SHA-256 is unavailable", e); }
    }

    private static void deleteRecursively(Path root) throws IOException {
        if (!Files.exists(root)) return;
        try (var paths = Files.walk(root)) {
            paths.sorted(java.util.Comparator.reverseOrder()).forEach(path -> {
                try { Files.deleteIfExists(path); } catch (IOException ignored) { }
            });
        }
    }

    @Override public void run() {
        if (!aiVersion.equalsIgnoreCase("v2")) {
            throw new IllegalArgumentException("Persistent prediction workers require AI script version v2.");
        }
        Path manifest;
        try {
            manifest = Files.createTempFile(Path.of(outDir), ".carafe-prediction-worker-" + workerId + "-", ".tsv");
            try (BufferedWriter writer = Files.newBufferedWriter(manifest, StandardCharsets.UTF_8)) {
                for (Chunk chunk : chunks) {
                    writer.write(chunk.inputFile + "\t" + chunk.outPrefix + "\t" + chunk.peptideForms + "\n");
                }
            }
        } catch (IOException e) { throw new RuntimeException("Could not create persistent prediction manifest", e); }

        String aiPred = get_jar_path() + File.separator + "v2" + File.separator + "ai_pred.py";
        if (!new File(aiPred).exists()) aiPred = get_py_path("/py/v2/ai_pred.py", "carafe_ai_pred", true);
        String mode = aiMode.equals("-") ? "general" : aiMode;
        ArrayList<String> command = new ArrayList<>(List.of(
                AIWorker.python_bin, "-u", aiPred, "--model_dir", modelDir,
                "--chunk_manifest", manifest.toString(), "--out_dir", outDir,
                "--device", device, "--instrument", instrument, "--tf_type", CParameter.tf_type,
                "--nce", String.valueOf(nce), "--mode", mode, "--verbose", String.valueOf(CParameter.verbose.getValue())));
        if (AIWorker.torch_compile) command.add("--torch_compile");
        if (AIWorker.fast_mode) command.add("--fast");
        if (AIWorker.ccs_enabled) command.add("--ccs");
        if (!AIWorker.mod2mass.equals("-")) { command.add("--mod2mass"); command.add(AIWorker.mod2mass); }
        if (!AIWorker.user_mod.isEmpty()) { command.add("--user_mod"); command.add(AIWorker.user_mod); }
        Cloger.getInstance().logger.info("Persistent prediction worker " + workerId + " has " + chunks.size() + " chunks on GPU " + cudaVisibleDevice);
        runCommand(command, manifest);
    }

    private void runCommand(List<String> command, Path manifest) {
        Process process;
        try {
            ProcessBuilder builder = new ProcessBuilder(command);
            builder.environment().put("CUDA_VISIBLE_DEVICES", String.valueOf(cudaVisibleDevice));
            process = builder.start();
        } catch (IOException e) { throw new RuntimeException("Could not start persistent prediction worker", e); }
        StreamLog error = new StreamLog(process.getErrorStream(), "Model prediction: persistent-worker-" + workerId + " => Error:", true);
        StreamLog output = new StreamLog(process.getInputStream(), "Model prediction: persistent-worker-" + workerId + " => Message:", true, line -> {
            if (!line.startsWith("CARAFE_CHUNK_DONE\t")) return;
            String[] fields = line.split("\\t", -1);
            if (fields.length == 3) {
                try { AIWorker.reportPredictionProgress(Integer.parseInt(fields[2])); }
                catch (NumberFormatException ignored) { Cloger.getInstance().logger.warn("Invalid persistent-worker progress line: " + line); }
            }
        });
        error.start(); output.start();
        try {
            int exit = process.waitFor(); error.join(); output.join();
            if (exit != 0) throw new RuntimeException("Persistent prediction worker " + workerId + " exited with code " + exit + ": " + StringUtils.join(command, ' '));
        } catch (InterruptedException e) { Thread.currentThread().interrupt(); throw new RuntimeException(e); }
        finally { try { Files.deleteIfExists(manifest); } catch (IOException ignored) {} }
    }
}
