package main.java.ai;

import main.java.input.CParameter;
import main.java.util.StreamLog;
import main.java.util.Cloger;
import org.apache.commons.lang3.StringUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static main.java.ai.AIGear.get_jar_path;
import static main.java.ai.AIGear.get_py_path;

public final  class AIWorker implements Runnable{

    public static String python_bin;
    public String model_dir;
    public String input_file;
    public String out_dir;
    public String out_prefix;
    public String device;
    public String ms_instrument;
    public double nce;
    public String ai_mode = "-";
    public String ai_version = "v1";
    /**
     * Use torch.compile to speed up training and inference.
     * Only available in v2.
     */
    public static boolean torch_compile = false;
    public static boolean fast_mode = false;
    public static boolean ccs_enabled = false;
    public static String mod2mass = "-";
    public static String user_mod = "";
    private static final AtomicInteger completedPredictionChunks = new AtomicInteger(0);
    private static final AtomicInteger completedPeptideForms = new AtomicInteger(0);
    private static volatile int totalPredictionChunks = 0;
    private static volatile int totalPeptideForms = 0;
    private static volatile long predictionStartTimeMillis = 0L;
    private final int peptideFormsInChunk;
    private final int cudaVisibleDevice;

    public AIWorker(String model_dir, String input_file, String out_dir, String out_prefix, String device, String ms_instrument, double nce, String ai_mode, String ai_version){
        this(model_dir, input_file, out_dir, out_prefix, device, ms_instrument, nce, ai_mode, ai_version, -1, -1);
    }

    public AIWorker(String model_dir, String input_file, String out_dir, String out_prefix, String device, String ms_instrument, double nce, String ai_mode, String ai_version, int peptideFormsInChunk){
        this(model_dir, input_file, out_dir, out_prefix, device, ms_instrument, nce, ai_mode, ai_version, peptideFormsInChunk, -1);
    }

    public AIWorker(String model_dir, String input_file, String out_dir, String out_prefix, String device, String ms_instrument, double nce, String ai_mode, String ai_version, int peptideFormsInChunk, int cudaVisibleDevice){

        this.model_dir = model_dir;
        this.input_file = input_file;
        this.out_dir = out_dir;
        this.out_prefix = out_prefix;
        this.device = device;
        this.ms_instrument = ms_instrument;
        this.nce = nce;
        this.ai_mode = ai_mode;
        this.ai_version = ai_version;
        this.peptideFormsInChunk = peptideFormsInChunk;
        this.cudaVisibleDevice = cudaVisibleDevice;
    }

    public static void initialisePredictionProgress(int peptideForms, int chunks) {
        completedPredictionChunks.set(0);
        completedPeptideForms.set(0);
        totalPeptideForms = peptideForms;
        totalPredictionChunks = chunks;
        predictionStartTimeMillis = System.currentTimeMillis();
        Cloger.getInstance().logger.info("Prediction progress: 0/" + peptideForms +
                " peptide forms (0/" + chunks + " chunks)");
    }

    private static String formatDuration(long seconds) {
        long hours = seconds / 3600;
        long minutes = (seconds % 3600) / 60;
        long remainingSeconds = seconds % 60;
        return String.format("%02d:%02d:%02d", hours, minutes, remainingSeconds);
    }

    public static void reportPredictionProgress(int peptideFormsInChunk) {
        if (peptideFormsInChunk < 0 || totalPeptideForms <= 0 || predictionStartTimeMillis <= 0L) {
            return;
        }
        int completedForms = completedPeptideForms.addAndGet(peptideFormsInChunk);
        int completedChunks = completedPredictionChunks.incrementAndGet();
        long elapsedSeconds = Math.max(1L, (System.currentTimeMillis() - predictionStartTimeMillis) / 1000L);
        long remainingSeconds = completedForms > 0
                ? Math.max(0L, Math.round((double) elapsedSeconds * (totalPeptideForms - completedForms) / completedForms))
                : 0L;
        double percent = 100.0 * completedForms / totalPeptideForms;
        Cloger.getInstance().logger.info(String.format(
                "Prediction progress: %,d/%,d peptide forms (%.1f%%), chunk %d/%d complete, ETA %s",
                completedForms, totalPeptideForms, percent, completedChunks, totalPredictionChunks,
                formatDuration(remainingSeconds)));
    }

    private void reportPredictionProgress() {
        reportPredictionProgress(peptideFormsInChunk);
    }

    @Override
    public void run() {
        Cloger.getInstance().logger.info(Thread.currentThread().getName()+": predicting "+this.input_file);
        String mode = this.ai_mode.equals("-")? "general": this.ai_mode;
        String ai_pred_name = this.ai_version.equalsIgnoreCase("v2") ? "v2/ai_pred.py" : "ai_pred.py";
        String ai_pred = get_jar_path() + File.separator + ai_pred_name.replace("/", File.separator);
        File F = new File(ai_pred);
        String py_resource_root = this.ai_version.equalsIgnoreCase("v2") ? "/py/" : "/main/java/ai/";
        if(!F.exists()){
            ai_pred = get_py_path(py_resource_root + ai_pred_name, "carafe_ai_pred", this.ai_version.equalsIgnoreCase("v2"));
        }

        String[] cmd_list_short = new String[] {
                python_bin,
                "-u",  // Forces immediate flushing to pipes
                ai_pred,
                "--model_dir", model_dir,
                "--in_file", this.input_file,
                "--out_dir", this.out_dir,
                "--out_prefix", this.out_prefix,
                "--device", this.device,
                "--instrument", this.ms_instrument,
                "--tf_type", CParameter.tf_type,
                "--nce", String.valueOf(this.nce),
                "--mode", mode,
                "--verbose", String.valueOf(CParameter.verbose.getValue())
        };
        ArrayList<String> cmd_list =  new ArrayList<>(Arrays.asList(cmd_list_short));

        if(torch_compile && this.ai_version.equalsIgnoreCase("v2")){
            // cmd += " --torch_compile";
            cmd_list.add("--torch_compile");
        }

        if(fast_mode){
            // cmd += " --fast";
            cmd_list.add("--fast");
        }
        if(ccs_enabled){
            // cmd += " --ccs";
            cmd_list.add("--ccs");
        }
        if(!mod2mass.equals("-")){
            // cmd += " --mod2mass \"" + mod2mass + "\"";
            cmd_list.add("--mod2mass");
            cmd_list.add("\"" + mod2mass + "\"");
        }
        if(!user_mod.isEmpty()){
            // cmd += " --user_mod \"" + user_mod + "\"";
            cmd_list.add("--user_mod");
            cmd_list.add("\"" + user_mod + "\"");
        }
        System.out.println("cmd: "+ StringUtils.join(cmd_list));
        // convert cmd_list to String []
        String[] cmd = new String[cmd_list.size()];
        cmd = cmd_list.toArray(cmd);
        run_cmd(cmd);
        reportPredictionProgress();
    }

    private boolean run_cmd(String[] cmd){
        boolean pass = true;
        Process p;
        try {
            ProcessBuilder processBuilder = new ProcessBuilder(cmd);
            if (cudaVisibleDevice >= 0) {
                processBuilder.environment().put("CUDA_VISIBLE_DEVICES", String.valueOf(cudaVisibleDevice));
            }
            p = processBuilder.start();
        } catch (IOException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        StreamLog errorLog = new StreamLog(p.getErrorStream(), "Model prediction: " + Thread.currentThread().getName()+" => Error:", true);
        StreamLog stdLog = new StreamLog(p.getInputStream(), "Model prediction: " + Thread.currentThread().getName()+" => Message:", true);

        errorLog.start();
        stdLog.start();

        try {
            int exitValue = p.waitFor();
            if (exitValue != 0) {
                pass = false;
                Cloger.getInstance().logger.error("Model prediction: " + Thread.currentThread().getName()+" => Error:" + exitValue);
            }
        } catch (InterruptedException e) {
            pass = false;
            throw new RuntimeException(e);
        }

        try {
            errorLog.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        try {
            stdLog.join();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }

        if (!pass) {
            throw new RuntimeException("Model prediction process failed when running: "+StringUtils.join(cmd));
        }

        return pass;
    }
}
