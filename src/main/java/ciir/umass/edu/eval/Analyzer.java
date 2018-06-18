package ciir.umass.edu.eval;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.stats.RandomPermutationTest;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

public class Analyzer {
    private static final Logger logger = Logger.getLogger(Analyzer.class.getName());

    /**
     * @param args
     */
    public static void main(final String[] args) {
        String directory = "";
        String baseline = "";
        if (args.length < 2) {
            logger.info(() -> "Usage: java -cp bin/RankLib.jar ciir.umass.edu.eval.Analyzer <Params>");
            logger.info(() -> "Params:");
            logger.info(() -> "\t-all <directory>\tDirectory of performance files (one per system)");
            logger.info(() -> "\t-base <file>\t\tPerformance file for the baseline (MUST be in the same directory)");
            logger.info(() -> "\t[ -np ] \t\tNumber of permutation (Fisher randomization test) [default="
                    + RandomPermutationTest.nPermutation + "]");
            return;
        }

        for (int i = 0; i < args.length; i++) {
            if (args[i].compareTo("-all") == 0) {
                directory = args[++i];
            } else if (args[i].compareTo("-base") == 0) {
                baseline = args[++i];
            } else if (args[i].compareTo("-np") == 0) {
                RandomPermutationTest.nPermutation = Integer.parseInt(args[++i]);
            }
        }

        final Analyzer a = new Analyzer();
        a.compare(directory, baseline);
        //a.compare("output/", "ca.feature.base");
    }

    static class Result {
        int status = 0;//success
        int win = 0;
        int loss = 0;
        int[] countByImprovementRange = null;
    }

    private final RandomPermutationTest randomizedTest = new RandomPermutationTest();
    private static double[] improvementRatioThreshold = new double[] { -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1000 };
    private final int indexOfZero = 4;

    private int locateSegment(final double value) {
        if (value > 0) {
            for (int i = indexOfZero; i < improvementRatioThreshold.length; i++) {
                if (value <= improvementRatioThreshold[i]) {
                    return i;
                }
            }
        } else if (value < 0) {
            for (int i = 0; i <= indexOfZero; i++) {
                if (value < improvementRatioThreshold[i]) {
                    return i;
                }
            }
        }
        return -1;
    }

    /**
     * Read performance (in some measure of effectiveness) file. Expecting: id [space]* metric-text [space]* performance
     * @param filename
     * @return Mapping from ranklist-id --> performance
     */
    public Map<String, Double> read(final String filename) {
        final Map<String, Double> performance = new HashMap<>();
        try (BufferedReader in = FileUtils.smartReader(filename)) {
            String content = null;
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }

                //expecting: id [space]* metric-text [space]* performance
                while (content.contains("  ")) {
                    content = content.replace("  ", " ");
                }
                content = content.replace(" ", "\t");
                final String[] s = content.split("\t");
                //String measure = s[0];
                final String id = s[1];
                final double p = Double.parseDouble(s[2]);
                performance.put(id, p);
            }
            logger.info(() -> "Reading " + filename + "... " + performance.size() + " ranked lists");
        } catch (final IOException ex) {
            throw RankLibError.create(ex);
        }
        return performance;
    }

    /**
     * Compare the performance of a set of systems to that of a baseline system
     * @param directory Contain files denoting the performance of the target systems to be compared
     * @param baseFile Performance file for the baseline system
     */
    public void compare(String directory, final String baseFile) {
        directory = FileUtils.makePathStandard(directory);
        final List<String> targets = FileUtils.getAllFiles2(directory);//ONLY filenames are stored
        for (int i = 0; i < targets.size(); i++) {
            if (targets.get(i).compareTo(baseFile) == 0) {
                targets.remove(i);
                i--;
            } else {
                targets.set(i, directory + targets.get(i));//convert filename to full path
            }
        }
        compare(targets, directory + baseFile);
    }

    /**
     * Compare the performance of a set of systems to that of a baseline system
     * @param targetFiles Performance files of the target systems to be compared (full path)
     * @param baseFile Performance file for the baseline system
     */
    public void compare(final List<String> targetFiles, final String baseFile) {
        final Map<String, Double> base = read(baseFile);
        final List<Map<String, Double>> targets = new ArrayList<>();
        for (int i = 0; i < targetFiles.size(); i++) {
            final Map<String, Double> hm = read(targetFiles.get(i));
            targets.add(hm);
        }
        final Result[] rs = compare(base, targets);

        //overall comparison
        logger.info(() -> "Overall comparison");
        logger.info(() -> "System\tPerformance\tImprovement\tWin\tLoss\tp-value");
        logger.info(() -> FileUtils.getFileName(baseFile) + " [baseline]\t" + SimpleMath.round(base.get("all").doubleValue(), 4));
        for (int i = 0; i < rs.length; i++) {
            if (rs[i].status == 0) {
                final double delta = targets.get(i).get("all") - base.get("all");
                final double dp = delta * 100 / base.get("all");
                logger.info(FileUtils.getFileName(targetFiles.get(i)) + "\t" + SimpleMath.round(targets.get(i).get("all").doubleValue(), 4)
                        + "\t" + ((delta > 0) ? "+" : "") + SimpleMath.round(delta, 4) + " (" + ((delta > 0) ? "+" : "")
                        + SimpleMath.round(dp, 2) + "%)" + "\t" + rs[i].win + "\t" + rs[i].loss + "\t"
                        + randomizedTest.test(targets.get(i), base));
            } else {
                logger.warning(
                        "WARNING: [" + targetFiles.get(i) + "] skipped: NOT comparable to the baseline due to different ranked list IDs.");
            }
        }
        //in more details
        logger.info(() -> "Detailed break down");
        if (logger.isLoggable(Level.INFO)) {
            String header = "";
            final String[] tmp = new String[improvementRatioThreshold.length];
            for (int i = 0; i < improvementRatioThreshold.length; i++) {
                String t = (int) (improvementRatioThreshold[i] * 100) + "%";
                if (improvementRatioThreshold[i] > 0) {
                    t = "+" + t;
                }
                tmp[i] = t;
            }
            header += "[ < " + tmp[0] + ")\t";
            for (int i = 0; i < improvementRatioThreshold.length - 2; i++) {
                if (i >= indexOfZero) {
                    header += "(" + tmp[i] + ", " + tmp[i + 1] + "]\t";
                } else {
                    header += "[" + tmp[i] + ", " + tmp[i + 1] + ")\t";
                }
            }
            header += "( > " + tmp[improvementRatioThreshold.length - 2] + "]";
            logger.info("\t" + header);

            for (int i = 0; i < targets.size(); i++) {
                String msg = FileUtils.getFileName(targetFiles.get(i));
                for (final int element : rs[i].countByImprovementRange) {
                    msg += "\t" + element;
                }
                logger.info(msg);
            }
        }
    }

    /**
     * Compare the performance of a set of systems to that of a baseline system
     * @param base
     * @param targets
     * @return
     */
    public Result[] compare(final Map<String, Double> base, final List<Map<String, Double>> targets) {
        //comparative statistics
        final Result[] rs = new Result[targets.size()];
        for (int i = 0; i < targets.size(); i++) {
            rs[i] = compare(base, targets.get(i));
        }
        return rs;
    }

    /**
     * Compare the performance of a target system to that of a baseline system
     * @param base
     * @param target
     * @return
     */
    public Result compare(final Map<String, Double> base, final Map<String, Double> target) {
        final Result r = new Result();
        if (base.size() != target.size()) {
            r.status = -1;
            return r;
        }

        r.countByImprovementRange = new int[improvementRatioThreshold.length];
        Arrays.fill(r.countByImprovementRange, 0);
        for (final String key : base.keySet()) {
            if (!target.containsKey(key)) {
                r.status = -2;
                return r;
            }
            if (key.compareTo("all") == 0) {
                continue;
            }
            final double p = base.get(key);
            final double pt = target.get(key);
            if (pt > p) {
                r.win++;
            } else if (pt < p) {
                r.loss++;
            }
            final double change = pt - p;
            if (change != 0) {
                r.countByImprovementRange[locateSegment(change)]++;
            }
        }
        return r;
    }
}
