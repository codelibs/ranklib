
/*===============================================================================
 * Copyright (c) 2010-2015 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

//- Some Java 7 file utilities for creating directories
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.RankLibError;

/**
 * @author vdang
 *
 * This class implements the generic Ranker interface. Each ranking algorithm implemented has to extend this class.
 */
public abstract class Ranker {
    private static final Logger logger = Logger.getLogger(Ranker.class.getName());

    protected List<RankList> samples = new ArrayList<>();//training samples
    protected int[] features = null;
    protected MetricScorer scorer = null;
    protected double scoreOnTrainingData = 0.0;
    protected double bestScoreOnValidationData = 0.0;

    protected List<RankList> validationSamples = null;
    protected StringBuilder logBuf = new StringBuilder(1000);

    protected Ranker() {

    }

    protected Ranker(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        this.samples = samples;
        this.features = features;
        this.scorer = scorer;
    }

    //Utility functions
    public void setTrainingSet(final List<RankList> samples) {
        this.samples = samples;

    }

    public void setFeatures(final int[] features) {
        this.features = features;
    }

    public void setValidationSet(final List<RankList> samples) {
        this.validationSamples = samples;
    }

    public void setMetricScorer(final MetricScorer scorer) {
        this.scorer = scorer;
    }

    public double getScoreOnTrainingData() {
        return scoreOnTrainingData;
    }

    public double getScoreOnValidationData() {
        return bestScoreOnValidationData;
    }

    public int[] getFeatures() {
        return features;
    }

    public RankList rank(final RankList rl) {
        final double[] scores = new double[rl.size()];
        for (int i = 0; i < rl.size(); i++) {
            scores[i] = eval(rl.get(i));
        }
        final int[] idx = MergeSorter.sort(scores, false);
        return new RankList(rl, idx);
    }

    public List<RankList> rank(final List<RankList> l) {
        final List<RankList> ll = new ArrayList<>(l.size());
        for (int i = 0; i < l.size(); i++) {
            ll.add(rank(l.get(i)));
        }
        return ll;
    }

    //- Create the model file directory to write models into if not already there
    public void save(final String modelFile) {
        // Determine if the directory to write to exists.  If not, create it.
        final Path parentPath = Paths.get(modelFile).toAbsolutePath().getParent();

        // Create the directory if it doesn't exist. Give it 755 perms
        if (Files.notExists(parentPath)) {
            try {
                final Set<PosixFilePermission> perms = PosixFilePermissions.fromString("rwxr-xr-x");
                final FileAttribute<Set<PosixFilePermission>> attr = PosixFilePermissions.asFileAttribute(perms);
                Files.createDirectory(parentPath, attr);
            } catch (final Exception e) {
                throw RankLibError.create("Error creating kcv model file directory " + modelFile, e);
            }
        }

        FileUtils.write(modelFile, "ASCII", model());
    }

    protected void printLog(final int[] len, final String[] msgs) {
        if (logger.isLoggable(Level.INFO)) {
            for (int i = 0; i < msgs.length; i++) {
                final String msg = msgs[i];
                if (msg.length() > len[i]) {
                    logBuf.append(msg.substring(0, len[i]));
                } else {
                    logBuf.append(msg);
                    for (int j = len[i] - msg.length(); j > 0; j--) {
                        logBuf.append(' ');
                    }
                }
                logBuf.append(" | ");
            }
        }
    }

    protected void printLogLn(final int[] len, final String[] msgs) {
        if (logger.isLoggable(Level.INFO)) {
            printLog(len, msgs);
            flushLog();
        }
    }

    protected void flushLog() {
        if (logger.isLoggable(Level.INFO)) {
            if (logBuf.length() > 0) {
                logger.info(logBuf.toString());
                logBuf.setLength(0);
            }
        }
    }

    protected void copy(final double[] source, final double[] target) {
        for (int j = 0; j < source.length; j++) {
            target[j] = source[j];
        }
    }

    /**
     * HAVE TO BE OVER-RIDDEN IN SUB-CLASSES
     */
    public abstract void init();

    public abstract void learn();

    public double eval(final DataPoint p) {
        return -1.0;
    }

    public abstract Ranker createNew();

    @Override
    public abstract String toString();

    public abstract String model();

    public abstract void loadFromString(String fullText);

    public abstract String name();

    public abstract void printParameters();
}
