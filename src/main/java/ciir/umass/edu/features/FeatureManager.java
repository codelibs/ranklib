/*===============================================================================
 * Copyright (c) 2010-2016 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.features;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.DenseDataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.SparseDataPoint;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;

public class FeatureManager {
    private static final Logger logger = Logger.getLogger(FeatureManager.class.getName());

    /**
     * @param args
     */
    public static void main(final String[] args) {

        final List<String> rankingFiles = new ArrayList<>();
        String outputDir = "";
        String modelFileName = "";
        boolean shuffle = false;
        boolean doFeatureStats = false;

        int nFold = 0;
        float tvs = -1;//train-validation split in each fold
        float tts = -1;//train-test validation split of the whole dataset
        final int argsLen = args.length;

        if ((argsLen < 3) && !Arrays.asList(args).contains("-feature_stats")
                || (argsLen != 2) && Arrays.asList(args).contains("-feature_stats")) {
            logger.info(() -> "Usage: java -cp bin/RankLib.jar ciir.umass.edu.features.FeatureManager <Params>");
            logger.info(() -> "Params:");
            logger.info(() -> "\t-input <file>\t\tSource data (ranked lists)");
            logger.info(() -> "\t-output <dir>\t\tThe output directory");

            logger.info(() -> "  [+] Shuffling");
            logger.info(
                    () -> "\t-shuffle\t\tCreate a copy of the input file in which the ordering of all ranked lists (e.g. queries) is randomized.");
            logger.info(() -> "\t\t\t\t(the order among objects (e.g. documents) within each ranked list is certainly unchanged).");

            logger.info(() -> "  [+] k-fold Partitioning (sequential split)");
            logger.info(() -> "\t-k <fold>\t\tThe number of folds");
            logger.info(() -> "\t[ -tvs <x \\in [0..1]> ] Train-validation split ratio (x)(1.0-x)");
            logger.info(() -> "  [+] Train-test split");
            logger.info(() -> "\t-tts <x \\in [0..1]> ] Train-test split ratio (x)(1.0-x)");

            logger.info(
                    () -> "  NOTE: If both -shuffle and -k are specified, the input data will be shuffled and then sequentially partitioned.");

            logger.info(() -> "Feature Statistics -- Saved model feature use frequencies and statistics.");
            logger.info(() -> "-input and -output parameters are not used.");
            logger.info(() -> "\t-feature_stats\tName of a saved, feature-limited, LTR model text file.");
            logger.info(() -> "\t\t\tDoes not process Coordinate Ascent, LambdaRank, ListNet or RankNet models.");
            logger.info(() -> "\t\t\tas they include all features rather than selected feature subsets.");
            return;
        }

        for (int i = 0; i < args.length; i++) {
            if (args[i].equalsIgnoreCase("-input")) {
                rankingFiles.add(args[++i]);
            } else if (args[i].equalsIgnoreCase("-k")) {
                nFold = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-shuffle")) {
                shuffle = true;
            } else if (args[i].equalsIgnoreCase("-tvs")) {
                tvs = Float.parseFloat(args[++i]);
            } else if (args[i].equalsIgnoreCase("-tts")) {
                tts = Float.parseFloat(args[++i]);
            } else if (args[i].equalsIgnoreCase("-output")) {
                outputDir = FileUtils.makePathStandard(args[++i]);
            } else if (args[i].equalsIgnoreCase("-feature_stats")) {
                doFeatureStats = true;
                modelFileName = args[++i];
            }
        }

        if (nFold > 0 && tts != -1) {
            logger.info(() -> "Error: Only one of k or tts should be specified.");
            return;
        }

        if (shuffle || nFold > 0 || tts != -1) {
            final List<RankList> samples = readInput(rankingFiles);

            if (samples.isEmpty()) {
                logger.info(() -> "Error: The input file is empty.");
                return;
            }

            String fn = FileUtils.getFileName(rankingFiles.get(0));

            if (shuffle) {
                fn += ".shuffled";
                logger.info(() -> "Shuffling... ");
                Collections.shuffle(samples);
                logger.info(() -> "Saving... ");
                FeatureManager.save(samples, outputDir + fn);
            }

            if (tts != -1) {
                final List<RankList> trains = new ArrayList<>();
                final List<RankList> tests = new ArrayList<>();

                logger.info(() -> "Splitting... ");
                prepareSplit(samples, tts, trains, tests);

                try {
                    logger.info(() -> "Saving splits...");
                    save(trains, outputDir + "train." + fn);
                    save(tests, outputDir + "test." + fn);
                } catch (final Exception ex) {
                    throw RankLibError.create("Cannot save partition data.\n" + "Occured in FeatureManager::main(): ", ex);
                }
            }

            if (nFold > 0) {
                final List<List<RankList>> trains = new ArrayList<>();
                final List<List<RankList>> tests = new ArrayList<>();
                final List<List<RankList>> valis = new ArrayList<>();
                logger.info(() -> "Partitioning... ");
                prepareCV(samples, nFold, tvs, trains, valis, tests);

                try {
                    for (int i = 0; i < trains.size(); i++) {
                        if (logger.isLoggable(Level.INFO)) {
                            logger.info("Saving fold " + (i + 1) + "/" + nFold + "... ");
                        }
                        save(trains.get(i), outputDir + "f" + (i + 1) + ".train." + fn);
                        save(tests.get(i), outputDir + "f" + (i + 1) + ".test." + fn);
                        if (tvs > 0) {
                            save(valis.get(i), outputDir + "f" + (i + 1) + ".validation." + fn);
                        }
                    }
                } catch (final Exception ex) {
                    throw RankLibError.create("Cannot save partition data.\n" + "Occured in FeatureManager::main(): ", ex);
                }
            }
        } else if (doFeatureStats) {
            //- Produce some a frequency distribution of chosen model features with some statistics.
            try {
                final FeatureStats fs = new FeatureStats(modelFileName);
                fs.writeFeatureStats();
            } catch (final Exception ex) {
                throw RankLibError.create(
                        "Failure processing saved " + modelFileName + " model file.\n" + "Error occurred in FeatureManager::main(): ", ex);
            }
        }
    }

    /**
     * Read a set of rankings from a single file.
     * @param inputFile
     * @return
     */
    public static List<RankList> readInput(final String inputFile) {
        return readInput(inputFile, false, false);
    }

    /**
     * Read a set of rankings from a single file.
     * @param inputFile
     * @param mustHaveRelDoc
     * @param useSparseRepresentation
     * @return
     */
    public static List<RankList> readInput(final String inputFile, final boolean mustHaveRelDoc, final boolean useSparseRepresentation) {
        final List<RankList> samples = new ArrayList<>();
        final int countRL = 0;
        int countEntries = 0;

        try (final BufferedReader in = FileUtils.smartReader(inputFile)) {
            String content = "";

            String lastID = "";
            boolean hasRel = false;
            List<DataPoint> rl = new ArrayList<>();

            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }

                if (content.indexOf("#") == 0) {
                    continue;
                }

                if (countEntries % 10000 == 0) {
                    logger.info(() -> "\rReading feature file [" + inputFile + "]: " + countRL + "... ");
                }

                DataPoint qp = null;

                if (useSparseRepresentation) {
                    qp = new SparseDataPoint(content);
                } else {
                    qp = new DenseDataPoint(content);
                }

                if (lastID.compareTo("") != 0 && lastID.compareTo(qp.getID()) != 0) {
                    if (!mustHaveRelDoc || hasRel) {
                        samples.add(new RankList(rl));
                    }
                    rl = new ArrayList<>();
                    hasRel = false;
                }

                if (qp.getLabel() > 0) {
                    hasRel = true;
                }
                lastID = qp.getID();
                rl.add(qp);
                countEntries++;
            }

            if (!rl.isEmpty() && (!mustHaveRelDoc || hasRel)) {
                samples.add(new RankList(rl));
            }

            logger.info(() -> "\rReading feature file [" + inputFile + "]...");
            if (logger.isLoggable(Level.INFO)) {
                logger.info("(" + samples.size() + " ranked lists, " + countEntries + " entries read)");
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in FeatureManager::readInput(): ", ex);
        }
        return samples;
    }

    /**
     * Read sets of rankings from multiple files. Then merge them altogether into a single ranking.
     * @param inputFiles
     * @return
     */
    public static List<RankList> readInput(final List<String> inputFiles) {
        final List<RankList> samples = new ArrayList<>();

        for (int i = 0; i < inputFiles.size(); i++) {
            final List<RankList> s = readInput(inputFiles.get(i), false, false);
            samples.addAll(s);
        }
        return samples;
    }

    /**
     * Read features specified in an input feature file. Expecting one feature per line.
     * @param featureDefFile
     * @return
     */
    public static int[] readFeature(final String featureDefFile) {
        int[] features = null;
        final List<String> fids = new ArrayList<>();

        try (BufferedReader in = FileUtils.smartReader(featureDefFile)) {
            String content = null;

            while ((content = in.readLine()) != null) {
                content = content.trim();

                if (content.length() == 0 || content.indexOf("#") == 0) {
                    continue;
                }

                fids.add(content.split("\t")[0].trim());
            }
            features = new int[fids.size()];

            for (int i = 0; i < fids.size(); i++) {
                features[i] = Integer.parseInt(fids.get(i));
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in FeatureManager::readFeature(): ", ex);
        }
        return features;
    }

    /**
     * Obtain all features present in a sample set.
     * Important: If your data (DataPoint objects) is loaded by RankLib (e.g. command-line use) or its APIs, there
     * is nothing to watch out for.
     * If you create the DataPoint objects yourself, make sure DataPoint.featureCount correctly reflects
     * the total number features present in your dataset.
     * @param samples
     * @return
     */
    public static int[] getFeatureFromSampleVector(final List<RankList> samples) {
        if (samples.isEmpty()) {
            throw RankLibError.create("Error in FeatureManager::getFeatureFromSampleVector(): There are no training samples.");
        }

        final int fc = DataPoint.getFeatureCount();
        final int[] features = new int[fc];

        for (int i = 1; i <= fc; i++) {
            features[i - 1] = i;
        }

        return features;
    }

    /**
     * Split the input sample set into k chunks (folds) of roughly equal size and create train/test data for each fold.
     * Note that NO randomization is done. If you want to randomly split the data, make sure that you randomize the order
     * in the input samples prior to calling this function.
     * @param samples
     * @param nFold
     * @param trainingData
     * @param testData
     */
    public static void prepareCV(final List<RankList> samples, final int nFold, final List<List<RankList>> trainingData,
            final List<List<RankList>> testData) {
        prepareCV(samples, nFold, -1, trainingData, null, testData);
    }

    /**
     * Split the input sample set into k chunks (folds) of roughly equal size and create train/test data for each fold. Then it further splits
     * the training data in each fold into train and validation. Note that NO randomization is done. If you want to randomly split the data,
     * make sure that you randomize the order in the input samples prior to calling this function.
     * @param samples
     * @param nFold
     * @param tvs Train/validation split ratio
     * @param trainingData
     * @param validationData
     * @param testData
     */
    public static void prepareCV(final List<RankList> samples, final int nFold, final float tvs, final List<List<RankList>> trainingData,
            final List<List<RankList>> validationData, final List<List<RankList>> testData) {
        final List<List<Integer>> trainSamplesIdx = new ArrayList<>();
        final int size = samples.size() / nFold;
        int start = 0;
        int total = 0;

        for (int f = 0; f < nFold; f++) {
            final List<Integer> t = new ArrayList<>();
            for (int i = 0; i < size && start + i < samples.size(); i++) {
                t.add(start + i);
            }
            trainSamplesIdx.add(t);
            total += t.size();
            start += size;
        }

        for (; total < samples.size(); total++) {
            trainSamplesIdx.get(trainSamplesIdx.size() - 1).add(total);
        }

        for (int i = 0; i < trainSamplesIdx.size(); i++) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("\rCreating data for fold-" + (i + 1) + "...");
            }
            final List<RankList> train = new ArrayList<>();
            final List<RankList> test = new ArrayList<>();
            final List<RankList> vali = new ArrayList<>();

            //train-test split
            final List<Integer> t = trainSamplesIdx.get(i);

            for (int j = 0; j < samples.size(); j++) {
                if (t.contains(j)) {
                    test.add(new RankList(samples.get(j)));
                } else {
                    train.add(new RankList(samples.get(j)));
                }
            }

            //train-validation split if specified
            if (tvs > 0) {
                final int validationSize = (int) (train.size() * (1.0 - tvs));
                for (int j = 0; j < validationSize; j++) {
                    vali.add(train.get(train.size() - 1));
                    train.remove(train.size() - 1);
                }
            }

            //save them
            trainingData.add(train);
            testData.add(test);

            if (tvs > 0) {
                validationData.add(vali);
            }
        }
        logger.info(() -> "\rCreating data for " + nFold + " folds...");

        printQueriesForSplit("Train", trainingData);
        printQueriesForSplit("Validate", validationData);
        printQueriesForSplit("Test", testData);
    }

    public static void printQueriesForSplit(final String name, final List<List<RankList>> split) {
        if (split == null) {
            logger.info(() -> "No " + name + " split.");
            return;
        }
        if (logger.isLoggable(Level.INFO)) {
            for (int i = 0; i < split.size(); i++) {
                final List<RankList> rankLists = split.get(i);
                logger.info(name + "[" + i + "]=");

                for (final RankList rankList : rankLists) {
                    logger.info(() -> " \"" + rankList.getID() + "\"");
                }
            }
        }
    }

    /**
     * Split the input sample set into 2 chunks: one for training and one for either validation or testing
     * @param samples
     * @param percentTrain The percentage of data used for training
     * @param trainingData
     * @param testData
     */
    public static void prepareSplit(final List<RankList> samples, final double percentTrain, final List<RankList> trainingData,
            final List<RankList> testData) {
        final int size = (int) (samples.size() * percentTrain);

        for (int i = 0; i < size; i++) {
            trainingData.add(new RankList(samples.get(i)));
        }

        for (int i = size; i < samples.size(); i++) {
            testData.add(new RankList(samples.get(i)));
        }
    }

    /**
     * Save a sample set to file
     * @param samples
     * @param outputFile
     */
    public static void save(final List<RankList> samples, final String outputFile) {
        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile)))) {
            for (final RankList sample : samples) {
                save(sample, out);
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in FeatureManager::save(): ", ex);
        }
    }

    /**
     * Write a ranked list to a file object.
     * @param r
     * @param out
     * @throws Exception
     */
    private static void save(final RankList r, final BufferedWriter out) throws Exception {
        for (int j = 0; j < r.size(); j++) {
            out.write(r.get(j).toString());
            out.newLine();
        }
    }

} //- end class FeatureManager
