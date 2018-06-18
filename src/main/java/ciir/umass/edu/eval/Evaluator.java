/*===============================================================================
 * Copyright (c) 2010-2016 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.eval;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.features.FeatureManager;
import ciir.umass.edu.features.LinearNormalizer;
import ciir.umass.edu.features.Normalizer;
import ciir.umass.edu.features.SumNormalizor;
import ciir.umass.edu.features.ZScoreNormalizor;
import ciir.umass.edu.learning.CoorAscent;
import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.LinearRegRank;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.RankerFactory;
import ciir.umass.edu.learning.RankerTrainer;
import ciir.umass.edu.learning.RankerType;
import ciir.umass.edu.learning.boosting.AdaRank;
import ciir.umass.edu.learning.boosting.RankBoost;
import ciir.umass.edu.learning.neuralnet.ListNet;
import ciir.umass.edu.learning.neuralnet.Neuron;
import ciir.umass.edu.learning.neuralnet.RankNet;
import ciir.umass.edu.learning.tree.LambdaMART;
import ciir.umass.edu.learning.tree.RFRanker;
import ciir.umass.edu.metric.ERRScorer;
import ciir.umass.edu.metric.METRIC;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.metric.MetricScorerFactory;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.MyThreadPool;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 *
 * This class is meant to provide the interface to run and compare different ranking algorithms.
 * It lets users specify general parameters (e.g. what algorithm to run, training/testing/validating
 *  data, etc.) as well as algorithm-specific parameters. Type "java -jar bin/RankLib.jar" at the
 *  command-line to see all the options.
 */
public class Evaluator {
    private static final Logger logger = Logger.getLogger(Evaluator.class.getName());

    /**
     * @param args
     */
    public static void main(final String[] args) {

        final String[] rType = new String[] { "MART", "RankNet", "RankBoost", "AdaRank", "Coordinate Ascent", "LambdaRank", "LambdaMART",
                "ListNet", "Random Forests", "Linear Regression" };
        final RankerType[] rType2 = new RankerType[] { RankerType.MART, RankerType.RANKNET, RankerType.RANKBOOST, RankerType.ADARANK,
                RankerType.COOR_ASCENT, RankerType.LAMBDARANK, RankerType.LAMBDAMART, RankerType.LISTNET, RankerType.RANDOM_FOREST,
                RankerType.LINEAR_REGRESSION };

        String trainFile = "";
        String featureDescriptionFile = "";
        float ttSplit = 0;//train-test split
        float tvSplit = 0;//train-validation split
        int foldCV = -1;
        String validationFile = "";
        String testFile = "";
        final List<String> testFiles = new ArrayList<>();
        int rankerType = 4;
        String trainMetric = "ERR@10";
        String testMetric = "";
        Evaluator.normalize = false;
        String savedModelFile = "";
        final List<String> savedModelFiles = new ArrayList<>();
        String kcvModelDir = "";
        String kcvModelFile = "";
        String rankFile = "";
        String prpFile = "";

        int nThread = -1; // nThread = #cpu-cores
        //for my personal use
        String indriRankingFile = "";
        String scoreFile = "";

        if (args.length < 2) {
            logger.info(() -> "Usage: java -jar RankLib.jar <Params>");
            logger.info(() -> "Params:");
            logger.info(() -> "  [+] Training (+ tuning and evaluation)");
            logger.info(() -> "\t-train <file>\t\tTraining data");
            logger.info(() -> "\t-ranker <type>\t\tSpecify which ranking algorithm to use");
            logger.info(() -> "\t\t\t\t0: MART (gradient boosted regression tree)");
            logger.info(() -> "\t\t\t\t1: RankNet");
            logger.info(() -> "\t\t\t\t2: RankBoost");
            logger.info(() -> "\t\t\t\t3: AdaRank");
            logger.info(() -> "\t\t\t\t4: Coordinate Ascent");
            logger.info(() -> "\t\t\t\t6: LambdaMART");
            logger.info(() -> "\t\t\t\t7: ListNet");
            logger.info(() -> "\t\t\t\t8: Random Forests");
            logger.info(() -> "\t\t\t\t9: Linear regression (L2 regularization)");
            logger.info(
                    () -> "\t[ -feature <file> ]\tFeature description file: list features to be considered by the learner, each on a separate line");
            logger.info(() -> "\t\t\t\tIf not specified, all features will be used.");
            //logger.info(()->"\t[ -metric2t <metric> ]\tMetric to optimize on the training data. Supported: MAP, NDCG@k, DCG@k, P@k, RR@k, BEST@k, ERR@k (default=" + trainMetric + ")");
            logger.info(() -> "\t[ -metric2t <metric> ]\tMetric to optimize on the training data.  "
                    + "Supported: MAP, NDCG@k, DCG@k, P@k, RR@k, ERR@k (default=ERR@10)");
            logger.info(() -> "\t[ -gmax <label> ]\tHighest judged relevance label. It affects the calculation of ERR " + "(default="
                    + (int) SimpleMath.logBase2(ERRScorer.MAX) + ", i.e. 5-point scale {0,1,2,3,4})");
            logger.info(() -> "\t[ -qrel <file> ]\tTREC-style relevance judgment file. It only affects MAP and NDCG (default=unspecified)");
            logger.info(() -> "\t[ -silent ]\t\tDo not print progress messages (which are printed by default)");
            logger.info(() -> "\t[ -missingZero ]\tSubstitute zero for missing feature values rather than throwing an exception.");

            //logger.info(()->"        Use the entire specified training data");
            logger.info(
                    () -> "\t[ -validate <file> ]\tSpecify if you want to tune your system on the validation data (default=unspecified)");
            logger.info(() -> "\t\t\t\tIf specified, the final model will be the one that performs best on the validation data");
            logger.info(
                    () -> "\t[ -tvs <x \\in [0..1]> ]\tIf you don't have separate validation data, use this to set train-validation split to be (x)(1.0-x)");

            logger.info(() -> "\t[ -save <model> ]\tSave the model learned (default=not-save)");

            logger.info(() -> "\t[ -test <file> ]\tSpecify if you want to evaluate the trained model on this data (default=unspecified)");
            logger.info(() -> "\t[ -tts <x \\in [0..1]> ]\tSet train-test split to be (x)(1.0-x). -tts will override -tvs");
            logger.info(
                    () -> "\t[ -metric2T <metric> ]\tMetric to evaluate on the test data (default to the same as specified for -metric2t)");

            logger.info(() -> "\t[ -norm <method>]\tNormalize all feature vectors (default=no-normalization). Method can be:");
            logger.info(() -> "\t\t\t\tsum: normalize each feature by the sum of all its values");
            logger.info(() -> "\t\t\t\tzscore: normalize each feature by its mean/standard deviation");
            logger.info(() -> "\t\t\t\tlinear: normalize each feature by its min/max values");

            //logger.info(()->"\t[ -sparse ]\t\tUse sparse representation for all feature vectors (default=dense)");

            logger.info(
                    () -> "\t[ -kcv <k> ]\t\tSpecify if you want to perform k-fold cross validation using the specified training data (default=NoCV)");
            logger.info(() -> "\t\t\t\t-tvs can be used to further reserve a portion of the training data in each fold for validation");
            //logger.info(()->"\t\t\t\tData for each fold is created from sequential partitions of the training data.");
            //logger.info(()->"\t\t\t\tRandomized partitioning can be done by shuffling the training data in advance.");
            //logger.info(()->"\t\t\t\tType \"java -cp bin/RankLib.jar ciir.umass.edu.feature.FeatureManager\" for help with shuffling.");

            logger.info(() -> "\t[ -kcvmd <dir> ]\tDirectory for models trained via cross-validation (default=not-save)");
            logger.info(
                    () -> "\t[ -kcvmn <model> ]\tName for model learned in each fold. It will be prefix-ed with the fold-number (default=empty)");

            logger.info(() -> "    [-] RankNet-specific parameters");
            logger.info(() -> "\t[ -epoch <T> ]\t\tThe number of epochs to train (default=" + RankNet.nIteration + ")");
            logger.info(() -> "\t[ -layer <layer> ]\tThe number of hidden layers (default=" + RankNet.nHiddenLayer + ")");
            logger.info(() -> "\t[ -node <node> ]\tThe number of hidden nodes per layer (default=" + RankNet.nHiddenNodePerLayer + ")");
            logger.info(() -> "\t[ -lr <rate> ]\t\tLearning rate (default="
                    + (new DecimalFormat("###.########")).format(RankNet.learningRate) + ")");

            logger.info(() -> "    [-] RankBoost-specific parameters");
            logger.info(() -> "\t[ -round <T> ]\t\tThe number of rounds to train (default=" + RankBoost.nIteration + ")");
            logger.info(() -> "\t[ -tc <k> ]\t\tNumber of threshold candidates to search. -1 to use all feature values (default="
                    + RankBoost.nThreshold + ")");

            logger.info(() -> "    [-] AdaRank-specific parameters");
            logger.info(() -> "\t[ -round <T> ]\t\tThe number of rounds to train (default=" + AdaRank.nIteration + ")");
            logger.info(() -> "\t[ -noeq ]\t\tTrain without enqueuing too-strong features (default=unspecified)");
            logger.info(
                    () -> "\t[ -tolerance <t> ]\tTolerance between two consecutive rounds of learning (default=" + AdaRank.tolerance + ")");
            logger.info(() -> "\t[ -max <times> ]\tThe maximum number of times a feature can be consecutively selected "
                    + "without changing performance (default=" + AdaRank.maxSelCount + ")");

            logger.info(() -> "    [-] Coordinate Ascent-specific parameters");
            logger.info(() -> "\t[ -r <k> ]\t\tThe number of random restarts (default=" + CoorAscent.nRestart + ")");
            logger.info(() -> "\t[ -i <iteration> ]\tThe number of iterations to search in each dimension (default="
                    + CoorAscent.nMaxIteration + ")");
            logger.info(() -> "\t[ -tolerance <t> ]\tPerformance tolerance between two solutions (default=" + CoorAscent.tolerance + ")");
            logger.info(() -> "\t[ -reg <slack> ]\tRegularization parameter (default=no-regularization)");

            logger.info(() -> "    [-] {MART, LambdaMART}-specific parameters");
            logger.info(() -> "\t[ -tree <t> ]\t\tNumber of trees (default=" + LambdaMART.nTrees + ")");
            logger.info(() -> "\t[ -leaf <l> ]\t\tNumber of leaves for each tree (default=" + LambdaMART.nTreeLeaves + ")");
            logger.info(() -> "\t[ -shrinkage <factor> ]\tShrinkage, or learning rate (default=" + LambdaMART.learningRate + ")");
            logger.info(() -> "\t[ -tc <k> ]\t\tNumber of threshold candidates for tree spliting. -1 to use all feature values (default="
                    + LambdaMART.nThreshold + ")");
            logger.info(() -> "\t[ -mls <n> ]\t\tMin leaf support -- minimum % of docs each leaf has to contain (default="
                    + LambdaMART.minLeafSupport + ")");
            logger.info(
                    () -> "\t[ -estop <e> ]\t\tStop early when no improvement is observed on validaton data in e consecutive rounds (default="
                            + LambdaMART.nRoundToStopEarly + ")");

            logger.info(() -> "    [-] ListNet-specific parameters");
            logger.info(() -> "\t[ -epoch <T> ]\t\tThe number of epochs to train (default=" + ListNet.nIteration + ")");
            logger.info(() -> "\t[ -lr <rate> ]\t\tLearning rate (default="
                    + (new DecimalFormat("###.########")).format(ListNet.learningRate) + ")");

            logger.info(() -> "    [-] Random Forests-specific parameters");
            logger.info(() -> "\t[ -bag <r> ]\t\tNumber of bags (default=" + RFRanker.nBag + ")");
            logger.info(() -> "\t[ -srate <r> ]\t\tSub-sampling rate (default=" + RFRanker.subSamplingRate + ")");
            logger.info(() -> "\t[ -frate <r> ]\t\tFeature sampling rate (default=" + RFRanker.featureSamplingRate + ")");
            final int type = (RFRanker.rType.ordinal() - RankerType.MART.ordinal());
            logger.info(() -> "\t[ -rtype <type> ]\tRanker to bag (default=" + type + ", i.e. " + rType[type] + ")");
            logger.info(() -> "\t[ -tree <t> ]\t\tNumber of trees in each bag (default=" + RFRanker.nTrees + ")");
            logger.info(() -> "\t[ -leaf <l> ]\t\tNumber of leaves for each tree (default=" + RFRanker.nTreeLeaves + ")");
            logger.info(() -> "\t[ -shrinkage <factor> ]\tShrinkage, or learning rate (default=" + RFRanker.learningRate + ")");
            logger.info(() -> "\t[ -tc <k> ]\t\tNumber of threshold candidates for tree spliting. -1 to use all feature values (default="
                    + RFRanker.nThreshold + ")");
            logger.info(() -> "\t[ -mls <n> ]\t\tMin leaf support -- minimum % of docs each leaf has to contain (default="
                    + RFRanker.minLeafSupport + ")");

            logger.info(() -> "    [-] Linear Regression-specific parameters");
            logger.info(() -> "\t[ -L2 <reg> ]\t\tL2 regularization parameter (default=" + LinearRegRank.lambda + ")");

            logger.info(() -> "  [+] Testing previously saved models");
            logger.info(() -> "\t-load <model>\t\tThe model to load");
            logger.info(() -> "\t\t\t\tMultiple -load can be used to specify models from multiple folds (in increasing order),");
            logger.info(() -> "\t\t\t\t  in which case the test/rank data will be partitioned accordingly.");
            logger.info(() -> "\t-test <file>\t\tTest data to evaluate the model(s) (specify either this or -rank but not both)");
            logger.info(() -> "\t-rank <file>\t\tRank the samples in the specified file (specify either this or -test but not both)");
            logger.info(() -> "\t[ -metric2T <metric> ]\tMetric to evaluate on the test data (default=ERR@10)");
            logger.info(() -> "\t[ -gmax <label> ]\tHighest judged relevance label. It affects the calculation of ERR (default="
                    + (int) SimpleMath.logBase2(ERRScorer.MAX) + ", i.e. 5-point scale {0,1,2,3,4})");
            logger.info(() -> "\t[ -score <file>]\tStore ranker's score for each object being ranked (has to be used with -rank)");
            logger.info(() -> "\t[ -qrel <file> ]\tTREC-style relevance judgment file. It only affects MAP and NDCG (default=unspecified)");
            logger.info(
                    () -> "\t[ -idv <file> ]\t\tSave model performance (in test metric) on individual ranked lists (has to be used with -test)");
            logger.info(() -> "\t[ -norm ]\t\tNormalize feature vectors (similar to -norm for training/tuning)");
            //logger.info(()->"\t[ -sparse ]\t\tUse sparse representation for all feature vectors (default=dense)");

            return;
        }

        for (int i = 0; i < args.length; i++) {
            if (args[i].equalsIgnoreCase("-train")) {
                trainFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-ranker")) {
                rankerType = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-feature")) {
                featureDescriptionFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-metric2t")) {
                trainMetric = args[++i];
            } else if (args[i].equalsIgnoreCase("-metric2T")) {
                testMetric = args[++i];
            } else if (args[i].equalsIgnoreCase("-gmax")) {
                ERRScorer.MAX = Math.pow(2, Double.parseDouble(args[++i]));
            } else if (args[i].equalsIgnoreCase("-qrel")) {
                qrelFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-tts")) {
                ttSplit = Float.parseFloat(args[++i]);
            } else if (args[i].equalsIgnoreCase("-tvs")) {
                tvSplit = Float.parseFloat(args[++i]);
            } else if (args[i].equalsIgnoreCase("-kcv")) {
                foldCV = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-validate")) {
                validationFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-test")) {
                testFile = args[++i];
                testFiles.add(testFile);
            } else if (args[i].equalsIgnoreCase("-norm")) {
                Evaluator.normalize = true;
                final String n = args[++i];
                if (n.equalsIgnoreCase("sum")) {
                    Evaluator.nml = new SumNormalizor();
                } else if (n.equalsIgnoreCase("zscore")) {
                    Evaluator.nml = new ZScoreNormalizor();
                } else if (n.equalsIgnoreCase("linear")) {
                    Evaluator.nml = new LinearNormalizer();
                } else {
                    throw RankLibError.create("Unknown normalizor: " + n);
                }
            } else if (args[i].equalsIgnoreCase("-sparse")) {
                useSparseRepresentation = true;
            } else if (args[i].equalsIgnoreCase("-save")) {
                Evaluator.modelFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-kcvmd")) {
                kcvModelDir = args[++i];
            } else if (args[i].equalsIgnoreCase("-kcvmn")) {
                kcvModelFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-missingZero")) {
                DataPoint.missingZero = true;
            } else if (args[i].equalsIgnoreCase("-load")) {
                savedModelFile = args[++i];
                savedModelFiles.add(args[i]);
            } else if (args[i].equalsIgnoreCase("-idv")) {
                prpFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-rank")) {
                rankFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-score")) {
                scoreFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-epoch")) {
                RankNet.nIteration = Integer.parseInt(args[++i]);
                ListNet.nIteration = Integer.parseInt(args[i]);
            } else if (args[i].equalsIgnoreCase("-layer")) {
                RankNet.nHiddenLayer = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-node")) {
                RankNet.nHiddenNodePerLayer = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-lr")) {
                RankNet.learningRate = Double.parseDouble(args[++i]);
                ListNet.learningRate = Neuron.learningRate;
            }

            //RankBoost
            else if (args[i].equalsIgnoreCase("-tc")) {
                RankBoost.nThreshold = Integer.parseInt(args[++i]);
                LambdaMART.nThreshold = Integer.parseInt(args[i]);
            }

            //AdaRank
            else if (args[i].equalsIgnoreCase("-noeq")) {
                AdaRank.trainWithEnqueue = false;
            } else if (args[i].equalsIgnoreCase("-max")) {
                AdaRank.maxSelCount = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-r")) {
                CoorAscent.nRestart = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-i")) {
                CoorAscent.nMaxIteration = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-round")) {
                RankBoost.nIteration = Integer.parseInt(args[++i]);
                AdaRank.nIteration = Integer.parseInt(args[i]);
            } else if (args[i].equalsIgnoreCase("-reg")) {
                CoorAscent.slack = Double.parseDouble(args[++i]);
                CoorAscent.regularized = true;
            } else if (args[i].equalsIgnoreCase("-tolerance")) {
                AdaRank.tolerance = Double.parseDouble(args[++i]);
                CoorAscent.tolerance = Double.parseDouble(args[i]);
            }

            //MART / LambdaMART / Random forest
            else if (args[i].equalsIgnoreCase("-tree")) {
                LambdaMART.nTrees = Integer.parseInt(args[++i]);
                RFRanker.nTrees = Integer.parseInt(args[i]);
            } else if (args[i].equalsIgnoreCase("-leaf")) {
                LambdaMART.nTreeLeaves = Integer.parseInt(args[++i]);
                RFRanker.nTreeLeaves = Integer.parseInt(args[i]);
            } else if (args[i].equalsIgnoreCase("-shrinkage")) {
                LambdaMART.learningRate = Float.parseFloat(args[++i]);
                RFRanker.learningRate = Float.parseFloat(args[i]);
            } else if (args[i].equalsIgnoreCase("-mls")) {
                LambdaMART.minLeafSupport = Integer.parseInt(args[++i]);
                RFRanker.minLeafSupport = LambdaMART.minLeafSupport;
            } else if (args[i].equalsIgnoreCase("-estop")) {
                LambdaMART.nRoundToStopEarly = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-bag")) {
                RFRanker.nBag = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-srate")) {
                RFRanker.subSamplingRate = Float.parseFloat(args[++i]);
            } else if (args[i].equalsIgnoreCase("-frate")) {
                RFRanker.featureSamplingRate = Float.parseFloat(args[++i]);
            } else if (args[i].equalsIgnoreCase("-rtype")) {
                final int rt = Integer.parseInt(args[++i]);
                if (rt == 0 || rt == 6) {
                    RFRanker.rType = rType2[rt];
                } else {
                    throw RankLibError.create(rType[rt] + " cannot be bagged. Random Forests only supports MART/LambdaMART.");
                }
            }

            else if (args[i].equalsIgnoreCase("-L2")) {
                LinearRegRank.lambda = Double.parseDouble(args[++i]);
            } else if (args[i].equalsIgnoreCase("-thread")) {
                nThread = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-nf")) {
                newFeatureFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-keep")) {
                keepOrigFeatures = true;
            } else if (args[i].equalsIgnoreCase("-t")) {
                topNew = Integer.parseInt(args[++i]);
            } else if (args[i].equalsIgnoreCase("-indri")) {
                indriRankingFile = args[++i];
            } else if (args[i].equalsIgnoreCase("-hr")) {
                mustHaveRelDoc = true;
            } else {
                throw RankLibError.create("Unknown command-line parameter: " + args[i]);
            }
        }

        if (nThread == -1) {
            nThread = Runtime.getRuntime().availableProcessors();
        }
        MyThreadPool.init(nThread);

        if (testMetric.compareTo("") == 0) {
            testMetric = trainMetric;
        }

        logger.info(() -> (keepOrigFeatures) ? "Keep orig. features" : "Discard orig. features");
        final Evaluator e = new Evaluator(rType2[rankerType], trainMetric, testMetric);

        if (trainFile.compareTo("") != 0) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Training data:\t" + trainFile);
            }

            //print out parameter settings
            if (foldCV != -1) {
                if (logger.isLoggable(Level.INFO)) {
                    logger.info("Cross validation: " + foldCV + " folds.");
                }
                if (tvSplit > 0) {
                    if (logger.isLoggable(Level.INFO)) {
                        logger.info("Train-Validation split: " + tvSplit);
                    }
                }
            } else {
                if (testFile.compareTo("") != 0) {
                    if (logger.isLoggable(Level.INFO)) {
                        logger.info("Test data:\t" + testFile);
                    }
                } else if (ttSplit > 0) {
                    if (logger.isLoggable(Level.INFO)) {
                        logger.info("Train-Test split: " + ttSplit);
                    }
                }

                if (validationFile.compareTo("") != 0) {
                    if (logger.isLoggable(Level.INFO)) {
                        logger.info("Validation data:\t" + validationFile);
                    }
                } else if (ttSplit <= 0 && tvSplit > 0) {
                    if (logger.isLoggable(Level.INFO)) {
                        logger.info("Train-Validation split: " + tvSplit);
                    }
                }
            }
            logger.info(() -> "Feature vector representation: " + ((useSparseRepresentation) ? "Sparse" : "Dense") + ".");
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Ranking method:\t" + rType[rankerType]);
            }
            if (featureDescriptionFile.compareTo("") != 0) {
                if (logger.isLoggable(Level.INFO)) {
                    logger.info("Feature description file:\t" + featureDescriptionFile);
                }
            } else {
                logger.info(() -> "Feature description file:\tUnspecified. All features will be used.");
            }
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Train metric:\t" + trainMetric);
                logger.info("Test metric:\t" + testMetric);
            }

            if (trainMetric.toUpperCase().startsWith("ERR") || testMetric.toUpperCase().startsWith("ERR")) {
                logger.info(() -> "Highest relevance label (to compute ERR): " + (int) SimpleMath.logBase2(ERRScorer.MAX));
            }
            if (qrelFile.compareTo("") != 0) {
                logger.info(() -> "TREC-format relevance judgment (only affects MAP and NDCG scores): " + qrelFile);
            }
            logger.info(() -> "Feature normalization: " + ((Evaluator.normalize) ? Evaluator.nml.name() : "No"));

            if (kcvModelDir.compareTo("") != 0) {
                if (logger.isLoggable(Level.INFO)) {
                    logger.info("Models directory: " + kcvModelDir);
                }
            }

            if (kcvModelFile.compareTo("") != 0) {
                if (logger.isLoggable(Level.INFO)) {
                    logger.info("Models' name: " + kcvModelFile);
                }
            }

            if (modelFile.compareTo("") != 0) {
                logger.info(() -> "Model file: " + modelFile);
                //logger.info(()->"#threads:\t" + nThread);
            }

            if (logger.isLoggable(Level.INFO)) {
                logger.info("[+] " + rType[rankerType] + "'s Parameters:");
            }
            final RankerFactory rf = new RankerFactory();

            rf.createRanker(rType2[rankerType]).printParameters();

            //starting to do some work
            if (foldCV != -1) {
                //if(kcvModelDir.compareTo("") != 0 && kcvModelFile.compareTo("") == 0)
                //	kcvModelFile = "default";
                //
                //- Behavioral changes: Write kcv models if kcvmd OR kcvmn defined.  Use
                //  default names for missing arguments: "kcvmodels" default directory
                //  and "kcv" default model name.
                if (kcvModelDir.compareTo("") != 0 && kcvModelFile.compareTo("") == 0) {
                    kcvModelFile = "kcv";
                } else if (kcvModelDir.compareTo("") == 0 && kcvModelFile.compareTo("") != 0) {
                    kcvModelDir = "kcvmodels";
                }

                //- models won't be saved if kcvModelDir=""   [OBSOLETE]
                //- Models saved if EITHER kcvmd OR kcvmn defined.  Use default names for missing values.
                e.evaluate(trainFile, featureDescriptionFile, foldCV, tvSplit, kcvModelDir, kcvModelFile);
            } else {
                if (ttSplit > 0.0) {
                    e.evaluate(trainFile, validationFile, featureDescriptionFile, ttSplit);//no validation will be done if validationFile=""
                } else if (tvSplit > 0.0) {
                    e.evaluate(trainFile, tvSplit, testFile, featureDescriptionFile);
                } else {
                    e.evaluate(trainFile, validationFile, testFile, featureDescriptionFile);//All files except for trainFile can be empty. This will be handled appropriately
                }
            }
        } else //scenario: test a saved model
        {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Model file:\t" + savedModelFile);
            }
            logger.info(() -> "Feature normalization: " + ((Evaluator.normalize) ? Evaluator.nml.name() : "No"));
            if (rankFile.compareTo("") != 0) {
                if (scoreFile.compareTo("") != 0) {
                    if (savedModelFiles.size() > 1) {
                        e.score(savedModelFiles, rankFile, scoreFile);
                    } else {
                        e.score(savedModelFile, rankFile, scoreFile);
                    }
                } else if (indriRankingFile.compareTo("") != 0) {
                    if (savedModelFiles.size() > 1) {
                        e.rank(savedModelFiles, rankFile, indriRankingFile);
                    } else if (savedModelFiles.size() == 1) {
                        e.rank(savedModelFile, rankFile, indriRankingFile);
                    } else {
                        //This is *ONLY* for debugging purposes. It is *NOT* exposed via cmd-line
                        //It will evaluate the input ranking (without being re-ranked by any model) using any measure specified via metric2T
                        e.rank(rankFile, indriRankingFile);
                    }
                } else {
                    throw RankLibError
                            .create("This function has been removed.\n" + "Consider using -score in addition to your current parameters, "
                                    + "and do the ranking yourself based on these scores.");
                    //e.rank(savedModelFile, rankFile);
                }
            } else {
                if (logger.isLoggable(Level.INFO)) {
                    logger.info("Test metric:\t" + testMetric);
                }
                if (testMetric.startsWith("ERR")) {
                    logger.info(() -> "Highest relevance label (to compute ERR): " + (int) SimpleMath.logBase2(ERRScorer.MAX));
                }

                if (savedModelFile.compareTo("") != 0) {
                    if (savedModelFiles.size() > 1)//models trained via cross-validation
                    {
                        if (testFiles.size() > 1) {
                            e.test(savedModelFiles, testFiles, prpFile);
                        } else {
                            e.test(savedModelFiles, testFile, prpFile);
                        }
                    } else if (savedModelFiles.size() == 1) {
                        e.test(savedModelFile, testFile, prpFile);
                    }
                } else if (scoreFile.compareTo("") != 0) {
                    e.testWithScoreFile(testFile, scoreFile);
                    //It will evaluate the input ranking (without being re-ranked by any model) using any measure specified via metric2T
                } else {
                    e.test(testFile, prpFile);
                }
            }
        }
        MyThreadPool.getInstance().shutdown();
    }

    //main settings
    public static boolean mustHaveRelDoc = false;
    public static boolean useSparseRepresentation = false;
    public static boolean normalize = false;
    public static Normalizer nml = new SumNormalizor();
    public static String modelFile = "";

    public static String qrelFile = "";//measure such as NDCG and MAP requires "complete" judgment.
    //The relevance labels attached to our samples might be only a subset of the entire relevance judgment set.
    //If we're working on datasets like Letor/Web10K or Yahoo! LTR, we can totally ignore this parameter.
    //However, if we sample top-K documents from baseline run (e.g. query-likelihood) to create training data for TREC collections,
    //there's a high chance some relevant document (the in qrel file TREC provides) does not appear in our top-K list -- thus the calculation of
    //MAP and NDCG is no longer precise. If so, specify that "external" relevance judgment here (via the -qrel cmd parameter)

    //tmp settings, for personal use
    public static String newFeatureFile = "";
    public static boolean keepOrigFeatures = false;
    public static int topNew = 2000;

    protected RankerFactory rFact = new RankerFactory();
    protected MetricScorerFactory mFact = new MetricScorerFactory();

    protected MetricScorer trainScorer = null;
    protected MetricScorer testScorer = null;
    protected RankerType type = RankerType.MART;

    public Evaluator(final RankerType rType, final METRIC trainMetric, final METRIC testMetric) {
        this.type = rType;
        trainScorer = mFact.createScorer(trainMetric);
        testScorer = mFact.createScorer(testMetric);
        if (qrelFile.compareTo("") != 0) {
            trainScorer.loadExternalRelevanceJudgment(qrelFile);
            testScorer.loadExternalRelevanceJudgment(qrelFile);
        }
    }

    public Evaluator(final RankerType rType, final METRIC trainMetric, final int trainK, final METRIC testMetric, final int testK) {
        this.type = rType;
        trainScorer = mFact.createScorer(trainMetric, trainK);
        testScorer = mFact.createScorer(testMetric, testK);
        if (qrelFile.compareTo("") != 0) {
            trainScorer.loadExternalRelevanceJudgment(qrelFile);
            testScorer.loadExternalRelevanceJudgment(qrelFile);
        }
    }

    public Evaluator(final RankerType rType, final METRIC trainMetric, final METRIC testMetric, final int k) {
        this.type = rType;
        trainScorer = mFact.createScorer(trainMetric, k);
        testScorer = mFact.createScorer(testMetric, k);
        if (qrelFile.compareTo("") != 0) {
            trainScorer.loadExternalRelevanceJudgment(qrelFile);
            testScorer.loadExternalRelevanceJudgment(qrelFile);
        }
    }

    public Evaluator(final RankerType rType, final METRIC metric, final int k) {
        this.type = rType;
        trainScorer = mFact.createScorer(metric, k);
        if (qrelFile.compareTo("") != 0) {
            trainScorer.loadExternalRelevanceJudgment(qrelFile);
        }
        testScorer = trainScorer;
    }

    public Evaluator(final RankerType rType, final String trainMetric, final String testMetric) {
        this.type = rType;
        trainScorer = mFact.createScorer(trainMetric);
        testScorer = mFact.createScorer(testMetric);
        if (qrelFile.compareTo("") != 0) {
            trainScorer.loadExternalRelevanceJudgment(qrelFile);
            testScorer.loadExternalRelevanceJudgment(qrelFile);
        }
    }

    public List<RankList> readInput(final String inputFile) {
        return FeatureManager.readInput(inputFile, mustHaveRelDoc, useSparseRepresentation);
    }

    public void normalize(final List<RankList> samples) {
        for (final RankList sample : samples) {
            nml.normalize(sample);
        }
    }

    public void normalize(final List<RankList> samples, final int[] fids) {
        for (final RankList sample : samples) {
            nml.normalize(sample, fids);
        }
    }

    public void normalizeAll(final List<List<RankList>> samples, final int[] fids) {
        for (final List<RankList> sample : samples) {
            normalize(sample, fids);
        }
    }

    public int[] readFeature(final String featureDefFile) {
        //if(featureDefFile.compareTo("") == 0)
        if (featureDefFile.isEmpty()) {
            return null;
        }
        return FeatureManager.readFeature(featureDefFile);
    }

    public double evaluate(final Ranker ranker, final List<RankList> rl) {
        List<RankList> l = rl;
        if (ranker != null) {
            l = ranker.rank(rl);
        }
        return testScorer.score(l);
    }

    /**
     * Evaluate the currently selected ranking algorithm using &lt;training data, validation data, testing data and the defined features&gt;.
     * @param trainFile
     * @param validationFile
     * @param testFile
     * @param featureDefFile
     */
    public void evaluate(final String trainFile, final String validationFile, final String testFile, final String featureDefFile) {
        final List<RankList> train = readInput(trainFile);//read input

        List<RankList> validation = null;
        //if(validationFile.compareTo("")!=0)
        if (!validationFile.isEmpty()) {
            validation = readInput(validationFile);
        }

        List<RankList> test = null;
        //if(testFile.compareTo("")!=0)
        if (!testFile.isEmpty()) {
            test = readInput(testFile);
        }

        int[] features = readFeature(featureDefFile);//read features
        if (features == null) {
            features = FeatureManager.getFeatureFromSampleVector(train);
        }

        if (normalize) {
            normalize(train, features);
            if (validation != null) {
                normalize(validation, features);
            }
            if (test != null) {
                normalize(test, features);
            }
        }

        final RankerTrainer trainer = new RankerTrainer();
        final Ranker ranker = trainer.train(type, train, validation, features, trainScorer);

        if (test != null) {
            final double rankScore = evaluate(ranker, test);
            logger.info(() -> testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        }
        if (modelFile.compareTo("") != 0) {
            ranker.save(modelFile);
            logger.info(() -> "Model saved to: " + modelFile);
        }
    }

    /**
     * Evaluate the currently selected ranking algorithm using percenTrain% of the samples for training the rest for testing.
     * @param sampleFile
     * @param validationFile Empty string for "no validation data"
     * @param featureDefFile
     * @param percentTrain
     */
    public void evaluate(final String sampleFile, final String validationFile, final String featureDefFile, final double percentTrain) {
        final List<RankList> trainingData = new ArrayList<>();
        final List<RankList> testData = new ArrayList<>();
        final int[] features = prepareSplit(sampleFile, featureDefFile, percentTrain, normalize, trainingData, testData);
        List<RankList> validation = null;

        //if(validationFile.compareTo("") != 0)
        if (!validationFile.isEmpty()) {
            validation = readInput(validationFile);
            if (normalize) {
                normalize(validation, features);
            }
        }

        final RankerTrainer trainer = new RankerTrainer();
        final Ranker ranker = trainer.train(type, trainingData, validation, features, trainScorer);

        final double rankScore = evaluate(ranker, testData);

        logger.info(() -> testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        if (modelFile.compareTo("") != 0) {
            ranker.save(modelFile);
            logger.info(() -> "Model saved to: " + modelFile);
        }
    }

    /**
     * Evaluate the currently selected ranking algorithm using percenTrain% of the training samples for training the rest as validation data.
     * Test data is specified separately.
     * @param trainFile
     * @param percentTrain
     * @param testFile Empty string for "no test data"
     * @param featureDefFile
     */
    public void evaluate(final String trainFile, final double percentTrain, final String testFile, final String featureDefFile) {
        final List<RankList> train = new ArrayList<>();
        final List<RankList> validation = new ArrayList<>();
        final int[] features = prepareSplit(trainFile, featureDefFile, percentTrain, normalize, train, validation);
        List<RankList> test = null;

        //if(testFile.compareTo("") != 0)
        if (!testFile.isEmpty()) {
            test = readInput(testFile);
            if (normalize) {
                normalize(test, features);
            }
        }

        final RankerTrainer trainer = new RankerTrainer();
        final Ranker ranker = trainer.train(type, train, validation, features, trainScorer);

        if (test != null) {
            final double rankScore = evaluate(ranker, test);
            logger.info(() -> testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        }
        if (modelFile.compareTo("") != 0) {
            ranker.save(modelFile);
            logger.info(() -> "Model saved to: " + modelFile);
        }
    }

    /**
     * Evaluate the currently selected ranking algorithm using &lt;data, defined features&gt; with k-fold cross validation.
     * @param sampleFile
     * @param featureDefFile
     * @param nFold
     * @param modelDir
     * @param modelFile
     */
    public void evaluate(final String sampleFile, final String featureDefFile, final int nFold, final String modelDir,
            final String modelFile) {
        evaluate(sampleFile, featureDefFile, nFold, -1, modelDir, modelFile);
    }

    /**
     * Evaluate the currently selected ranking algorithm using &lt;data, defined features&gt; with k-fold cross validation.
     * @param sampleFile
     * @param featureDefFile
     * @param nFold
     * @param tvs Train-validation split ratio.
     * @param modelDir
     * @param modelFile
     */
    public void evaluate(final String sampleFile, final String featureDefFile, final int nFold, final float tvs, final String modelDir,
            final String modelFile) {
        final List<List<RankList>> trainingData = new ArrayList<>();
        final List<List<RankList>> validationData = new ArrayList<>();
        final List<List<RankList>> testData = new ArrayList<>();

        //read all samples
        //List<RankList> samples = FeatureManager.readInput(sampleFile);
        final List<RankList> samples = readInput(sampleFile);

        //get features
        int[] features = readFeature(featureDefFile);//read features
        if (features == null) {
            features = FeatureManager.getFeatureFromSampleVector(samples);
        }

        FeatureManager.prepareCV(samples, nFold, tvs, trainingData, validationData, testData);

        //normalization
        if (normalize) {
            for (int i = 0; i < nFold; i++) {
                normalizeAll(trainingData, features);
                normalizeAll(validationData, features);
                normalizeAll(testData, features);
            }
        }

        Ranker ranker = null;
        double scoreOnTrain = 0.0;
        double scoreOnTest = 0.0;
        double totalScoreOnTest = 0.0;
        int totalTestSampleSize = 0;

        final double[][] scores = new double[nFold][];
        for (int i = 0; i < nFold; i++) {
            scores[i] = new double[] { 0.0, 0.0 };
        }
        for (int i = 0; i < nFold; i++) {
            final List<RankList> train = trainingData.get(i);
            List<RankList> vali = null;
            if (tvs > 0) {
                vali = validationData.get(i);
            }
            final List<RankList> test = testData.get(i);

            final RankerTrainer trainer = new RankerTrainer();
            ranker = trainer.train(type, train, vali, features, trainScorer);

            final double s2 = evaluate(ranker, test);
            scoreOnTrain += ranker.getScoreOnTrainingData();
            scoreOnTest += s2;
            totalScoreOnTest += s2 * test.size();
            totalTestSampleSize += test.size();

            //save performance in each fold
            scores[i][0] = ranker.getScoreOnTrainingData();
            scores[i][1] = s2;

            //if(modelDir.compareTo("") != 0)
            if (!modelDir.isEmpty()) {
                ranker.save(FileUtils.makePathStandard(modelDir) + "f" + (i + 1) + "." + modelFile);
                if (logger.isLoggable(Level.INFO)) {
                    logger.info("Fold-" + (i + 1) + " model saved to: " + modelFile);
                }
            }
        }
        if (logger.isLoggable(Level.INFO)) {
            logger.info(() -> "Summary:");
            logger.info(() -> testScorer.name() + "\t|   Train\t| Test");
            for (int i = 0; i < nFold; i++) {
                logger.info("Fold " + (i + 1) + "\t|   " + SimpleMath.round(scores[i][0], 4) + "\t|  " + SimpleMath.round(scores[i][1], 4)
                        + "\t");
            }
            logger.info(
                    "Avg.\t|   " + SimpleMath.round(scoreOnTrain / nFold, 4) + "\t|  " + SimpleMath.round(scoreOnTest / nFold, 4) + "\t");
            logger.info("Total\t|   " + "\t" + "\t|  " + SimpleMath.round(totalScoreOnTest / totalTestSampleSize, 4) + "\t");
        }
    }

    /**
     * Evaluate the performance (in -metric2T) of the input rankings
     * @param testFile Input rankings
     */
    public void test(final String testFile) {
        final List<RankList> test = readInput(testFile);
        final double rankScore = evaluate(null, test);
        logger.info(() -> testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
    }

    public void test(final String testFile, final String prpFile) {
        final List<RankList> test = readInput(testFile);
        double rankScore = 0.0;
        final List<String> ids = new ArrayList<>();
        final List<Double> scores = new ArrayList<>();
        for (final RankList l : test) {
            final double score = testScorer.score(l);
            ids.add(l.getID());
            scores.add(score);
            rankScore += score;
        }
        rankScore /= test.size();
        ids.add("all");
        scores.add(rankScore);
        if (logger.isLoggable(Level.INFO)) {
            logger.info(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        }

        //if(prpFile.compareTo("") != 0)
        if (!prpFile.isEmpty()) {
            savePerRankListPerformanceFile(ids, scores, prpFile);
            logger.info(() -> "Per-ranked list performance saved to: " + prpFile);
        }
    }

    /**
     * Evaluate the performance (in -metric2T) of a pre-trained model. Save its performance on each of the ranked list if this is specified.
     * @param modelFile Pre-trained model
     * @param testFile Test data
     * @param prpFile Per-ranked list performance file: Model's performance on each of the ranked list. These won't be saved if prpFile="".
     */
    public void test(final String modelFile, final String testFile, final String prpFile) {
        final Ranker ranker = rFact.loadRankerFromFile(modelFile);
        final int[] features = ranker.getFeatures();
        final List<RankList> test = readInput(testFile);
        if (normalize) {
            normalize(test, features);
        }

        double rankScore = 0.0;
        final List<String> ids = new ArrayList<>();
        final List<Double> scores = new ArrayList<>();
        for (final RankList aTest : test) {
            final RankList l = ranker.rank(aTest);
            final double score = testScorer.score(l);
            ids.add(l.getID());
            scores.add(score);
            rankScore += score;
        }
        rankScore /= test.size();
        ids.add("all");
        scores.add(rankScore);
        if (logger.isLoggable(Level.INFO)) {
            logger.info(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        }

        if (!prpFile.isEmpty()) {
            savePerRankListPerformanceFile(ids, scores, prpFile);
            logger.info(() -> "Per-ranked list performance saved to: " + prpFile);
        }
    }

    /**
     * Evaluate the performance (in -metric2T) of k pre-trained models. Data in the test file will be splitted into k fold, where k=|models|.
     * Each model will be evaluated on the data from the corresponding fold.
     * @param modelFiles Pre-trained models
     * @param testFile Test data
     * @param prpFile Per-ranked list performance file: Model's performance on each of the ranked list. These won't be saved if prpFile="".
     */
    public void test(final List<String> modelFiles, final String testFile, final String prpFile) {
        final List<List<RankList>> trainingData = new ArrayList<>();
        final List<List<RankList>> testData = new ArrayList<>();

        //read all samples
        final int nFold = modelFiles.size();
        //List<RankList> samples = FeatureManager.readInput(testFile);
        final List<RankList> samples = readInput(testFile);

        logger.info(() -> "Preparing " + nFold + "-fold test data... ");
        FeatureManager.prepareCV(samples, nFold, trainingData, testData);
        double rankScore = 0.0;
        final List<String> ids = new ArrayList<>();
        final List<Double> scores = new ArrayList<>();
        for (int f = 0; f < nFold; f++) {
            final List<RankList> test = testData.get(f);
            final Ranker ranker = rFact.loadRankerFromFile(modelFiles.get(f));
            final int[] features = ranker.getFeatures();
            if (normalize) {
                normalize(test, features);
            }

            for (final RankList aTest : test) {
                final RankList l = ranker.rank(aTest);
                final double score = testScorer.score(l);
                ids.add(l.getID());
                scores.add(score);
                rankScore += score;
            }
        }
        rankScore = rankScore / ids.size();
        ids.add("all");
        scores.add(rankScore);
        if (logger.isLoggable(Level.INFO)) {
            logger.info(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        }
        if (!prpFile.isEmpty()) {
            savePerRankListPerformanceFile(ids, scores, prpFile);
            logger.info(() -> "Per-ranked list performance saved to: " + prpFile);
        }
    }

    /**
     * Similar to the above, except data has already been splitted. The k-th model will be applied on the k-th test file.
     * @param modelFiles
     * @param testFiles
     * @param prpFile
     */
    public void test(final List<String> modelFiles, final List<String> testFiles, final String prpFile) {
        final int nFold = modelFiles.size();
        double rankScore = 0.0;
        final List<String> ids = new ArrayList<>();
        final List<Double> scores = new ArrayList<>();
        for (int f = 0; f < nFold; f++) {
            final List<RankList> test = readInput(testFiles.get(f));
            final Ranker ranker = rFact.loadRankerFromFile(modelFiles.get(f));
            final int[] features = ranker.getFeatures();

            if (normalize) {
                normalize(test, features);
            }

            for (final RankList aTest : test) {
                final RankList l = ranker.rank(aTest);
                final double score = testScorer.score(l);
                ids.add(l.getID());
                scores.add(score);
                rankScore += score;
            }
        }
        rankScore = rankScore / ids.size();
        ids.add("all");
        scores.add(rankScore);
        if (logger.isLoggable(Level.INFO)) {
            logger.info(testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        }

        if (!prpFile.isEmpty()) {
            savePerRankListPerformanceFile(ids, scores, prpFile);
            logger.info(() -> "Per-ranked list performance saved to: " + prpFile);
        }
    }

    /**
     * Re-order the input rankings and measure their effectiveness (in -metric2T)
     * @param testFile Input rankings
     * @param scoreFile The model score file on each of the documents
     */
    public void testWithScoreFile(final String testFile, final String scoreFile) {
        try (BufferedReader in = FileUtils.smartReader(scoreFile)) {
            final List<RankList> test = readInput(testFile);
            String content = "";
            final List<Double> scores = new ArrayList<>();
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.compareTo("") == 0) {
                    continue;
                }
                scores.add(Double.parseDouble(content));
            }
            int k = 0;
            for (int i = 0; i < test.size(); i++) {
                RankList rl = test.get(i);
                final double[] s = new double[rl.size()];
                for (int j = 0; j < rl.size(); j++) {
                    s[j] = scores.get(k++);
                }
                rl = new RankList(rl, MergeSorter.sort(s, false));
                test.set(i, rl);
            }

            final double rankScore = evaluate(null, test);
            logger.info(() -> testScorer.name() + " on test data: " + SimpleMath.round(rankScore, 4));
        } catch (final IOException e) {
            throw RankLibError.create(e);
        }
    }

    /**
     * Write the model's score for each of the documents in a test rankings.
     * @param modelFile Pre-trained model
     * @param testFile Test data
     * @param outputFile Output file
     */
    public void score(final String modelFile, final String testFile, final String outputFile) {
        final Ranker ranker = rFact.loadRankerFromFile(modelFile);
        final int[] features = ranker.getFeatures();
        final List<RankList> test = readInput(testFile);
        if (normalize) {
            normalize(test, features);
        }

        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"))) {
            for (final RankList l : test) {
                for (int j = 0; j < l.size(); j++) {
                    out.write(l.getID() + "\t" + j + "\t" + ranker.eval(l.get(j)) + "");
                    out.newLine();
                }
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in Evaluator::rank(): ", ex);
        }
    }

    /**
     * Write the models' score for each of the documents in a test rankings. These test rankings are splitted into k chunks where k=|models|.
     * Each model is applied on the data from the corresponding fold.
     * @param modelFiles
     * @param testFile
     * @param outputFile
     */
    public void score(final List<String> modelFiles, final String testFile, final String outputFile) {
        final List<List<RankList>> trainingData = new ArrayList<>();
        final List<List<RankList>> testData = new ArrayList<>();

        //read all samples
        final int nFold = modelFiles.size();
        //List<RankList> samples = FeatureManager.readInput(testFile);
        final List<RankList> samples = readInput(testFile);
        logger.info(() -> "Preparing " + nFold + "-fold test data... ");
        FeatureManager.prepareCV(samples, nFold, trainingData, testData);
        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"))) {
            for (int f = 0; f < nFold; f++) {
                final List<RankList> test = testData.get(f);
                final Ranker ranker = rFact.loadRankerFromFile(modelFiles.get(f));
                final int[] features = ranker.getFeatures();
                if (normalize) {
                    normalize(test, features);
                }
                for (final RankList l : test) {
                    for (int j = 0; j < l.size(); j++) {
                        out.write(l.getID() + "\t" + j + "\t" + ranker.eval(l.get(j)) + "");
                        out.newLine();
                    }
                }
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in Evaluator::score(): ", ex);
        }
    }

    /**
     * Similar to the above, except data has already been split. The k-th model will be applied on the k-th test file.
     * @param modelFiles
     * @param testFiles
     * @param outputFile
     */
    public void score(final List<String> modelFiles, final List<String> testFiles, final String outputFile) {
        final int nFold = modelFiles.size();
        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "UTF-8"))) {
            for (int f = 0; f < nFold; f++) {
                final List<RankList> test = readInput(testFiles.get(f));
                final Ranker ranker = rFact.loadRankerFromFile(modelFiles.get(f));
                final int[] features = ranker.getFeatures();

                if (normalize) {
                    normalize(test, features);
                }

                for (final RankList l : test) {
                    for (int j = 0; j < l.size(); j++) {
                        out.write(l.getID() + "\t" + j + "\t" + ranker.eval(l.get(j)));
                        out.newLine();
                    }
                }
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in Evaluator::score(): ", ex);
        }
    }

    /**
     * Use a pre-trained model to re-rank the test rankings. Save the output ranking in indri's run format
     * @param modelFile
     * @param testFile
     * @param indriRanking
     */
    public void rank(final String modelFile, final String testFile, final String indriRanking) {
        final Ranker ranker = rFact.loadRankerFromFile(modelFile);
        final int[] features = ranker.getFeatures();
        final List<RankList> test = readInput(testFile);

        if (normalize) {
            normalize(test, features);
        }
        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(indriRanking), "UTF-8"))) {
            for (final RankList l : test) {
                final double[] scores = new double[l.size()];
                for (int j = 0; j < l.size(); j++) {
                    scores[j] = ranker.eval(l.get(j));
                }
                final int[] idx = MergeSorter.sort(scores, false);
                for (int j = 0; j < idx.length; j++) {
                    final int k = idx[j];
                    final String str = l.getID() + " Q0 " + l.get(k).getDescription().replace("#", "").trim() + " " + (j + 1) + " "
                            + SimpleMath.round(scores[k], 5) + " indri";
                    out.write(str);
                    out.newLine();
                }
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in Evaluator::rank(): ", ex);
        }
    }

    /**
     * Generate a ranking in Indri's format from the input ranking
     * @param testFile
     * @param indriRanking
     */
    public void rank(final String testFile, final String indriRanking) {
        final List<RankList> test = readInput(testFile);

        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(indriRanking), "UTF-8"))) {
            for (final RankList l : test) {
                for (int j = 0; j < l.size(); j++) {
                    final String str = l.getID() + " Q0 " + l.get(j).getDescription().replace("#", "").trim() + " " + (j + 1) + " "
                            + SimpleMath.round(1.0 - 0.0001 * j, 5) + " indri";
                    out.write(str);
                    out.newLine();
                }
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in Evaluator::rank(): ", ex);
        }
    }

    /**
     * Use k pre-trained models to re-rank the test rankings. Test rankings will be splitted into k fold, where k=|models|.
     * Each model will be used to rank the data from the corresponding fold. Save the output ranking in indri's run format.
     * @param modelFiles
     * @param testFile
     * @param indriRanking
     */
    public void rank(final List<String> modelFiles, final String testFile, final String indriRanking) {
        final List<List<RankList>> trainingData = new ArrayList<>();
        final List<List<RankList>> testData = new ArrayList<>();

        //read all samples
        final int nFold = modelFiles.size();
        final List<RankList> samples = readInput(testFile);
        logger.info(() -> "Preparing " + nFold + "-fold test data... ");
        FeatureManager.prepareCV(samples, nFold, trainingData, testData);

        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(indriRanking), "UTF-8"))) {
            for (int f = 0; f < nFold; f++) {
                final List<RankList> test = testData.get(f);
                final Ranker ranker = rFact.loadRankerFromFile(modelFiles.get(f));
                final int[] features = ranker.getFeatures();
                if (normalize) {
                    normalize(test, features);
                }

                for (final RankList l : test) {
                    final double[] scores = new double[l.size()];
                    for (int j = 0; j < l.size(); j++) {
                        scores[j] = ranker.eval(l.get(j));
                    }
                    final int[] idx = MergeSorter.sort(scores, false);
                    for (int j = 0; j < idx.length; j++) {
                        final int k = idx[j];
                        final String str = l.getID() + " Q0 " + l.get(k).getDescription().replace("#", "").trim() + " " + (j + 1) + " "
                                + SimpleMath.round(scores[k], 5) + " indri";
                        out.write(str);
                        out.newLine();
                    }
                }
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in Evaluator::rank(): ", ex);
        }
    }

    /**
     * Similar to the above, except data has already been splitted. The k-th model will be applied on the k-th test file.
     * @param modelFiles
     * @param testFiles
     * @param indriRanking
     */
    public void rank(final List<String> modelFiles, final List<String> testFiles, final String indriRanking) {
        final int nFold = modelFiles.size();

        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(indriRanking), "UTF-8"))) {
            for (int f = 0; f < nFold; f++) {
                final List<RankList> test = readInput(testFiles.get(f));
                final Ranker ranker = rFact.loadRankerFromFile(modelFiles.get(f));
                final int[] features = ranker.getFeatures();

                if (normalize) {
                    normalize(test, features);
                }

                for (final RankList l : test) {
                    final double[] scores = new double[l.size()];

                    for (int j = 0; j < l.size(); j++) {
                        scores[j] = ranker.eval(l.get(j));
                    }

                    final int[] idx = MergeSorter.sort(scores, false);

                    for (int j = 0; j < idx.length; j++) {
                        final int k = idx[j];
                        final String str = l.getID() + " Q0 " + l.get(k).getDescription().replace("#", "").trim() + " " + (j + 1) + " "
                                + SimpleMath.round(scores[k], 5) + " indri";
                        out.write(str);
                        out.newLine();
                    }
                }
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in Evaluator::rank(): ", ex);
        }
    }

    /**
     * Split the input file into two with respect to a specified split size.
     * @param sampleFile Input data file
     * @param featureDefFile Feature definition file (if it's an empty string, all features in the input file will be used)
     * @param percentTrain How much of the input data will be used for training? (the remaining will be reserved for test/validation)
     * @param normalize Whether to do normalization.
     * @param trainingData [Output] Training data (after splitting)
     * @param testData [Output] Test (or validation) data (after splitting)
     * @return A list of ids of the features to be used for learning.
     */
    private int[] prepareSplit(final String sampleFile, final String featureDefFile, final double percentTrain, final boolean normalize,
            final List<RankList> trainingData, final List<RankList> testData) {
        //read input
        final List<RankList> data = readInput(sampleFile);

        //read features
        int[] features = readFeature(featureDefFile);

        // no features specified ==> use all features in the training file
        if (features == null) {
            features = FeatureManager.getFeatureFromSampleVector(data);
        }

        if (normalize) {
            normalize(data, features);
        }

        FeatureManager.prepareSplit(data, percentTrain, trainingData, testData);
        return features;
    }

    /**
     * Save systems' performance to file
     * @param ids Ranked list IDs.
     * @param scores Evaluation score (in whatever measure specified/calculated upstream such as NDCG@k, ERR@k, etc.)
     * @param prpFile Output filename.
     */
    public void savePerRankListPerformanceFile(final List<String> ids, final List<Double> scores, final String prpFile) {
        try (BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(prpFile)))) {
            for (int i = 0; i < ids.size(); i++) {
                out.write(testScorer.name() + "   " + ids.get(i) + "   " + scores.get(i));
                out.newLine();
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in Evaluator::savePerRankListPerformanceFile(): ", ex);
        }
    }
}
