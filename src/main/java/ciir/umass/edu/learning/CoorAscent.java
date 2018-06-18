/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.KeyValuePair;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 *
 * This class implements the linear ranking model known as Coordinate Ascent. It was proposed in this paper:
 *  D. Metzler and W.B. Croft. Linear feature-based models for information retrieval. Information Retrieval, 10(3): 257-274, 2007.
 */
public class CoorAscent extends Ranker {
    private static final Logger logger = Logger.getLogger(CoorAscent.class.getName());

    //Parameters
    public static int nRestart = 5;
    public static int nMaxIteration = 25;
    public static double stepBase = 0.05;
    public static double stepScale = 2.0;
    public static double tolerance = 0.001;
    public static boolean regularized = false;
    public static double slack = 0.001;//regularized parameter

    //Local variables
    public double[] weight = null;

    protected int current_feature = -1;//used only during learning
    protected double weight_change = -1.0;//used only during learning

    public CoorAscent() {

    }

    public CoorAscent(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    @Override
    public void init() {
        logger.info(() -> "Initializing... ");
        weight = new double[features.length];
        Arrays.fill(weight, 1.0 / features.length);
    }

    @Override
    public void learn() {
        final double[] regVector = new double[weight.length];
        copy(weight, regVector);//uniform weight distribution

        //this holds the final best model/score
        double[] bestModel = null;
        double bestModelScore = 0.0;

        // look in both directions and with feature removed.
        final int[] sign = new int[] { 1, -1, 0 };

        logger.info(() -> "Training starts...");

        for (int r = 0; r < nRestart; r++) {
            if (logger.isLoggable(Level.INFO)) {
                logger.info("[+] Random restart #" + (r + 1) + "/" + nRestart + "...");
            }
            int consecutive_fails = 0;

            //initialize weight vector
            for (int i = 0; i < weight.length; i++) {
                weight[i] = 1.0f / features.length;
            }

            current_feature = -1;
            final double startScore = scorer.score(rank(samples));//compute all the scores (in whatever metric specified) and store them as cache

            //local best (within the current restart cycle)
            double bestScore = startScore;
            final double[] bestWeight = new double[weight.length];
            copy(weight, bestWeight);

            //There must be at least one feature increasing whose weight helps
            while ((weight.length > 1 && consecutive_fails < weight.length - 1) || (weight.length == 1 && consecutive_fails == 0)) {
                logger.info(() -> "Shuffling features' order...");
                logger.info(() -> "Optimizing weight vector... ");
                printLogLn(new int[] { 7, 8, 7 }, new String[] { "Feature", "weight", scorer.name() });

                final int[] fids = getShuffledFeatures();//contain index of elements in the variable @features
                //Try maximizing each feature individually
                for (int i = 0; i < fids.length; i++) {
                    current_feature = fids[i];//this will trigger the "else" branch in the procedure rank()

                    final double origWeight = weight[fids[i]];
                    double totalStep = 0;
                    double bestTotalStep = 0;
                    boolean succeeds = false;//whether or not we succeed in finding a better weight value for the current feature
                    for (int s = 0; s < sign.length; s++)//search by both increasing and decreasing
                    {
                        final int dir = sign[s];
                        double step = 0.001 * dir;
                        if (origWeight != 0.0 && Math.abs(step) > 0.5 * Math.abs(origWeight)) {
                            step = stepBase * Math.abs(origWeight);
                        }
                        totalStep = step;
                        int numIter = nMaxIteration;
                        if (dir == 0) {
                            numIter = 1;
                            totalStep = -origWeight;
                        }
                        for (int j = 0; j < numIter; j++) {
                            final double w = origWeight + totalStep;
                            weight_change = step;//weight_change is used in the "else" branch in the procedure rank()
                            weight[fids[i]] = w;
                            double score = scorer.score(rank(samples));
                            if (regularized) {
                                final double penalty = slack * getDistance(weight, regVector);
                                score -= penalty;
                                //logger.info(()->"Penalty: " + penalty);
                            }
                            if (score > bestScore)//better than the local best, replace the local best with this model
                            {
                                bestScore = score;
                                bestTotalStep = totalStep;
                                succeeds = true;
                                final String bw = ((weight[fids[i]] > 0) ? "+" : "") + SimpleMath.round(weight[fids[i]], 4);
                                printLogLn(new int[] { 7, 8, 7 },
                                        new String[] { features[fids[i]] + "", bw + "", SimpleMath.round(bestScore, 4) + "" });
                            }
                            if (j < nMaxIteration - 1) {
                                step *= stepScale;
                                totalStep += step;
                            }
                        }
                        if (succeeds) {
                            break;//no need to search the other direction (e.g. sign = '-')
                        } else if (s < sign.length - 1) {
                            weight_change = -totalStep;
                            updateCached();//restore the cached to reflect the orig. weight for the current feature
                            //so that we can start searching in the other direction (since the optimization in the first direction failed)
                            weight[fids[i]] = origWeight;//restore the weight to its initial value
                        }
                    }
                    if (succeeds) {
                        weight_change = bestTotalStep - totalStep;
                        updateCached();//restore the cached to reflect the best weight for the current feature
                        weight[fids[i]] = origWeight + bestTotalStep;
                        consecutive_fails = 0;//since we found a better weight value
                        final double sum = normalize(weight);
                        scaleCached(sum);
                        copy(weight, bestWeight);
                    } else {
                        consecutive_fails++;
                        weight_change = -totalStep;
                        updateCached();//restore the cached to reflect the orig. weight for the current feature since the optimization failed
                        //Restore the orig. weight value
                        weight[fids[i]] = origWeight;
                    }
                }

                //if we haven't made much progress then quit
                if (bestScore - startScore < tolerance) {
                    break;
                }
            }
            //update the (global) best model with the best model found in this round
            if (validationSamples != null) {
                current_feature = -1;
                bestScore = scorer.score(rank(validationSamples));
            }
            if (bestModel == null || bestScore > bestModelScore) {
                bestModelScore = bestScore;
                bestModel = bestWeight;
            }
        }

        copy(bestModel, weight);
        current_feature = -1;//turn off the cache mode
        scoreOnTrainingData = SimpleMath.round(scorer.score(rank(samples)), 4);
        logger.info(() -> "Finished sucessfully.");
        logger.info(() -> scorer.name() + " on training data: " + scoreOnTrainingData);

        if (validationSamples != null) {
            bestScoreOnValidationData = scorer.score(rank(validationSamples));
            logger.info(() -> scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
        }
    }

    @Override
    public RankList rank(final RankList rl) {
        final double[] score = new double[rl.size()];
        if (current_feature == -1) {
            for (int i = 0; i < rl.size(); i++) {
                for (int j = 0; j < features.length; j++) {
                    score[i] += weight[j] * rl.get(i).getFeatureValue(features[j]);
                }
                rl.get(i).setCached(score[i]);//use cache of a data point to store its score given the model at this state
            }
        } else//This branch is only active during the training process. Here we trade the "clean" codes for efficiency
        {
            for (int i = 0; i < rl.size(); i++) {
                //cached score = a_1*x_1 + a_2*x_2 + ... + a_n*x_n
                //a_2 ==> a'_2
                //new score = cached score + (a'_2 - a_2)*x_2  ====> NO NEED TO RE-COMPUTE THE WHOLE THING
                score[i] = rl.get(i).getCached() + weight_change * rl.get(i).getFeatureValue(features[current_feature]);
                rl.get(i).setCached(score[i]);
            }
        }
        final int[] idx = MergeSorter.sort(score, false);
        return new RankList(rl, idx);
    }

    @Override
    public double eval(final DataPoint p) {
        double score = 0.0;
        for (int i = 0; i < features.length; i++) {
            score += weight[i] * p.getFeatureValue(features[i]);
        }
        return score;
    }

    @Override
    public Ranker createNew() {
        return new CoorAscent();
    }

    @Override
    public String toString() {
        String output = "";
        for (int i = 0; i < weight.length; i++) {
            output += features[i] + ":" + weight[i] + ((i == weight.length - 1) ? "" : " ");
        }
        return output;
    }

    @Override
    public String model() {
        String output = "## " + name() + "\n";
        output += "## Restart = " + nRestart + "\n";
        output += "## MaxIteration = " + nMaxIteration + "\n";
        output += "## StepBase = " + stepBase + "\n";
        output += "## StepScale = " + stepScale + "\n";
        output += "## Tolerance = " + tolerance + "\n";
        output += "## Regularized = " + regularized + "\n";
        output += "## Slack = " + slack + "\n";
        output += toString();
        return output;
    }

    @Override
    public void loadFromString(final String fullText) {
        try (final BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            String content = "";

            KeyValuePair kvp = null;
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                if (content.indexOf("##") == 0) {
                    continue;
                }
                kvp = new KeyValuePair(content);
                break;
            }
            assert (kvp != null);

            final List<String> keys = kvp.keys();
            final List<String> values = kvp.values();
            weight = new double[keys.size()];
            features = new int[keys.size()];
            for (int i = 0; i < keys.size(); i++) {
                features[i] = Integer.parseInt(keys.get(i));
                weight[i] = Double.parseDouble(values.get(i));
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in CoorAscent::load(): ", ex);
        }
    }

    @Override
    public void printParameters() {
        logger.info(() -> "No. of random restarts: " + nRestart);
        logger.info(() -> "No. of iterations to search in each direction: " + nMaxIteration);
        logger.info(() -> "Tolerance: " + tolerance);
        if (regularized) {
            logger.info(() -> "Reg. param: " + slack);
        } else {
            logger.info(() -> "Regularization: No");
        }
    }

    @Override
    public String name() {
        return "Coordinate Ascent";
    }

    private void updateCached() {
        for (int j = 0; j < samples.size(); j++) {
            final RankList rl = samples.get(j);
            for (int i = 0; i < rl.size(); i++) {
                //cached score = a_1*x_1 + a_2*x_2 + ... + a_n*x_n
                //a_2 ==> a'_2
                //new score = cached score + (a'_2 - a_2)*x_2  ====> NO NEED TO RE-COMPUTE THE WHOLE THING
                final double score = rl.get(i).getCached() + weight_change * rl.get(i).getFeatureValue(features[current_feature]);
                rl.get(i).setCached(score);
            }
        }
    }

    private void scaleCached(final double sum) {
        for (int j = 0; j < samples.size(); j++) {
            final RankList rl = samples.get(j);
            for (int i = 0; i < rl.size(); i++) {
                rl.get(i).setCached(rl.get(i).getCached() / sum);
            }
        }
    }

    private int[] getShuffledFeatures() {
        final int[] fids = new int[features.length];
        final List<Integer> l = new ArrayList<>();
        for (int i = 0; i < features.length; i++) {
            l.add(i);
        }
        Collections.shuffle(l);
        for (int i = 0; i < l.size(); i++) {
            fids[i] = l.get(i);
        }
        return fids;
    }

    private double getDistance(final double[] w1, final double[] w2) {
        assert (w1.length == w2.length);
        double s1 = 0.0;
        double s2 = 0.0;
        for (int i = 0; i < w1.length; i++) {
            s1 += Math.abs(w1[i]);
            s2 += Math.abs(w2[i]);
        }
        double dist = 0.0;
        for (int i = 0; i < w1.length; i++) {
            final double t = w1[i] / s1 - w2[i] / s2;
            dist += t * t;
        }
        return Math.sqrt(dist);
    }

    private double normalize(final double[] weights) {
        double sum = 0.0;
        for (final double weight2 : weights) {
            sum += Math.abs(weight2);
        }
        if (sum > 0) {
            for (int j = 0; j < weights.length; j++) {
                weights[j] /= sum;
            }
        } else {
            sum = 1;
            for (int j = 0; j < weights.length; j++) {
                weights[j] = 1.0 / weights.length;
            }
        }
        return sum;
    }

    public void copyModel(final CoorAscent ranker) {
        weight = new double[features.length];
        if (ranker.weight.length != weight.length) {
            RankLibError.create("These two models use different feature set!!");
        }
        copy(ranker.weight, weight);
        logger.info(() -> "Model loaded.");
    }

    public double distance(final CoorAscent ca) {
        return getDistance(weight, ca.weight);
    }
}