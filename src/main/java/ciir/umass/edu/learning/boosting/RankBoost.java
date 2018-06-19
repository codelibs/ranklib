/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.boosting;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 *
 * This class implements RankBoost.
 *  Y. Freund, R. Iyer, R. Schapire, and Y. Singer. An efficient boosting algorithm for combining preferences.
 *  The Journal of Machine Learning Research, 4: 933-969, 2003.
 */
public class RankBoost extends Ranker {
    private static final Logger logger = Logger.getLogger(RankBoost.class.getName());

    public static int nIteration = 300;//number of rounds
    public static int nThreshold = 10;

    protected double[][][] sweight = null;//sample weight D(x_0, x_1) -- the weight of x_1 ranked above x_2
    protected double[][] potential = null;//pi(x)
    protected List<List<int[]>> sortedSamples = new ArrayList<>();
    protected double[][] thresholds = null;//candidate values for weak rankers' threshold, selected from feature values
    protected int[][] tSortedIdx = null;//sorted (descend) index for @thresholds

    protected List<RBWeakRanker> wRankers = null;//best weak rankers at each round
    protected List<Double> rWeight = null;//alpha (weak rankers' weight)

    //to store the best model on validation data (if specified)
    protected List<RBWeakRanker> bestModelRankers = new ArrayList<>();
    protected List<Double> bestModelWeights = new ArrayList<>();

    private double R_t = 0.0;
    private double Z_t = 1.0;
    private int totalCorrectPairs = 0;//crucial pairs

    public RankBoost() {

    }

    public RankBoost(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    private int[] reorder(final RankList rl, final int fid) {
        final double[] score = new double[rl.size()];
        for (int i = 0; i < rl.size(); i++) {
            score[i] = rl.get(i).getFeatureValue(fid);
        }
        return MergeSorter.sort(score, false);
    }

    /**
     * compute the potential (pi) based on the current sample (pair) weight distribution D_t.
     */
    private void updatePotential() {
        for (int i = 0; i < samples.size(); i++) {
            final RankList rl = samples.get(i);
            for (int j = 0; j < rl.size(); j++) {
                double p = 0.0;
                for (int k = j + 1; k < rl.size(); k++) {
                    p += sweight[i][j][k];
                }
                for (int k = 0; k < j; k++) {
                    p -= sweight[i][k][j];
                }
                potential[i][j] = p;
            }
        }
    }

    /**
     * Find the <feature, threshold> that maximize r (which will approximately minimize the exponential error on the training data).
     * Create the weak ranker h_t from this pair <feature, threshold>. h_t(p) > threshold => h_t(p)=1; h_t(p)=0 otherwise.
     * @return The learned weak ranker. The value of <i>current_r</i> is also updated to be the best r observed.
     */
    private RBWeakRanker learnWeakRanker() {
        int bestFid = -1;
        double maxR = -10;
        double bestThreshold = -1.0;
        for (int i = 0; i < features.length; i++) {
            final List<int[]> sSortedIndex = sortedSamples.get(i);//samples sorted (descending) by the current feature
            final int[] idx = tSortedIdx[i];//candidate thresholds for the current features
            final int[] last = new int[samples.size()];//the last "touched" (and taken) position in each sample rank list
            for (int j = 0; j < samples.size(); j++) {
                last[j] = -1;
            }

            double r = 0.0;
            for (final int element : idx) {
                final double t = thresholds[i][element];
                //we want something t < threshold <= tp
                for (int k = 0; k < samples.size(); k++) {
                    final RankList rl = samples.get(k);
                    final int[] sk = sSortedIndex.get(k);
                    for (int l = last[k] + 1; l < rl.size(); l++) {
                        final DataPoint p = rl.get(sk[l]);
                        if (p.getFeatureValue(features[i]) > t)//take it
                        {
                            r += potential[k][sk[l]];
                            last[k] = l;
                        } else {
                            break;
                        }
                    }
                }
                //finish computing r
                if (r > maxR) {
                    maxR = r;
                    bestThreshold = t;
                    bestFid = features[i];
                }
            }
        }
        if (bestFid == -1) {
            return null;
        }

        R_t = Z_t * maxR;//save it so we won't have to re-compute when we need it

        return new RBWeakRanker(bestFid, bestThreshold);
    }

    @Override
    public void init() {
        logger.info(() -> "Initializing... ");

        wRankers = new ArrayList<>();
        rWeight = new ArrayList<>();

        //for each (true) ranked list, we only care about correctly ranked pair (e.g. L={1,2,3} => <1,2>, <1,3>, <2,3>)
        //	count the number of correctly ranked pairs from sample ranked list
        totalCorrectPairs = 0;
        for (int i = 0; i < samples.size(); i++) {
            samples.set(i, samples.get(i).getCorrectRanking());//make sure the training samples are in correct ranking
            final RankList rl = samples.get(i);
            for (int j = 0; j < rl.size() - 1; j++) {
                for (int k = rl.size() - 1; k >= j + 1 && rl.get(j).getLabel() > rl.get(k).getLabel(); k--) {
                    //for(int k=j+1;k<rl.size();k++)
                    //if(rl.get(j).getLabel() > rl.get(k).getLabel())
                    totalCorrectPairs++;
                }
            }
        }

        //compute weight for all correctly ranked pairs
        sweight = new double[samples.size()][][];
        for (int i = 0; i < samples.size(); i++) {
            final RankList rl = samples.get(i);
            sweight[i] = new double[rl.size()][];
            for (int j = 0; j < rl.size() - 1; j++) {
                sweight[i][j] = new double[rl.size()];
                for (int k = j + 1; k < rl.size(); k++) {
                    if (rl.get(j).getLabel() > rl.get(k).getLabel()) {
                        sweight[i][j][k] = 1.0 / totalCorrectPairs;
                    } else {
                        sweight[i][j][k] = 0.0;//not crucial pairs
                    }
                }
            }
        }

        //init potential matrix
        potential = new double[samples.size()][];
        for (int i = 0; i < samples.size(); i++) {
            potential[i] = new double[samples.get(i).size()];
        }

        if (nThreshold <= 0) {
            //create a table of candidate thresholds (for each feature) for weak rankers (they are just all possible feature values)
            int count = 0;
            for (int i = 0; i < samples.size(); i++) {
                count += samples.get(i).size();
            }

            thresholds = new double[features.length][];
            for (int i = 0; i < features.length; i++) {
                thresholds[i] = new double[count];
            }

            int c = 0;
            for (int i = 0; i < samples.size(); i++) {
                final RankList rl = samples.get(i);
                for (int j = 0; j < rl.size(); j++) {
                    for (int k = 0; k < features.length; k++) {
                        thresholds[k][c] = rl.get(j).getFeatureValue(features[k]);
                    }
                    c++;
                }
            }
        } else {
            final double[] fmax = new double[features.length];
            final double[] fmin = new double[features.length];
            for (int i = 0; i < features.length; i++) {
                fmax[i] = -1E6;
                fmin[i] = 1E6;
            }

            for (int i = 0; i < samples.size(); i++) {
                final RankList rl = samples.get(i);
                for (int j = 0; j < rl.size(); j++) {
                    for (int k = 0; k < features.length; k++) {
                        final double f = rl.get(j).getFeatureValue(features[k]);
                        if (f > fmax[k]) {
                            fmax[k] = f;
                        }
                        if (f < fmin[k]) {
                            fmin[k] = f;
                        }
                    }
                }
            }

            thresholds = new double[features.length][];
            for (int i = 0; i < features.length; i++) {
                final double step = (Math.abs(fmax[i] - fmin[i])) / nThreshold;
                thresholds[i] = new double[nThreshold + 1];
                thresholds[i][0] = fmax[i];
                for (int j = 1; j < nThreshold; j++) {
                    thresholds[i][j] = thresholds[i][j - 1] - step;
                }
                thresholds[i][nThreshold] = fmin[i] - 1.0E8;
            }
        }

        //sort this table with respect to each feature (each row of the matrix @thresholds)
        tSortedIdx = new int[features.length][];
        for (int i = 0; i < features.length; i++) {
            tSortedIdx[i] = MergeSorter.sort(thresholds[i], false);
        }

        //now create a sorted lists of every samples ranked list with respect to each feature
        //e.g. Feature f_i <==> all sample ranked list is now ranked with respect to f_i
        for (final int feature : features) {
            final List<int[]> idx = new ArrayList<>();
            for (int j = 0; j < samples.size(); j++) {
                idx.add(reorder(samples.get(j), feature));
            }
            sortedSamples.add(idx);
        }
    }

    @Override
    public void learn() {
        logger.info(() -> "Training starts...");
        printLogLn(new int[] { 7, 8, 9, 9, 9, 9 },
                new String[] { "#iter", "Sel. F.", "Threshold", "Error", scorer.name() + "-T", scorer.name() + "-V" });

        for (int t = 1; t <= nIteration; t++) {
            updatePotential();
            //learn the weak ranker
            final RBWeakRanker wr = learnWeakRanker();
            if (wr == null) {
                break;
            }
            //compute weak ranker weight
            final double alpha_t = 0.5 * SimpleMath.ln((Z_t + R_t) / (Z_t - R_t));//@current_r is computed in learnWeakRanker()

            wRankers.add(wr);
            rWeight.add(alpha_t);

            //update sample pairs' weight distribution
            Z_t = 0.0;//normalization factor
            for (int i = 0; i < samples.size(); i++) {
                final RankList rl = samples.get(i);
                final double[][] D_t = new double[rl.size()][];
                for (int j = 0; j < rl.size() - 1; j++) {
                    D_t[j] = new double[rl.size()];
                    for (int k = j + 1; k < rl.size(); k++) {
                        //we should rank x_j higher than x_k
                        //so if our h_t does so, decrease the weight of this pair
                        //otherwise, increase its weight
                        D_t[j][k] = sweight[i][j][k] * Math.exp(alpha_t * (wr.score(rl.get(k)) - wr.score(rl.get(j))));
                        Z_t += D_t[j][k];
                    }
                }
                sweight[i] = D_t;
            }

            printLog(new int[] { 7, 8, 9, 9 },
                    new String[] { t + "", wr.getFid() + "", SimpleMath.round(wr.getThreshold(), 4) + "", SimpleMath.round(R_t, 4) + "" });
            if (t % 1 == 0) {
                printLog(new int[] { 9 }, new String[] { SimpleMath.round(scorer.score(rank(samples)), 4) + "" });
                if (validationSamples != null) {
                    final double score = scorer.score(rank(validationSamples));
                    if (score > bestScoreOnValidationData) {
                        bestScoreOnValidationData = score;
                        bestModelRankers.clear();
                        bestModelRankers.addAll(wRankers);
                        bestModelWeights.clear();
                        bestModelWeights.addAll(rWeight);
                    }
                    printLog(new int[] { 9 }, new String[] { SimpleMath.round(score, 4) + "" });
                }
            }

            //logger.info(()->"Z_t = " + Z + "\tr = " + current_r + "\t" + Math.sqrt(1.0 - current_r*current_r));
            //normalize sweight to make sure it is a valid distribution
            for (int i = 0; i < samples.size(); i++) {
                final RankList rl = samples.get(i);
                for (int j = 0; j < rl.size() - 1; j++) {
                    for (int k = j + 1; k < rl.size(); k++) {
                        sweight[i][j][k] /= Z_t;
                    }
                }
            }

        }

        //if validation data is specified ==> best model on this data has been saved
        //we now restore the current model to that best model
        if (validationSamples != null && bestModelRankers.size() > 0) {
            wRankers.clear();
            rWeight.clear();
            wRankers.addAll(bestModelRankers);
            rWeight.addAll(bestModelWeights);
        }

        scoreOnTrainingData = SimpleMath.round(scorer.score(rank(samples)), 4);
        logger.info(() -> "Finished sucessfully.");
        logger.info(() -> scorer.name() + " on training data: " + scoreOnTrainingData);
        if (validationSamples != null) {
            bestScoreOnValidationData = scorer.score(rank(validationSamples));
            logger.info(() -> scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
        }
    }

    @Override
    public double eval(final DataPoint p) {
        double score = 0.0;
        for (int j = 0; j < wRankers.size(); j++) {
            score += rWeight.get(j) * wRankers.get(j).score(p);
        }
        return score;
    }

    @Override
    public Ranker createNew() {
        return new RankBoost();
    }

    @Override
    public String toString() {
        String output = "";
        for (int i = 0; i < wRankers.size(); i++) {
            output += wRankers.get(i).toString() + ":" + rWeight.get(i) + ((i == rWeight.size() - 1) ? "" : " ");
        }
        return output;
    }

    @Override
    public String model() {
        String output = "## " + name() + "\n";
        output += "## Iteration = " + nIteration + "\n";
        output += "## No. of threshold candidates = " + nThreshold + "\n";
        output += toString();
        return output;
    }

    @Override
    public void loadFromString(final String fullText) {
        try (final BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            String content = "";

            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                if (content.indexOf("##") == 0) {
                    continue;
                }
                break;
            }

            rWeight = new ArrayList<>();
            wRankers = new ArrayList<>();

            final int idx = content.lastIndexOf('#');
            if (idx != -1) {
                content = content.substring(0, idx).trim();//remove the comment part at the end of the line
            }

            final String[] fs = content.split(" ");
            for (int i = 0; i < fs.length; i++) {
                fs[i] = fs[i].trim();
                if (fs[i].compareTo("") == 0) {
                    continue;
                }
                final String[] strs = fs[i].split(":");
                final int fid = Integer.parseInt(strs[0]);
                final double threshold = Double.parseDouble(strs[1]);
                final double weight = Double.parseDouble(strs[2]);
                rWeight.add(weight);
                wRankers.add(new RBWeakRanker(fid, threshold));
            }

            features = new int[rWeight.size()];
            for (int i = 0; i < rWeight.size(); i++) {
                features[i] = wRankers.get(i).getFid();
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in RankBoost::load(): ", ex);
        }
    }

    @Override
    public void printParameters() {
        logger.info(() -> "No. of rounds: " + nIteration);
        logger.info(() -> "No. of threshold candidates: " + nThreshold);
    }

    @Override
    public String name() {
        return "RankBoost";
    }
}
