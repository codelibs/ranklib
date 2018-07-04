/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.tree;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.parsing.ModelLineProducer;
import ciir.umass.edu.utilities.MergeSorter;
import ciir.umass.edu.utilities.MyThreadPool;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 *
 *  This class implements LambdaMART.
 *  Q. Wu, C.J.C. Burges, K. Svore and J. Gao. Adapting Boosting for Information Retrieval Measures.
 *  Journal of Information Retrieval, 2007.
 */
public class LambdaMART extends Ranker {
    private static final Logger logger = Logger.getLogger(LambdaMART.class.getName());

    //Parameters
    public static int nTrees = 1000;//the number of trees
    public static float learningRate = 0.1F;//or shrinkage
    public static int nThreshold = 256;
    public static int nRoundToStopEarly = 100;//If no performance gain on the *VALIDATION* data is observed in #rounds, stop the training process right away.
    public static int nTreeLeaves = 10;
    public static int minLeafSupport = 1;

    //Local variables
    protected float[][] thresholds = null;
    protected Ensemble ensemble = null;
    protected double[] modelScores = null;//on training data

    protected double[][] modelScoresOnValidation = null;
    protected int bestModelOnValidation = Integer.MAX_VALUE - 2;

    //Training instances prepared for MART
    protected DataPoint[] martSamples = null;//Need initializing only once
    protected int[][] sortedIdx = null;//sorted list of samples in @martSamples by each feature -- Need initializing only once
    protected FeatureHistogram hist = null;
    protected double[] pseudoResponses = null;//different for each iteration
    protected double[] weights = null;//different for each iteration
    protected double[] impacts = null; // accumulated impact of each feature

    public LambdaMART() {
    }

    public LambdaMART(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    @Override
    public void init() {
        logger.info(() -> "Initializing... ");
        //initialize samples for MART
        int dpCount = 0;
        for (int i = 0; i < samples.size(); i++) {
            final RankList rl = samples.get(i);
            dpCount += rl.size();
        }
        int current = 0;
        martSamples = new DataPoint[dpCount];
        modelScores = new double[dpCount];
        pseudoResponses = new double[dpCount];
        impacts = new double[features.length];
        weights = new double[dpCount];
        for (int i = 0; i < samples.size(); i++) {
            final RankList rl = samples.get(i);
            for (int j = 0; j < rl.size(); j++) {
                martSamples[current + j] = rl.get(j);
                modelScores[current + j] = 0.0F;
                pseudoResponses[current + j] = 0.0F;
                weights[current + j] = 0;
            }
            current += rl.size();
        }

        //sort (MART) samples by each feature so that we can quickly retrieve a sorted list of samples by any feature later on.
        sortedIdx = new int[features.length][];
        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            sortSamplesByFeature(0, features.length - 1);
        } else//multi-thread
        {
            final int[] partition = p.partition(features.length);
            for (int i = 0; i < partition.length - 1; i++) {
                p.execute(new SortWorker(this, partition[i], partition[i + 1] - 1));
            }
            p.await();
        }

        //Create a table of candidate thresholds (for each feature). Later on, we will select the best tree split from these candidates
        thresholds = new float[features.length][];
        for (int f = 0; f < features.length; f++) {
            //For this feature, keep track of the list of unique values and the max/min
            final List<Float> values = new ArrayList<>();
            float fmax = Float.NEGATIVE_INFINITY;
            float fmin = Float.MAX_VALUE;
            for (int i = 0; i < martSamples.length; i++) {
                final int k = sortedIdx[f][i];//get samples sorted with respect to this feature
                final float fv = martSamples[k].getFeatureValue(features[f]);
                values.add(fv);
                if (fmax < fv) {
                    fmax = fv;
                }
                if (fmin > fv) {
                    fmin = fv;
                }
                //skip all samples with the same feature value
                int j = i + 1;
                while (j < martSamples.length) {
                    if (martSamples[sortedIdx[f][j]].getFeatureValue(features[f]) > fv) {
                        break;
                    }
                    j++;
                }
                i = j - 1;//[i, j] gives the range of samples with the same feature value
            }

            if (values.size() <= nThreshold || nThreshold == -1) {
                thresholds[f] = new float[values.size() + 1];
                for (int i = 0; i < values.size(); i++) {
                    thresholds[f][i] = values.get(i);
                }
                thresholds[f][values.size()] = Float.MAX_VALUE;
            } else {
                final float step = (Math.abs(fmax - fmin)) / nThreshold;
                thresholds[f] = new float[nThreshold + 1];
                thresholds[f][0] = fmin;
                for (int j = 1; j < nThreshold; j++) {
                    thresholds[f][j] = thresholds[f][j - 1] + step;
                }
                thresholds[f][nThreshold] = Float.MAX_VALUE;
            }
        }

        if (validationSamples != null) {
            modelScoresOnValidation = new double[validationSamples.size()][];
            for (int i = 0; i < validationSamples.size(); i++) {
                modelScoresOnValidation[i] = new double[validationSamples.get(i).size()];
                Arrays.fill(modelScoresOnValidation[i], 0);
            }
        }

        //compute the feature histogram (this is used to speed up the procedure of finding the best tree split later on)
        hist = new FeatureHistogram();
        hist.construct(martSamples, pseudoResponses, sortedIdx, features, thresholds, impacts);
        //we no longer need the sorted indexes of samples
        sortedIdx = null;

    }

    @Override
    public void learn() {
        ensemble = new Ensemble();

        logger.info(() -> "Training starts...");
        printLogLn(new int[] { 7, 9, 9 }, new String[] { "#iter", scorer.name() + "-T", scorer.name() + "-V" });

        //Start the gradient boosting process
        for (int m = 0; m < nTrees; m++) {
            printLog(new int[] { 7 }, new String[] { Integer.toString(m + 1) });

            //Compute lambdas (which act as the "pseudo responses")
            //Create training instances for MART:
            //  - Each document is a training sample
            //	- The lambda for this document serves as its training label
            computePseudoResponses();

            //update the histogram with these training labels (the feature histogram will be used to find the best tree split)
            hist.update(pseudoResponses);

            //Fit a regression tree
            final RegressionTree rt = new RegressionTree(nTreeLeaves, martSamples, pseudoResponses, hist, minLeafSupport);
            rt.fit();

            //Add this tree to the ensemble (our model)
            ensemble.add(rt, learningRate);

            //update the outputs of the tree (with gamma computed using the Newton-Raphson method)
            updateTreeOutput(rt);

            //Update the model's outputs on all training samples
            final List<Split> leaves = rt.leaves();
            for (int i = 0; i < leaves.size(); i++) {
                final Split s = leaves.get(i);
                final int[] idx = s.getSamples();
                for (int j = 0; j < idx.length; j++) {
                    modelScores[idx[j]] += learningRate * s.getOutput();
                }
            }

            //clear references to data that is no longer used
            rt.clearSamples();

            //Evaluate the current model
            scoreOnTrainingData = computeModelScoreOnTraining();
            //**** NOTE ****
            //The above function to evaluate the current model on the training data is equivalent to a single call:
            //
            //		scoreOnTrainingData = scorer.score(rank(samples);
            //
            //However, this function is more efficient since it uses the cached outputs of the model (as opposed to re-evaluating the model
            //on the entire training set).

            printLog(new int[] { 9 }, new String[] { Double.toString(SimpleMath.round(scoreOnTrainingData, 4)) });

            //Evaluate the current model on the validation data (if available)
            if (validationSamples != null) {
                //Update the model's scores on all validation samples
                for (int i = 0; i < modelScoresOnValidation.length; i++) {
                    for (int j = 0; j < modelScoresOnValidation[i].length; j++) {
                        modelScoresOnValidation[i][j] += learningRate * rt.eval(validationSamples.get(i).get(j));
                    }
                }

                //again, equivalent to scoreOnValidation=scorer.score(rank(validationSamples)), but more efficient since we use the cached models' outputs
                final double score = computeModelScoreOnValidation();

                printLog(new int[] { 9 }, new String[] { Double.toString(SimpleMath.round(score, 4)) });
                if (score > bestScoreOnValidationData) {
                    bestScoreOnValidationData = score;
                    bestModelOnValidation = ensemble.treeCount() - 1;
                }
            }
            flushLog();

            //Should we stop early?
            if (m - bestModelOnValidation > nRoundToStopEarly) {
                break;
            }
        }

        //Rollback to the best model observed on the validation data
        while (ensemble.treeCount() > bestModelOnValidation + 1) {
            ensemble.remove(ensemble.treeCount() - 1);
        }

        //Finishing up
        scoreOnTrainingData = scorer.score(rank(samples));
        logger.info(() -> "Finished sucessfully.");
        logger.info(() -> scorer.name() + " on training data: " + SimpleMath.round(scoreOnTrainingData, 4));
        if (validationSamples != null) {
            bestScoreOnValidationData = scorer.score(rank(validationSamples));
            logger.info(() -> scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
        }

        logger.info(() -> "-- FEATURE IMPACTS");
        final int ftrsSorted[] = MergeSorter.sort(this.impacts, false);
        for (final int ftr : ftrsSorted) {
            logger.info(() -> " Feature " + features[ftr] + " reduced error " + impacts[ftr]);
        }
    }

    @Override
    public double eval(final DataPoint dp) {
        return ensemble.eval(dp);
    }

    @Override
    public Ranker createNew() {
        return new LambdaMART();
    }

    @Override
    public String toString() {
        return ensemble.toString();
    }

    @Override
    public String model() {
        final StringBuilder output = new StringBuilder();
        output.append("## " + name() + "\n");
        output.append("## No. of trees = " + nTrees + "\n");
        output.append("## No. of leaves = " + nTreeLeaves + "\n");
        output.append("## No. of threshold candidates = " + nThreshold + "\n");
        output.append("## Learning rate = " + learningRate + "\n");
        output.append("## Stop early = " + nRoundToStopEarly + "\n");
        output .append("\n");
        output.append(toString());
        return output.toString();
    }

    @Override
    public void loadFromString(final String fullText) {
        final ModelLineProducer lineByLine = new ModelLineProducer();
        lineByLine.parse(fullText, (model, endEns) -> {});
        //load the ensemble
        ensemble = new Ensemble(lineByLine.getModel().toString());
        features = ensemble.getFeatures();
    }

    @Override
    public void printParameters() {
        logger.info(() -> "No. of trees: " + nTrees);
        logger.info(() -> "No. of leaves: " + nTreeLeaves);
        logger.info(() -> "No. of threshold candidates: " + nThreshold);
        logger.info(() -> "Min leaf support: " + minLeafSupport);
        logger.info(() -> "Learning rate: " + learningRate);
        logger.info(() -> "Stop early: " + nRoundToStopEarly + " rounds without performance gain on validation data");
    }

    @Override
    public String name() {
        return "LambdaMART";
    }

    public Ensemble getEnsemble() {
        return ensemble;
    }

    protected void computePseudoResponses() {
        Arrays.fill(pseudoResponses, 0F);
        Arrays.fill(weights, 0);
        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            computePseudoResponses(0, samples.size() - 1, 0);
        } else //multi-threading
        {
            final List<LambdaComputationWorker> workers = new ArrayList<>();
            //divide the entire dataset into chunks of equal size for each worker thread
            final int[] partition = p.partition(samples.size());
            int current = 0;
            for (int i = 0; i < partition.length - 1; i++) {
                //execute the worker
                final LambdaComputationWorker wk = new LambdaComputationWorker(this, partition[i], partition[i + 1] - 1, current);
                workers.add(wk);//keep it so we can get back results from it later on
                p.execute(wk);

                if (i < partition.length - 2) {
                    for (int j = partition[i]; j <= partition[i + 1] - 1; j++) {
                        current += samples.get(j).size();
                    }
                }
            }

            //wait for all workers to complete before we move on to the next stage
            p.await();
        }
    }

    protected void computePseudoResponses(final int start, final int end, int current) {
        final int cutoff = scorer.getK();
        //compute the lambda for each document (a.k.a "pseudo response")
        for (int i = start; i <= end; i++) {
            final RankList orig = samples.get(i);
            final int[] idx = MergeSorter.sort(modelScores, current, current + orig.size() - 1, false);
            final RankList rl = new RankList(orig, idx, current);
            final double[][] changes = scorer.swapChange(rl);
            //NOTE: j, k are indices in the sorted (by modelScore) list, not the original
            // ==> need to map back with idx[j] and idx[k]
            for (int j = 0; j < rl.size(); j++) {
                final DataPoint p1 = rl.get(j);
                final int mj = idx[j];
                for (int k = 0; k < rl.size(); k++) {
                    if (j > cutoff && k > cutoff) {
                        break;
                    }
                    final DataPoint p2 = rl.get(k);
                    final int mk = idx[k];
                    if (p1.getLabel() > p2.getLabel()) {
                        final double deltaNDCG = Math.abs(changes[j][k]);
                        if (deltaNDCG > 0) {
                            final double rho = 1.0 / (1 + Math.exp(modelScores[mj] - modelScores[mk]));
                            final double lambda = rho * deltaNDCG;
                            pseudoResponses[mj] += lambda;
                            pseudoResponses[mk] -= lambda;
                            final double delta = rho * (1.0 - rho) * deltaNDCG;
                            weights[mj] += delta;
                            weights[mk] += delta;
                        }
                    }
                }
            }
            current += orig.size();
        }
    }

    protected void updateTreeOutput(final RegressionTree rt) {
        final List<Split> leaves = rt.leaves();
        for (int i = 0; i < leaves.size(); i++) {
            float s1 = 0F;
            float s2 = 0F;
            final Split s = leaves.get(i);
            final int[] idx = s.getSamples();
            for (final int k : idx) {
                s1 += pseudoResponses[k];
                s2 += weights[k];
            }
            if (s2 == 0) {
                s.setOutput(0);
            } else {
                s.setOutput(s1 / s2);
            }
        }
    }

    protected int[] sortSamplesByFeature(final DataPoint[] samples, final int fid) {
        final double[] score = new double[samples.length];
        for (int i = 0; i < samples.length; i++) {
            score[i] = samples[i].getFeatureValue(fid);
        }
        final int[] idx = MergeSorter.sort(score, true);
        return idx;
    }

    /**
     * This function is equivalent to the inherited function rank(...), but it uses the cached model's outputs instead of computing them from scratch.
     * @param rankListIndex
     * @param current
     * @return
     */
    protected RankList rank(final int rankListIndex, final int current) {
        final RankList orig = samples.get(rankListIndex);
        final double[] scores = new double[orig.size()];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = modelScores[current + i];
        }
        final int[] idx = MergeSorter.sort(scores, false);
        return new RankList(orig, idx);
    }

    protected float computeModelScoreOnTraining() {
        /*float s = 0;
        int current = 0;
        MyThreadPool p = MyThreadPool.getInstance();
        if(p.size() == 1)//single-thread
        	s = computeModelScoreOnTraining(0, samples.size()-1, current);
        else
        {
        	List<Worker> workers = new ArrayList<Worker>();
        	//divide the entire dataset into chunks of equal size for each worker thread
        	int[] partition = p.partition(samples.size());
        	for(int i=0;i<partition.length-1;i++)
        	{
        		//execute the worker
        		Worker wk = new Worker(this, partition[i], partition[i+1]-1, current);
        		workers.add(wk);//keep it so we can get back results from it later on
        		p.execute(wk);

        		if(i < partition.length-2)
        			for(int j=partition[i]; j<=partition[i+1]-1;j++)
        				current += samples.get(j).size();
        	}
        	//wait for all workers to complete before we move on to the next stage
        	p.await();
        	for(int i=0;i<workers.size();i++)
        		s += workers.get(i).score;
        }*/
        float s = computeModelScoreOnTraining(0, samples.size() - 1, 0);
        s = s / samples.size();
        return s;
    }

    protected float computeModelScoreOnTraining(final int start, final int end, final int current) {
        float s = 0;
        int c = current;

        for (int i = start; i <= end; i++) {
            s += scorer.score(rank(i, c));
            c += samples.get(i).size();
        }
        return s;
    }

    protected float computeModelScoreOnValidation() {
        /*float score = 0;
        MyThreadPool p = MyThreadPool.getInstance();
        if(p.size() == 1)//single-thread
        	score = computeModelScoreOnValidation(0, validationSamples.size()-1);
        else
        {
        	List<Worker> workers = new ArrayList<Worker>();
        	//divide the entire dataset into chunks of equal size for each worker thread
        	int[] partition = p.partition(validationSamples.size());
        	for(int i=0;i<partition.length-1;i++)
        	{
        		//execute the worker
        		Worker wk = new Worker(this, partition[i], partition[i+1]-1);
        		workers.add(wk);//keep it so we can get back results from it later on
        		p.execute(wk);
        	}
        	//wait for all workers to complete before we move on to the next stage
        	p.await();
        	for(int i=0;i<workers.size();i++)
        		score += workers.get(i).score;
        }*/
        final float score = computeModelScoreOnValidation(0, validationSamples.size() - 1);
        return score / validationSamples.size();
    }

    protected float computeModelScoreOnValidation(final int start, final int end) {
        float score = 0;
        for (int i = start; i <= end; i++) {
            final int[] idx = MergeSorter.sort(modelScoresOnValidation[i], false);
            score += scorer.score(new RankList(validationSamples.get(i), idx));
        }
        return score;
    }

    protected void sortSamplesByFeature(final int fStart, final int fEnd) {
        for (int i = fStart; i <= fEnd; i++) {
            sortedIdx[i] = sortSamplesByFeature(martSamples, features[i]);
        }
    }

    //For multi-threading processing
    class SortWorker implements Runnable {
        LambdaMART ranker = null;
        int start = -1;
        int end = -1;

        SortWorker(final LambdaMART ranker, final int start, final int end) {
            this.ranker = ranker;
            this.start = start;
            this.end = end;
        }

        @Override
        public void run() {
            ranker.sortSamplesByFeature(start, end);
        }
    }

    class LambdaComputationWorker implements Runnable {
        LambdaMART ranker = null;
        int rlStart = -1;
        int rlEnd = -1;
        int martStart = -1;

        LambdaComputationWorker(final LambdaMART ranker, final int rlStart, final int rlEnd, final int martStart) {
            this.ranker = ranker;
            this.rlStart = rlStart;
            this.rlEnd = rlEnd;
            this.martStart = martStart;
        }

        @Override
        public void run() {
            ranker.computePseudoResponses(rlStart, rlEnd, martStart);
        }
    }

    class Worker implements Runnable {
        LambdaMART ranker = null;
        int rlStart = -1;
        int rlEnd = -1;
        int martStart = -1;
        int type = -1;

        //compute score on validation
        float score = 0;

        Worker(final LambdaMART ranker, final int rlStart, final int rlEnd) {
            type = 3;
            this.ranker = ranker;
            this.rlStart = rlStart;
            this.rlEnd = rlEnd;
        }

        Worker(final LambdaMART ranker, final int rlStart, final int rlEnd, final int martStart) {
            type = 4;
            this.ranker = ranker;
            this.rlStart = rlStart;
            this.rlEnd = rlEnd;
            this.martStart = martStart;
        }

        @Override
        public void run() {
            if (type == 4) {
                score = ranker.computeModelScoreOnTraining(rlStart, rlEnd, martStart);
            } else if (type == 3) {
                score = ranker.computeModelScoreOnValidation(rlStart, rlEnd);
            }
        }
    }
}
