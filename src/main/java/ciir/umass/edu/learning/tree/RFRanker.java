/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.tree;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.logging.Logger;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.learning.RankerFactory;
import ciir.umass.edu.learning.RankerType;
import ciir.umass.edu.learning.Sampler;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

public class RFRanker extends Ranker {
    private static final Logger logger = Logger.getLogger(RFRanker.class.getName());

    //Parameters
    //[a] general bagging parameters
    public static int nBag = 300;
    public static float subSamplingRate = 1.0f;//sampling of samples (*WITH* replacement)
    public static float featureSamplingRate = 0.3f;//sampling of features (*WITHOUT* replacement)
    //[b] what to do in each bag
    public static RankerType rType = RankerType.MART;//which algorithm to bag
    public static int nTrees = 1;//how many trees in each bag. If nTree > 1 ==> each bag will contain an ensemble of gradient boosted trees.
    public static int nTreeLeaves = 100;
    public static float learningRate = 0.1F;//or shrinkage. *ONLY* matters if nTrees > 1.
    public static int nThreshold = 256;
    public static int minLeafSupport = 1;

    //Variables
    protected Ensemble[] ensembles = null;//bag of ensembles, each can be a single tree or an ensemble of gradient boosted trees

    public RFRanker() {
    }

    public RFRanker(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    @Override
    public void init() {
        logger.info(() -> "Initializing... ");
        ensembles = new Ensemble[nBag];
        //initialize parameters for the tree(s) built in each bag
        LambdaMART.nTrees = nTrees;
        LambdaMART.nTreeLeaves = nTreeLeaves;
        LambdaMART.learningRate = learningRate;
        LambdaMART.nThreshold = nThreshold;
        LambdaMART.minLeafSupport = minLeafSupport;
        LambdaMART.nRoundToStopEarly = -1;//no early-stopping since we're doing bagging
        //turn on feature sampling
        FeatureHistogram.samplingRate = featureSamplingRate;
    }

    @Override
    public void learn() {
        final RankerFactory rf = new RankerFactory();
        logger.info(() -> "Training starts...");
        printLogLn(new int[] { 9, 9, 11 }, new String[] { "bag", scorer.name() + "-B", scorer.name() + "-OOB" });
        //start the bagging process
        for (int i = 0; i < nBag; i++) {
            final Sampler sp = new Sampler();
            //create a "bag" of samples by random sampling from the training set
            final List<RankList> bag = sp.doSampling(samples, subSamplingRate, true);
            //"out-of-bag" samples
            //List<RankList> outOfBag = sp.getRemains();
            final LambdaMART r = (LambdaMART) rf.createRanker(rType, bag, features, scorer);
            //r.setValidationSet(outOfBag);

            r.init();
            r.learn();
            //logger.info(()->new int[]{9, 9, 11}, new String[]{"b["+(i+1)+"]", SimpleMath.round(r.getScoreOnTrainingData(), 4)+"", SimpleMath.round(r.getScoreOnValidationData(), 4)+""});
            printLogLn(new int[] { 9, 9 }, new String[] { "b[" + (i + 1) + "]", SimpleMath.round(r.getScoreOnTrainingData(), 4) + "" });
            ensembles[i] = r.getEnsemble();
        }
        //Finishing up
        scoreOnTrainingData = scorer.score(rank(samples));
        logger.info(() -> "Finished sucessfully.");
        logger.info(() -> scorer.name() + " on training data: " + SimpleMath.round(scoreOnTrainingData, 4));
        if (validationSamples != null) {
            bestScoreOnValidationData = scorer.score(rank(validationSamples));
            logger.info(() -> scorer.name() + " on validation data: " + SimpleMath.round(bestScoreOnValidationData, 4));
        }
    }

    @Override
    public double eval(final DataPoint dp) {
        double s = 0;
        for (final Ensemble ensemble : ensembles) {
            s += ensemble.eval(dp);
        }
        return s / ensembles.length;
    }

    @Override
    public Ranker createNew() {
        return new RFRanker();
    }

    @Override
    public String toString() {
        String str = "";
        for (int i = 0; i < nBag; i++) {
            str += ensembles[i].toString() + "\n";
        }
        return str;
    }

    @Override
    public String model() {
        String output = "## " + name() + "\n";
        output += "## No. of bags = " + nBag + "\n";
        output += "## Sub-sampling = " + subSamplingRate + "\n";
        output += "## Feature-sampling = " + featureSamplingRate + "\n";
        output += "## No. of trees = " + nTrees + "\n";
        output += "## No. of leaves = " + nTreeLeaves + "\n";
        output += "## No. of threshold candidates = " + nThreshold + "\n";
        output += "## Learning rate = " + learningRate + "\n";
        output += "\n";
        output += toString();
        return output;
    }

    @Override
    public void loadFromString(final String fullText) {
        try (final BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            String content = "";
            String model = "";
            final List<Ensemble> ens = new ArrayList<>();
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                if (content.indexOf("##") == 0) {
                    continue;
                }
                //actual model component
                model += content;
                if (content.indexOf("</ensemble>") != -1) {
                    //load the ensemble
                    ens.add(new Ensemble(model));
                    model = "";
                }
            }
            final HashSet<Integer> uniqueFeatures = new HashSet<>();
            ensembles = new Ensemble[ens.size()];
            for (int i = 0; i < ens.size(); i++) {
                ensembles[i] = ens.get(i);
                //obtain used features
                final int[] fids = ens.get(i).getFeatures();
                for (int f = 0; f < fids.length; f++) {
                    if (!uniqueFeatures.contains(fids[f])) {
                        uniqueFeatures.add(fids[f]);
                    }
                }
            }
            int fi = 0;
            features = new int[uniqueFeatures.size()];
            for (final Integer f : uniqueFeatures) {
                features[fi++] = f.intValue();
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in RFRanker::load(): ", ex);
        }
    }

    @Override
    public void printParameters() {
        logger.info(() -> "No. of bags: " + nBag);
        logger.info(() -> "Sub-sampling: " + subSamplingRate);
        logger.info(() -> "Feature-sampling: " + featureSamplingRate);
        logger.info(() -> "No. of trees: " + nTrees);
        logger.info(() -> "No. of leaves: " + nTreeLeaves);
        logger.info(() -> "No. of threshold candidates: " + nThreshold);
        logger.info(() -> "Learning rate: " + learningRate);
    }

    @Override
    public String name() {
        return "Random Forests";
    }

    public Ensemble[] getEnsembles() {
        return ensembles;
    }
}
