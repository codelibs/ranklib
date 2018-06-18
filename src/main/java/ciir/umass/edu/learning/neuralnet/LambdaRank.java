/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.neuralnet;

import java.util.List;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;

public class LambdaRank extends RankNet {
    //Parameters
    //Inherits *ALL* parameters from RankNet

    //Variables
    protected float[][] targetValue = null;

    public LambdaRank() {

    }

    public LambdaRank(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    @Override
    protected int[][] batchFeedForward(final RankList rl) {
        final int[][] pairMap = new int[rl.size()][];
        targetValue = new float[rl.size()][];
        for (int i = 0; i < rl.size(); i++) {
            addInput(rl.get(i));
            propagate(i);

            int count = 0;
            for (int j = 0; j < rl.size(); j++) {
                if (rl.get(i).getLabel() > rl.get(j).getLabel() || rl.get(i).getLabel() < rl.get(j).getLabel()) {
                    count++;
                }
            }

            pairMap[i] = new int[count];
            targetValue[i] = new float[count];

            int k = 0;
            for (int j = 0; j < rl.size(); j++) {
                if (rl.get(i).getLabel() > rl.get(j).getLabel() || rl.get(i).getLabel() < rl.get(j).getLabel()) {
                    pairMap[i][k] = j;
                    if (rl.get(i).getLabel() > rl.get(j).getLabel()) {
                        targetValue[i][k] = 1;
                    } else {
                        targetValue[i][k] = 0;
                    }
                    k++;
                }
            }
        }
        return pairMap;
    }

    @Override
    protected void batchBackPropagate(final int[][] pairMap, final float[][] pairWeight) {
        for (int i = 0; i < pairMap.length; i++) {
            final PropParameter p = new PropParameter(i, pairMap, pairWeight, targetValue);
            //back-propagate
            outputLayer.computeDelta(p);//starting at the output layer
            for (int j = layers.size() - 2; j >= 1; j--) {
                layers.get(j).updateDelta(p);
            }

            //weight update
            outputLayer.updateWeight(p);
            for (int j = layers.size() - 2; j >= 1; j--) {
                layers.get(j).updateWeight(p);
            }
        }
    }

    @Override
    protected RankList internalReorder(final RankList rl) {
        return rank(rl);
    }

    @Override
    protected float[][] computePairWeight(final int[][] pairMap, final RankList rl) {
        final double[][] changes = scorer.swapChange(rl);
        final float[][] weight = new float[pairMap.length][];
        for (int i = 0; i < weight.length; i++) {
            weight[i] = new float[pairMap[i].length];
            for (int j = 0; j < pairMap[i].length; j++) {
                final int sign = (rl.get(i).getLabel() > rl.get(pairMap[i][j]).getLabel()) ? 1 : -1;
                weight[i][j] = (float) Math.abs(changes[i][pairMap[i][j]]) * sign;
            }
        }
        return weight;
    }

    @Override
    protected void estimateLoss() {
        misorderedPairs = 0;
        for (int j = 0; j < samples.size(); j++) {
            final RankList rl = samples.get(j);
            for (int k = 0; k < rl.size() - 1; k++) {
                final double o1 = eval(rl.get(k));
                for (int l = k + 1; l < rl.size(); l++) {
                    if (rl.get(k).getLabel() > rl.get(l).getLabel()) {
                        final double o2 = eval(rl.get(l));
                        //error += crossEntropy(o1, o2, 1.0f);
                        if (o1 < o2) {
                            misorderedPairs++;
                        }
                    }
                }
            }
        }
        error = 1.0 - scoreOnTrainingData;
        if (error > lastError) {
            //Neuron.learningRate *= 0.8;
            straightLoss++;
        } else {
            straightLoss = 0;
        }
        lastError = error;
    }

    @Override
    public Ranker createNew() {
        return new LambdaRank();
    }

    @Override
    public String name() {
        return "LambdaRank";
    }
}
