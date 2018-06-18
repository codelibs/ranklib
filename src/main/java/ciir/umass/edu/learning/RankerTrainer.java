/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import java.util.List;
import java.util.logging.Logger;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 *
 * This class is for users who want to use this library programmatically. It provides trained rankers of different types with respect to user-specified parameters.
 */
public class RankerTrainer {
    private static final Logger logger = Logger.getLogger(RankerTrainer.class.getName());

    protected RankerFactory rf = new RankerFactory();
    protected double trainingTime = 0;

    public Ranker train(final RankerType type, final List<RankList> train, final int[] features, final MetricScorer scorer) {
        final Ranker ranker = rf.createRanker(type, train, features, scorer);
        final long start = System.nanoTime();
        ranker.init();
        ranker.learn();
        trainingTime = System.nanoTime() - start;
        //printTrainingTime();
        return ranker;
    }

    public Ranker train(final RankerType type, final List<RankList> train, final List<RankList> validation, final int[] features,
            final MetricScorer scorer) {
        final Ranker ranker = rf.createRanker(type, train, features, scorer);
        ranker.setValidationSet(validation);
        final long start = System.nanoTime();
        ranker.init();
        ranker.learn();
        trainingTime = System.nanoTime() - start;
        //printTrainingTime();
        return ranker;
    }

    public double getTrainingTime() {
        return trainingTime;
    }

    public void printTrainingTime() {
        logger.info(() -> "Training time: " + SimpleMath.round((trainingTime) / 1e9, 2) + " seconds");
    }
}
