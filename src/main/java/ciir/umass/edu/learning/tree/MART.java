/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.tree;

import java.util.List;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;

/**
 * @author vdang
 *
 *  This class implements MART for (point-wise) ranking:
 *  J.H. Friedman. Greedy function approximation: A gradient boosting machine.
 *  Technical Report, IMS Reitz Lecture, Stanford, 1999; see also Annals of Statistics, 2001.
 */
public class MART extends LambdaMART {
    //Parameters
    //Inherits *ALL* parameters from LambdaMART

    public MART() {
    }

    public MART(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    @Override
    public Ranker createNew() {
        return new MART();
    }

    @Override
    public String name() {
        return "MART";
    }

    @Override
    protected void computePseudoResponses() {
        for (int i = 0; i < martSamples.length; i++) {
            pseudoResponses[i] = martSamples[i].getLabel() - modelScores[i];
        }
    }

    @Override
    protected void updateTreeOutput(final RegressionTree rt) {
        final List<Split> leaves = rt.leaves();
        for (int i = 0; i < leaves.size(); i++) {
            float s1 = 0.0F;
            final Split s = leaves.get(i);
            final int[] idx = s.getSamples();
            for (final int k : idx) {
                s1 += pseudoResponses[k];
            }
            s.setOutput(s1 / idx.length);
        }
    }
}
