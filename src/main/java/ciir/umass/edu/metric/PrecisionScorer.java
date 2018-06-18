/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import java.util.Arrays;

import ciir.umass.edu.learning.RankList;

/**
 * @author vdang
 */
public class PrecisionScorer extends MetricScorer {

    public PrecisionScorer() {
        this.k = 10;
    }

    public PrecisionScorer(final int k) {
        this.k = k;
    }

    @Override
    public double score(final RankList rl) {
        int count = 0;

        int size = k;
        if (k > rl.size() || k <= 0) {
            size = rl.size();
        }

        for (int i = 0; i < size; i++) {
            if (rl.get(i).getLabel() > 0.0) {
                count++;
            }
        }
        return ((double) count) / size;
    }

    @Override
    public MetricScorer copy() {
        return new PrecisionScorer();
    }

    @Override
    public String name() {
        return "P@" + k;
    }

    @Override
    public double[][] swapChange(final RankList rl) {
        final int size = (rl.size() > k) ? k : rl.size();
        /*int relCount = 0;
        for(int i=0;i<size;i++)
        	if(rl.get(i).getLabel() > 0.0)//relevant
        		relCount++;*/

        final double[][] changes = new double[rl.size()][];
        for (int i = 0; i < rl.size(); i++) {
            changes[i] = new double[rl.size()];
            Arrays.fill(changes[i], 0);
        }

        for (int i = 0; i < size; i++) {
            for (int j = size; j < rl.size(); j++) {
                final int c = getBinaryRelevance(rl.get(j).getLabel()) - getBinaryRelevance(rl.get(i).getLabel());
                changes[i][j] = changes[j][i] = ((float) c) / size;
            }
        }
        return changes;
    }

    private int getBinaryRelevance(final float label) {
        if (label > 0.0) {
            return 1;
        }
        return 0;
    }
}
