/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.boosting;

import java.util.ArrayList;
import java.util.List;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.Sorter;

/**
 * @author vdang
 *
 * Weak rankers for AdaRank.
 */
public class WeakRanker {
    private int fid = -1;

    public WeakRanker(final int fid) {
        this.fid = fid;
    }

    public int getFID() {
        return fid;
    }

    public RankList rank(final RankList l) {
        final double[] score = new double[l.size()];
        for (int i = 0; i < l.size(); i++) {
            score[i] = l.get(i).getFeatureValue(fid);
        }
        final int[] idx = Sorter.sort(score, false);
        return new RankList(l, idx);
    }

    public List<RankList> rank(final List<RankList> l) {
        final List<RankList> ll = new ArrayList<>();
        for (int i = 0; i < l.size(); i++) {
            ll.add(rank(l.get(i)));
        }
        return ll;
    }
}
