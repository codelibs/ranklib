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
import java.util.Random;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.utilities.MyThreadPool;
import ciir.umass.edu.utilities.WorkerThread;

/**
 * @author vdang
 */
public class FeatureHistogram {
    class Config {
        int featureIdx = -1;
        int thresholdIdx = -1;
        double S = -1;
    }

    //Parameter
    public static float samplingRate = 1;

    //Variables
    public int[] features = null;
    public float[][] thresholds = null;
    public double[][] sum = null;
    public double sumResponse = 0;
    public double sqSumResponse = 0;
    public int[][] count = null;
    public int[][] sampleToThresholdMap = null;

    //whether to re-use its parents @sum and @count instead of cleaning up the parent and re-allocate for the children.
    //@sum and @count of any intermediate tree node (except for root) can be re-used.
    private boolean reuseParent = false;

    public FeatureHistogram() {

    }

    public void construct(final DataPoint[] samples, final double[] labels, final int[][] sampleSortedIdx, final int[] features,
            final float[][] thresholds) {
        this.features = features;
        this.thresholds = thresholds;

        sumResponse = 0;
        sqSumResponse = 0;

        sum = new double[features.length][];
        count = new int[features.length][];
        sampleToThresholdMap = new int[features.length][];

        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            construct(samples, labels, sampleSortedIdx, thresholds, 0, features.length - 1);
        } else {
            p.execute(new Worker(this, samples, labels, sampleSortedIdx, thresholds), features.length);
        }
    }

    protected void construct(final DataPoint[] samples, final double[] labels, final int[][] sampleSortedIdx, final float[][] thresholds,
            final int start, final int end) {
        for (int i = start; i <= end; i++) {
            final int fid = features[i];
            //get the list of samples associated with this node (sorted in ascending order with respect to the current feature)
            final int[] idx = sampleSortedIdx[i];

            double sumLeft = 0;
            final float[] threshold = thresholds[i];
            final double[] sumLabel = new double[threshold.length];
            final int[] c = new int[threshold.length];
            final int[] stMap = new int[samples.length];

            int last = -1;
            for (int t = 0; t < threshold.length; t++) {
                int j = last + 1;
                //find the first sample that exceeds the current threshold
                for (; j < idx.length; j++) {
                    final int k = idx[j];
                    if (samples[k].getFeatureValue(fid) > threshold[t]) {
                        break;
                    }
                    sumLeft += labels[k];
                    if (i == 0) {
                        sumResponse += labels[k];
                        sqSumResponse += labels[k] * labels[k];
                    }
                    stMap[k] = t;
                }
                last = j - 1;
                sumLabel[t] = sumLeft;
                c[t] = last + 1;
            }
            sampleToThresholdMap[i] = stMap;
            sum[i] = sumLabel;
            count[i] = c;
        }
    }

    protected void update(final double[] labels) {
        sumResponse = 0;
        sqSumResponse = 0;

        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            update(labels, 0, features.length - 1);
        } else {
            p.execute(new Worker(this, labels), features.length);
        }
    }

    protected void update(final double[] labels, final int start, final int end) {
        for (int f = start; f <= end; f++) {
            Arrays.fill(sum[f], 0);
        }
        for (int k = 0; k < labels.length; k++) {
            for (int f = start; f <= end; f++) {
                final int t = sampleToThresholdMap[f][k];
                sum[f][t] += labels[k];
                if (f == 0) {
                    sumResponse += labels[k];
                    sqSumResponse += labels[k] * labels[k];
                }
                //count doesn't change, so no need to re-compute
            }
        }
        for (int f = start; f <= end; f++) {
            for (int t = 1; t < thresholds[f].length; t++) {
                sum[f][t] += sum[f][t - 1];
            }
        }
    }

    public void construct(final FeatureHistogram parent, final int[] soi, final double[] labels) {
        this.features = parent.features;
        this.thresholds = parent.thresholds;
        sumResponse = 0;
        sqSumResponse = 0;
        sum = new double[features.length][];
        count = new int[features.length][];
        sampleToThresholdMap = parent.sampleToThresholdMap;

        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            construct(parent, soi, labels, 0, features.length - 1);
        } else {
            p.execute(new Worker(this, parent, soi, labels), features.length);
        }
    }

    protected void construct(final FeatureHistogram parent, final int[] soi, final double[] labels, final int start, final int end) {
        //init
        for (int i = start; i <= end; i++) {
            final float[] threshold = thresholds[i];
            sum[i] = new double[threshold.length];
            count[i] = new int[threshold.length];
            Arrays.fill(sum[i], 0);
            Arrays.fill(count[i], 0);
        }

        //update
        for (final int k : soi) {
            for (int f = start; f <= end; f++) {
                final int t = sampleToThresholdMap[f][k];
                sum[f][t] += labels[k];
                count[f][t]++;
                if (f == 0) {
                    sumResponse += labels[k];
                    sqSumResponse += labels[k] * labels[k];
                }
            }
        }

        for (int f = start; f <= end; f++) {
            for (int t = 1; t < thresholds[f].length; t++) {
                sum[f][t] += sum[f][t - 1];
                count[f][t] += count[f][t - 1];
            }
        }
    }

    public void construct(final FeatureHistogram parent, final FeatureHistogram leftSibling, final boolean reuseParent) {
        this.reuseParent = reuseParent;
        this.features = parent.features;
        this.thresholds = parent.thresholds;
        sumResponse = parent.sumResponse - leftSibling.sumResponse;
        sqSumResponse = parent.sqSumResponse - leftSibling.sqSumResponse;

        if (reuseParent) {
            sum = parent.sum;
            count = parent.count;
        } else {
            sum = new double[features.length][];
            count = new int[features.length][];
        }
        sampleToThresholdMap = parent.sampleToThresholdMap;

        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            construct(parent, leftSibling, 0, features.length - 1);
        } else {
            p.execute(new Worker(this, parent, leftSibling), features.length);
        }
    }

    protected void construct(final FeatureHistogram parent, final FeatureHistogram leftSibling, final int start, final int end) {
        for (int f = start; f <= end; f++) {
            final float[] threshold = thresholds[f];
            if (!reuseParent) {
                sum[f] = new double[threshold.length];
                count[f] = new int[threshold.length];
            }
            for (int t = 0; t < threshold.length; t++) {
                sum[f][t] = parent.sum[f][t] - leftSibling.sum[f][t];
                count[f][t] = parent.count[f][t] - leftSibling.count[f][t];
            }
        }
    }

    protected Config findBestSplit(final int[] usedFeatures, final int minLeafSupport, final int start, final int end) {
        final Config cfg = new Config();
        final int totalCount = count[start][count[start].length - 1];
        for (int f = start; f <= end; f++) {
            final int i = usedFeatures[f];
            final float[] threshold = thresholds[i];

            for (int t = 0; t < threshold.length; t++) {
                final int countLeft = count[i][t];
                final int countRight = totalCount - countLeft;
                if (countLeft < minLeafSupport || countRight < minLeafSupport) {
                    continue;
                }

                final double sumLeft = sum[i][t];
                final double sumRight = sumResponse - sumLeft;

                final double S = sumLeft * sumLeft / countLeft + sumRight * sumRight / countRight;
                if (cfg.S < S) {
                    cfg.S = S;
                    cfg.featureIdx = i;
                    cfg.thresholdIdx = t;
                }
            }
        }
        return cfg;
    }

    public boolean findBestSplit(final Split sp, final double[] labels, final int minLeafSupport) {
        if (sp.getDeviance() >= 0.0 && sp.getDeviance() <= 0.0) {
            return false;//no need to split
        }

        int[] usedFeatures = null;//index of the features to be used for tree splitting
        if (samplingRate < 1)//need to do sub sampling (feature sampling)
        {
            final int size = (int) (samplingRate * features.length);
            usedFeatures = new int[size];
            //put all features into a pool
            final List<Integer> fpool = new ArrayList<>();
            for (int i = 0; i < features.length; i++) {
                fpool.add(i);
            }
            //do sampling, without replacement
            final Random r = new Random();
            for (int i = 0; i < size; i++) {
                final int sel = r.nextInt(fpool.size());
                usedFeatures[i] = fpool.get(sel);
                fpool.remove(sel);
            }
        } else//no sub-sampling, all features will be used
        {
            usedFeatures = new int[features.length];
            for (int i = 0; i < features.length; i++) {
                usedFeatures[i] = i;
            }
        }

        //find the best split
        Config best = new Config();
        final MyThreadPool p = MyThreadPool.getInstance();
        if (p.size() == 1) {
            best = findBestSplit(usedFeatures, minLeafSupport, 0, usedFeatures.length - 1);
        } else {
            final WorkerThread[] workers = p.execute(new Worker(this, usedFeatures, minLeafSupport), usedFeatures.length);
            for (final WorkerThread worker : workers) {
                final Worker wk = (Worker) worker;
                if (best.S < wk.cfg.S) {
                    best = wk.cfg;
                }
            }
        }

        if (best.S == -1) {
            return false;
        }

        //if(minS >= sp.getDeviance())
        //return null;

        final double[] sumLabel = sum[best.featureIdx];
        final int[] sampleCount = count[best.featureIdx];

        final double s = sumLabel[sumLabel.length - 1];
        final int c = sampleCount[sumLabel.length - 1];

        final double sumLeft = sumLabel[best.thresholdIdx];
        final int countLeft = sampleCount[best.thresholdIdx];

        final double sumRight = s - sumLeft;
        final int countRight = c - countLeft;

        final int[] left = new int[countLeft];
        final int[] right = new int[countRight];
        int l = 0;
        int r = 0;
        int k = 0;
        final int[] idx = sp.getSamples();
        for (final int element : idx) {
            k = element;
            if (sampleToThresholdMap[best.featureIdx][k] <= best.thresholdIdx) {
                left[l++] = k;
            } else {
                right[r++] = k;
            }
        }

        final FeatureHistogram lh = new FeatureHistogram();
        lh.construct(sp.hist, left, labels);
        final FeatureHistogram rh = new FeatureHistogram();
        rh.construct(sp.hist, lh, !sp.isRoot());

        final double var = sqSumResponse - sumResponse * sumResponse / idx.length;
        final double varLeft = lh.sqSumResponse - lh.sumResponse * lh.sumResponse / left.length;
        final double varRight = rh.sqSumResponse - rh.sumResponse * rh.sumResponse / right.length;

        sp.set(features[best.featureIdx], thresholds[best.featureIdx][best.thresholdIdx], var);
        sp.setLeft(new Split(left, lh, varLeft, sumLeft));
        sp.setRight(new Split(right, rh, varRight, sumRight));

        sp.clearSamples();

        return true;
    }

    class Worker extends WorkerThread {
        FeatureHistogram fh = null;
        int type = -1;

        //find best split (type == 0)
        int[] usedFeatures = null;
        int minLeafSup = -1;
        Config cfg = null;

        //update (type = 1)
        double[] labels = null;

        //construct (type = 2)
        FeatureHistogram parent = null;
        int[] soi = null;

        //construct (type = 3)
        FeatureHistogram leftSibling = null;

        //construct (type = 4)
        DataPoint[] samples;
        int[][] sampleSortedIdx;
        float[][] thresholds;

        public Worker() {
        }

        public Worker(final FeatureHistogram fh, final int[] usedFeatures, final int minLeafSup) {
            type = 0;
            this.fh = fh;
            this.usedFeatures = usedFeatures;
            this.minLeafSup = minLeafSup;
        }

        public Worker(final FeatureHistogram fh, final double[] labels) {
            type = 1;
            this.fh = fh;
            this.labels = labels;
        }

        public Worker(final FeatureHistogram fh, final FeatureHistogram parent, final int[] soi, final double[] labels) {
            type = 2;
            this.fh = fh;
            this.parent = parent;
            this.soi = soi;
            this.labels = labels;
        }

        public Worker(final FeatureHistogram fh, final FeatureHistogram parent, final FeatureHistogram leftSibling) {
            type = 3;
            this.fh = fh;
            this.parent = parent;
            this.leftSibling = leftSibling;
        }

        public Worker(final FeatureHistogram fh, final DataPoint[] samples, final double[] labels, final int[][] sampleSortedIdx,
                final float[][] thresholds) {
            type = 4;
            this.fh = fh;
            this.samples = samples;
            this.labels = labels;
            this.sampleSortedIdx = sampleSortedIdx;
            this.thresholds = thresholds;
        }

        @Override
        public void run() {
            if (type == 0) {
                cfg = fh.findBestSplit(usedFeatures, minLeafSup, start, end);
            } else if (type == 1) {
                fh.update(labels, start, end);
            } else if (type == 2) {
                fh.construct(parent, soi, labels, start, end);
            } else if (type == 3) {
                fh.construct(parent, leftSibling, start, end);
            } else if (type == 4) {
                fh.construct(samples, labels, sampleSortedIdx, thresholds, start, end);
            }
        }

        @Override
        public WorkerThread clone() {
            final Worker wk = new Worker();
            wk.fh = fh;
            wk.type = type;

            //find best split (type == 0)
            wk.usedFeatures = usedFeatures;
            wk.minLeafSup = minLeafSup;
            //wk.cfg = cfg;

            //update (type = 1)
            wk.labels = labels;

            //construct (type = 2)
            wk.parent = parent;
            wk.soi = soi;

            //construct (type = 3)
            wk.leftSibling = leftSibling;

            //construct (type = 1)
            wk.samples = samples;
            wk.sampleSortedIdx = sampleSortedIdx;
            wk.thresholds = thresholds;

            return wk;
        }
    }
}
