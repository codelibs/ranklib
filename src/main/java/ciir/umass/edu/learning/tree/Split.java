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
import java.util.List;

import ciir.umass.edu.learning.DataPoint;

/**
 *
 * @author vdang
 *
 */
public class Split {
    //Key attributes of a split (tree node)
    private int featureID = -1;
    private float threshold = 0F;
    private double avgLabel = 0.0F;

    //Intermediate variables (ONLY used during learning)
    //*DO NOT* attempt to access them once the training is done
    private boolean isRoot = false;
    private double sumLabel = 0.0;
    private double sqSumLabel = 0.0;
    private Split left = null;
    private Split right = null;
    private double deviance = 0F;//mean squared error "S"
    private int[][] sortedSampleIDs = null;
    public int[] samples = null;
    public FeatureHistogram hist = null;

    public Split() {

    }

    public Split(final int featureID, final float threshold, final double deviance) {
        this.featureID = featureID;
        this.threshold = threshold;
        this.deviance = deviance;
    }

    public Split(final int[][] sortedSampleIDs, final double deviance, final double sumLabel, final double sqSumLabel) {
        this.sortedSampleIDs = sortedSampleIDs;
        this.deviance = deviance;
        this.sumLabel = sumLabel;
        this.sqSumLabel = sqSumLabel;
        avgLabel = sumLabel / sortedSampleIDs[0].length;
    }

    public Split(final int[] samples, final FeatureHistogram hist, final double deviance, final double sumLabel) {
        this.samples = samples;
        this.hist = hist;
        this.deviance = deviance;
        this.sumLabel = sumLabel;
        avgLabel = sumLabel / samples.length;
    }

    public void set(final int featureID, final float threshold, final double deviance) {
        this.featureID = featureID;
        this.threshold = threshold;
        this.deviance = deviance;
    }

    public void setLeft(final Split s) {
        left = s;
    }

    public void setRight(final Split s) {
        right = s;
    }

    public void setOutput(final float output) {
        avgLabel = output;
    }

    public Split getLeft() {
        return left;
    }

    public Split getRight() {
        return right;
    }

    public double getDeviance() {
        return deviance;
    }

    public double getOutput() {
        return avgLabel;
    }

    public List<Split> leaves() {
        final List<Split> list = new ArrayList<>();
        leaves(list);
        return list;
    }

    private void leaves(final List<Split> leaves) {
        if (featureID == -1) {
            leaves.add(this);
        } else {
            left.leaves(leaves);
            right.leaves(leaves);
        }
    }

    public double eval(final DataPoint dp) {
        Split n = this;
        while (n.featureID != -1) {
            if (dp.getFeatureValue(n.featureID) <= n.threshold) {
                n = n.left;
            } else {
                n = n.right;
            }
        }
        return n.avgLabel;
    }

    @Override
    public String toString() {
        return toString("");
    }

    public String toString(final String indent) {
        String strOutput = indent + "<split>" + "\n";
        strOutput += getString(indent + "\t");
        strOutput += indent + "</split>" + "\n";
        return strOutput;
    }

    public String getString(final String indent) {
        String strOutput = "";
        if (featureID == -1) {
            strOutput += indent + "<output> " + avgLabel + " </output>" + "\n";
        } else {
            strOutput += indent + "<feature> " + featureID + " </feature>" + "\n";
            strOutput += indent + "<threshold> " + threshold + " </threshold>" + "\n";
            strOutput += indent + "<split pos=\"left\">" + "\n";
            strOutput += left.getString(indent + "\t");
            strOutput += indent + "</split>" + "\n";
            strOutput += indent + "<split pos=\"right\">" + "\n";
            strOutput += right.getString(indent + "\t");
            strOutput += indent + "</split>" + "\n";
        }
        return strOutput;
    }

    //Internal functions(ONLY used during learning)
    //*DO NOT* attempt to call them once the training is done
    public boolean split(final double[] trainingLabels, final int minLeafSupport) {
        return hist.findBestSplit(this, trainingLabels, minLeafSupport);
    }

    public int[] getSamples() {
        if (sortedSampleIDs != null) {
            return sortedSampleIDs[0];
        }
        return samples;
    }

    public int[][] getSampleSortedIndex() {
        return sortedSampleIDs;
    }

    public double getSumLabel() {
        return sumLabel;
    }

    public double getSqSumLabel() {
        return sqSumLabel;
    }

    public void clearSamples() {
        sortedSampleIDs = null;
        samples = null;
        hist = null;
    }

    public void setRoot(final boolean isRoot) {
        this.isRoot = isRoot;
    }

    public boolean isRoot() {
        return isRoot;
    }
}