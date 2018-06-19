/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.KeyValuePair;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

public class LinearRegRank extends Ranker {
    private static final Logger logger = Logger.getLogger(LinearRegRank.class.getName());

    public static double lambda = 1E-10;//L2-norm regularization parameter

    //Local variables
    protected double[] weight = null;

    public LinearRegRank() {
    }

    public LinearRegRank(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    @Override
    public void init() {
        logger.info(() -> "Initializing...");
    }

    @Override
    public void learn() {
        logger.info(() -> "Training starts...");
        logger.info(() -> "Learning the least square model... ");

        //closed form solution: beta = ((xTx - lambda*I)^(-1)) * (xTy)
        //where x is an n-by-f matrix (n=#data-points, f=#features), y is an n-element vector of relevance labels
        /*int nSample = 0;
        for(int i=0;i<samples.size();i++)
        	nSample += samples.get(i).size();*/
        int nVar = 0;
        for (final RankList rl : samples) {
            final int c = rl.getFeatureCount();
            if (c > nVar) {
                nVar = c;
            }
        }

        final double[][] xTx = new double[nVar][];
        for (int i = 0; i < nVar; i++) {
            xTx[i] = new double[nVar];
            Arrays.fill(xTx[i], 0.0);
        }
        final double[] xTy = new double[nVar];
        Arrays.fill(xTy, 0.0);

        for (int s = 0; s < samples.size(); s++) {
            final RankList rl = samples.get(s);
            for (int i = 0; i < rl.size(); i++) {
                xTy[nVar - 1] += rl.get(i).getLabel();
                for (int j = 0; j < nVar - 1; j++) {
                    xTy[j] += rl.get(i).getFeatureValue(j + 1) * rl.get(i).getLabel();
                    for (int k = 0; k < nVar; k++) {
                        final double t = (k < nVar - 1) ? rl.get(i).getFeatureValue(k + 1) : 1f;
                        xTx[j][k] += rl.get(i).getFeatureValue(j + 1) * t;
                    }
                }
                for (int k = 0; k < nVar - 1; k++) {
                    xTx[nVar - 1][k] += rl.get(i).getFeatureValue(k + 1);
                }
                xTx[nVar - 1][nVar - 1] += 1f;
            }
        }
        if (lambda != 0.0)//regularized
        {
            for (int i = 0; i < xTx.length; i++) {
                xTx[i][i] += lambda;
            }
        }
        weight = solve(xTx, xTy);

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
        double score = weight[weight.length - 1];
        for (int i = 0; i < features.length; i++) {
            score += weight[i] * p.getFeatureValue(features[i]);
        }
        return score;
    }

    @Override
    public Ranker createNew() {
        return new LinearRegRank();
    }

    @Override
    public String toString() {
        String output = "0:" + weight[0] + " ";
        for (int i = 0; i < features.length; i++) {
            output += features[i] + ":" + weight[i] + ((i == weight.length - 1) ? "" : " ");
        }
        return output;
    }

    @Override
    public String model() {
        String output = "## " + name() + "\n";
        output += "## Lambda = " + lambda + "\n";
        output += toString();
        return output;
    }

    @Override
    public void loadFromString(final String fullText) {
        try (final BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            String content = "";

            KeyValuePair kvp = null;
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                if (content.indexOf("##") == 0) {
                    continue;
                }
                kvp = new KeyValuePair(content);
                break;
            }

            assert (kvp != null);
            final List<String> keys = kvp.keys();
            final List<String> values = kvp.values();
            weight = new double[keys.size()];
            features = new int[keys.size() - 1];//weight = <weight for each feature, constant>
            int idx = 0;
            for (int i = 0; i < keys.size(); i++) {
                final int fid = Integer.parseInt(keys.get(i));
                if (fid > 0) {
                    features[idx] = fid;
                    weight[idx] = Double.parseDouble(values.get(i));
                    idx++;
                } else {
                    weight[weight.length - 1] = Double.parseDouble(values.get(i));
                }
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in LinearRegRank::load(): ", ex);
        }
    }

    @Override
    public void printParameters() {
        logger.info(() -> "L2-norm regularization: lambda = " + lambda);
    }

    @Override
    public String name() {
        return "Linear Regression";
    }

    /**
     * Solve a system of linear equations Ax=B, in which A has to be a square matrix with the same length as B
     * @param A
     * @param B
     * @return x
     */
    protected double[] solve(final double[][] A, final double[] B) {
        if (A.length == 0 || B.length == 0) {
            throw RankLibError.create("Error: some of the input arrays is empty.");
        }
        if (A[0].length == 0) {
            throw RankLibError.create("Error: some of the input arrays is empty.");
        }
        if (A.length != B.length) {
            throw RankLibError.create("Error: Solving Ax=B: A and B have different dimension.");
        }

        //init
        final double[][] a = new double[A.length][];
        final double[] b = new double[B.length];
        System.arraycopy(B, 0, b, 0, B.length);
        for (int i = 0; i < a.length; i++) {
            a[i] = new double[A[i].length];
            if (i > 0) {
                if (a[i].length != a[i - 1].length) {
                    throw RankLibError.create("Error: Solving Ax=B: A is NOT a square matrix.");
                }
            }
            System.arraycopy(A[i], 0, a[i], 0, A[i].length);
        }
        //apply the gaussian elimination process to convert the matrix A to upper triangular form
        double pivot = 0.0;
        double multiplier = 0.0;
        for (int j = 0; j < b.length - 1; j++)//loop through all columns of the matrix A
        {
            pivot = a[j][j];
            for (int i = j + 1; i < b.length; i++)//loop through all remaining rows
            {
                multiplier = a[i][j] / pivot;
                //i-th row = i-th row - (multiplier * j-th row)
                for (int k = j + 1; k < b.length; k++) {
                    a[i][k] -= a[j][k] * multiplier;
                }
                b[i] -= b[j] * multiplier;
            }
        }
        //a*x=b
        //a is now an upper triangular matrix, now the solution x can be obtained with elementary linear algebra
        final double[] x = new double[b.length];
        final int n = b.length;
        x[n - 1] = b[n - 1] / a[n - 1][n - 1];
        for (int i = n - 2; i >= 0; i--)//walk back up to the first row -- we only need to care about the right to the diagonal
        {
            double val = b[i];
            for (int j = i + 1; j < n; j++) {
                val -= a[i][j] * x[j];
            }
            x[i] = val / a[i][i];
        }

        return x;
    }
}
