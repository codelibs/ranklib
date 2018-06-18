/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.neuralnet;

import java.io.BufferedReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

public class ListNet extends RankNet {
    private static final Logger logger = Logger.getLogger(ListNet.class.getName());

    //Parameters
    public static int nIteration = 1500;
    public static double learningRate = 0.00001;
    public static int nHiddenLayer = 0;//FIXED, it doesn't work with hidden layer

    public ListNet() {
    }

    public ListNet(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    protected float[] feedForward(final RankList rl) {
        final float[] labels = new float[rl.size()];
        for (int i = 0; i < rl.size(); i++) {
            addInput(rl.get(i));
            propagate(i);
            labels[i] = rl.get(i).getLabel();
        }
        return labels;
    }

    protected void backPropagate(final float[] labels) {
        //back-propagate
        final PropParameter p = new PropParameter(labels);
        outputLayer.computeDelta(p);//starting at the output layer

        //weight update
        outputLayer.updateWeight(p);
    }

    @Override
    protected void estimateLoss() {
        error = 0.0;
        double sumLabelExp = 0;
        double sumScoreExp = 0;
        for (int i = 0; i < samples.size(); i++) {
            final RankList rl = samples.get(i);
            final double[] scores = new double[rl.size()];
            double err = 0;
            for (int j = 0; j < rl.size(); j++) {
                scores[j] = eval(rl.get(j));
                sumLabelExp += Math.exp(rl.get(j).getLabel());
                sumScoreExp += Math.exp(scores[j]);
            }
            for (int j = 0; j < rl.size(); j++) {
                final double p1 = Math.exp(rl.get(j).getLabel()) / sumLabelExp;
                final double p2 = (Math.exp(scores[j]) / sumScoreExp);
                err += -p1 * SimpleMath.logBase2(p2);
            }
            error += err / rl.size();
        }
        //if(error > lastError && Neuron.learningRate > 0.0000001)
        //Neuron.learningRate *= 0.9;
        lastError = error;
    }

    @Override
    public void init() {
        logger.info(() -> "Initializing... ");

        //Set up the network
        setInputOutput(features.length, 1, 1);
        wire();

        if (validationSamples != null) {
            for (int i = 0; i < layers.size(); i++) {
                bestModelOnValidation.add(new ArrayList<Double>());
            }
        }

        Neuron.learningRate = learningRate;
    }

    @Override
    public void learn() {
        logger.info(() -> "Training starts...");
        printLogLn(new int[] { 7, 14, 9, 9 }, new String[] { "#epoch", "C.E. Loss", scorer.name() + "-T", scorer.name() + "-V" });

        for (int i = 1; i <= nIteration; i++) {
            for (int j = 0; j < samples.size(); j++) {
                final float[] labels = feedForward(samples.get(j));
                backPropagate(labels);
                clearNeuronOutputs();
            }
            //estimateLoss();
            printLog(new int[] { 7, 14 }, new String[] { i + "", SimpleMath.round(error, 6) + "" });
            if (i % 1 == 0) {
                scoreOnTrainingData = scorer.score(rank(samples));
                printLog(new int[] { 9 }, new String[] { SimpleMath.round(scoreOnTrainingData, 4) + "" });
                if (validationSamples != null) {
                    final double score = scorer.score(rank(validationSamples));
                    if (score > bestScoreOnValidationData) {
                        bestScoreOnValidationData = score;
                        saveBestModelOnValidation();
                    }
                    printLog(new int[] { 9 }, new String[] { SimpleMath.round(score, 4) + "" });
                }
            }
        }

        //if validation data is specified ==> best model on this data has been saved
        //we now restore the current model to that best model
        if (validationSamples != null) {
            restoreBestModelOnValidation();
        }

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
        return super.eval(p);
    }

    @Override
    public Ranker createNew() {
        return new ListNet();
    }

    @Override
    public String toString() {
        return super.toString();
    }

    @Override
    public String model() {
        String output = "## " + name() + "\n";
        output += "## Epochs = " + nIteration + "\n";
        output += "## No. of features = " + features.length + "\n";

        //print used features
        for (int i = 0; i < features.length; i++) {
            output += features[i] + ((i == features.length - 1) ? "" : " ");
        }
        output += "\n";
        //print network information
        output += "0\n";//[# hidden layers, *ALWAYS* 0 since we're using linear net]
        //print learned weights
        output += toString();
        return output;
    }

    @Override
    public void loadFromString(final String fullText) {
        try (final BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            String content = "";

            final List<String> l = new ArrayList<>();
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                if (content.indexOf("##") == 0) {
                    continue;
                }
                l.add(content);
            }
            //load the network
            //the first line contains features information
            final String[] tmp = l.get(0).split(" ");
            features = new int[tmp.length];
            for (int i = 0; i < tmp.length; i++) {
                features[i] = Integer.parseInt(tmp[i]);
            }
            //the 2nd line is a scalar indicating the number of hidden layers
            final int nHiddenLayer = Integer.parseInt(l.get(1));
            final int[] nn = new int[nHiddenLayer];
            //the next @nHiddenLayer lines contain the number of neurons in each layer
            int i = 2;
            for (; i < 2 + nHiddenLayer; i++) {
                nn[i - 2] = Integer.parseInt(l.get(i));
            }
            //create the network
            setInputOutput(features.length, 1);
            for (int j = 0; j < nHiddenLayer; j++) {
                addHiddenLayer(nn[j]);
            }
            wire();
            //fill in weights
            for (; i < l.size(); i++)//loop through all layers
            {
                final String[] s = l.get(i).split(" ");
                final int iLayer = Integer.parseInt(s[0]);//which layer?
                final int iNeuron = Integer.parseInt(s[1]);//which neuron?
                final Neuron n = layers.get(iLayer).get(iNeuron);
                for (int k = 0; k < n.getOutLinks().size(); k++) {
                    n.getOutLinks().get(k).setWeight(Double.parseDouble(s[k + 2]));
                }
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in ListNet::load(): ", ex);
        }
    }

    @Override
    public void printParameters() {
        logger.info(() -> "No. of epochs: " + nIteration);
        logger.info(() -> "Learning rate: " + learningRate);
    }

    @Override
    public String name() {
        return "ListNet";
    }
}
