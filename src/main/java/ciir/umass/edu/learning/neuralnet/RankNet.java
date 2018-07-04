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
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.learning.Ranker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.SimpleMath;

/**
 * @author vdang
 *
 *  This class implements RankNet.
 *  C.J.C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton and G. Hullender. Learning to rank using gradient descent.
 *  In Proc. of ICML, pages 89-96, 2005.
 */
public class RankNet extends Ranker {
    private static final Logger logger = Logger.getLogger(RankNet.class.getName());

    //Parameters
    public static int nIteration = 100;
    public static int nHiddenLayer = 1;
    public static int nHiddenNodePerLayer = 10;
    public static double learningRate = 0.00005;

    //Variables
    protected List<Layer> layers = new ArrayList<>();
    protected Layer inputLayer = null;
    protected Layer outputLayer = null;

    //to store the best model on validation data (if specified)
    protected List<List<Double>> bestModelOnValidation = new ArrayList<>();

    protected int totalPairs = 0;
    protected int misorderedPairs = 0;
    protected double error = 0.0;
    protected double lastError = Double.MAX_VALUE;
    protected int straightLoss = 0;

    public RankNet() {

    }

    public RankNet(final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        super(samples, features, scorer);
    }

    /**
     * Setting up the Neural Network
     */
    protected void setInputOutput(final int nInput, final int nOutput) {
        inputLayer = new Layer(nInput + 1);//plus the "bias" (output threshold)
        outputLayer = new Layer(nOutput);
        layers.clear();
        layers.add(inputLayer);
        layers.add(outputLayer);
    }

    protected void setInputOutput(final int nInput, final int nOutput, final int nType) {
        inputLayer = new Layer(nInput + 1, nType);//plus the "bias" (output threshold)
        outputLayer = new Layer(nOutput, nType);
        layers.clear();
        layers.add(inputLayer);
        layers.add(outputLayer);
    }

    protected void addHiddenLayer(final int size) {
        layers.add(layers.size() - 1, new Layer(size));
    }

    protected void wire() {
        //wire the input layer to the first hidden layer
        for (int i = 0; i < inputLayer.size() - 1; i++) {
            for (int j = 0; j < layers.get(1).size(); j++) {
                connect(0, i, 1, j);
            }
        }

        //wire one layer to the next, starting at layer 1 (the first hidden layer)
        for (int i = 1; i < layers.size() - 1; i++) {
            for (int j = 0; j < layers.get(i).size(); j++) {
                for (int k = 0; k < layers.get(i + 1).size(); k++) {
                    connect(i, j, i + 1, k);
                }
            }
        }

        //wire the "bias" neuron to all others (in all layers)
        for (int i = 1; i < layers.size(); i++) {
            for (int j = 0; j < layers.get(i).size(); j++) {
                connect(0, inputLayer.size() - 1, i, j);
            }
        }
    }

    protected void connect(final int sourceLayer, final int sourceNeuron, final int targetLayer, final int targetNeuron) {
        new Synapse(layers.get(sourceLayer).get(sourceNeuron), layers.get(targetLayer).get(targetNeuron));
    }

    /**
     *  Auxiliary functions for pair-wise preference network learning.
     */
    protected void addInput(final DataPoint p) {
        for (int k = 0; k < inputLayer.size() - 1; k++) {
            inputLayer.get(k).addOutput(p.getFeatureValue(features[k]));
        }
        //  and now the bias node with a fix "1.0"
        inputLayer.get(inputLayer.size() - 1).addOutput(1.0f);
    }

    protected void propagate(final int i) {
        for (int k = 1; k < layers.size(); k++) {
            layers.get(k).computeOutput(i);
        }
    }

    protected int[][] batchFeedForward(final RankList rl) {
        final int[][] pairMap = new int[rl.size()][];
        for (int i = 0; i < rl.size(); i++) {
            addInput(rl.get(i));
            propagate(i);

            int count = 0;
            for (int j = 0; j < rl.size(); j++) {
                if (rl.get(i).getLabel() > rl.get(j).getLabel()) {
                    count++;
                }
            }

            pairMap[i] = new int[count];
            int k = 0;
            for (int j = 0; j < rl.size(); j++) {
                if (rl.get(i).getLabel() > rl.get(j).getLabel()) {
                    pairMap[i][k++] = j;
                }
            }
        }
        return pairMap;
    }

    protected void batchBackPropagate(final int[][] pairMap, final float[][] pairWeight) {
        for (int i = 0; i < pairMap.length; i++) {
            //back-propagate
            final PropParameter p = new PropParameter(i, pairMap);
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

    protected void clearNeuronOutputs() {
        for (int k = 0; k < layers.size(); k++) {
            layers.get(k).clearOutputs();
        }
    }

    protected float[][] computePairWeight(final int[][] pairMap, final RankList rl) {
        return null;
    }

    protected RankList internalReorder(final RankList rl) {
        return rl;
    }

    /**
     * Model validation
     */
    protected void saveBestModelOnValidation() {
        for (int i = 0; i < layers.size() - 1; i++)//loop through all layers
        {
            final List<Double> l = bestModelOnValidation.get(i);
            l.clear();
            for (int j = 0; j < layers.get(i).size(); j++)//loop through all neurons on in the current layer
            {
                final Neuron n = layers.get(i).get(j);
                for (int k = 0; k < n.getOutLinks().size(); k++) {
                    l.add(n.getOutLinks().get(k).getWeight());
                }
            }
        }
    }

    protected void restoreBestModelOnValidation() {
        try {
            for (int i = 0; i < layers.size() - 1; i++)//loop through all layers
            {
                final List<Double> l = bestModelOnValidation.get(i);
                int c = 0;
                for (int j = 0; j < layers.get(i).size(); j++)//loop through all neurons on in the current layer
                {
                    final Neuron n = layers.get(i).get(j);
                    for (int k = 0; k < n.getOutLinks().size(); k++) {
                        n.getOutLinks().get(k).setWeight(l.get(c++));
                    }
                }
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in NeuralNetwork.restoreBestModelOnValidation(): ", ex);
        }
    }

    protected double crossEntropy(final double o1, final double o2, final double targetValue) {
        final double oij = o1 - o2;
        return -targetValue * oij + SimpleMath.logBase2(1 + Math.exp(oij));
    }

    protected void estimateLoss() {
        misorderedPairs = 0;
        error = 0.0;
        for (int j = 0; j < samples.size(); j++) {
            final RankList rl = samples.get(j);
            for (int k = 0; k < rl.size() - 1; k++) {
                final double o1 = eval(rl.get(k));
                for (int l = k + 1; l < rl.size(); l++) {
                    if (rl.get(k).getLabel() > rl.get(l).getLabel()) {
                        final double o2 = eval(rl.get(l));
                        error += crossEntropy(o1, o2, 1.0f);
                        if (o1 < o2) {
                            misorderedPairs++;
                        }
                    }
                }
            }
        }
        error = SimpleMath.round(error / totalPairs, 4);

        //Neuron.learningRate *= 0.8;
        lastError = error;
    }

    /**
     * Main public functions
     */
    @Override
    public void init() {
        logger.info(() -> "Initializing... ");

        //Set up the network
        setInputOutput(features.length, 1);
        for (int i = 0; i < nHiddenLayer; i++) {
            addHiddenLayer(nHiddenNodePerLayer);
        }
        wire();

        totalPairs = 0;
        for (int i = 0; i < samples.size(); i++) {
            final RankList rl = samples.get(i).getCorrectRanking();
            for (int j = 0; j < rl.size() - 1; j++) {
                for (int k = j + 1; k < rl.size(); k++) {
                    if (rl.get(j).getLabel() > rl.get(k).getLabel()) {
                        totalPairs++;
                    }
                }
            }
        }

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
        printLogLn(new int[] { 7, 14, 9, 9 }, new String[] { "#epoch", "% mis-ordered", scorer.name() + "-T", scorer.name() + "-V" });
        printLogLn(new int[] { 7, 14, 9, 9 }, new String[] { " ", "  pairs", " ", " " });

        for (int i = 1; i <= nIteration; i++) {
            for (int j = 0; j < samples.size(); j++) {
                final RankList rl = internalReorder(samples.get(j));
                final int[][] pairMap = batchFeedForward(rl);
                final float[][] pairWeight = computePairWeight(pairMap, rl);
                batchBackPropagate(pairMap, pairWeight);
                clearNeuronOutputs();
            }

            scoreOnTrainingData = scorer.score(rank(samples));
            estimateLoss();
            printLog(new int[] { 7, 14 }, new String[] { Integer.toString(i), Double.toString(SimpleMath.round(((double) misorderedPairs) / totalPairs, 4)) });
            if (i % 1 == 0) {
                printLog(new int[] { 9 }, new String[] { Double.toString(SimpleMath.round(scoreOnTrainingData, 4)) });
                if (validationSamples != null) {
                    final double score = scorer.score(rank(validationSamples));
                    if (score > bestScoreOnValidationData) {
                        bestScoreOnValidationData = score;
                        saveBestModelOnValidation();
                    }
                    printLog(new int[] { 9 }, new String[] { Double.toString(SimpleMath.round(score, 4)) });
                }
            }
            flushLog();
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
        //feed input
        for (int k = 0; k < inputLayer.size() - 1; k++) {
            inputLayer.get(k).setOutput(p.getFeatureValue(features[k]));
        }
        //and now the bias node with a fix "1.0"
        inputLayer.get(inputLayer.size() - 1).setOutput(1.0f);
        //propagate
        for (int k = 1; k < layers.size(); k++) {
            layers.get(k).computeOutput();
        }
        return outputLayer.get(0).getOutput();
    }

    @Override
    public Ranker createNew() {
        return new RankNet();
    }

    @Override
    public String toString() {
        final StringBuilder output = new StringBuilder();
        for (int i = 0; i < layers.size() - 1; i++)//loop through all layers
        {
            for (int j = 0; j < layers.get(i).size(); j++)//loop through all neurons on in the current layer
            {
                output.append(i + " " + j + " ");
                final Neuron n = layers.get(i).get(j);
                for (int k = 0; k < n.getOutLinks().size(); k++) {
                    output.append(n.getOutLinks().get(k).getWeight() + ((k == n.getOutLinks().size() - 1) ? "" : " "));
                }
                output.append('\n');
            }
        }
        return output.toString();
    }

    @Override
    public String model() {
        final StringBuilder output = new StringBuilder();
        output.append("## " + name() + "\n");
        output.append("## Epochs = " + nIteration + "\n");
        output.append("## No. of features = " + features.length + "\n");
        output.append("## No. of hidden layers = " + (layers.size() - 2) + "\n");
        for (int i = 1; i < layers.size() - 1; i++) {
            output.append("## Layer " + i + ": " + layers.get(i).size() + " neurons\n");
        }

        //print used features
        for (int i = 0; i < features.length; i++) {
            output.append(features[i] + ((i == features.length - 1) ? "" : " "));
        }
        output.append('\n');
        //print network information
        output.append(layers.size() - 2 + "\n");//[# hidden layers]
        for (int i = 1; i < layers.size() - 1; i++) {
            output.append(layers.get(i).size() + "\n");//[#neurons]
        }
        //print learned weights
        output.append(toString());
        return output.toString();
    }

    @Override
    public void loadFromString(final String fullText) {
        try (final BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            String content = null;

            final List<String> l = new ArrayList<>();
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0 || content.indexOf("##") == 0) {
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
            final int nhl = Integer.parseInt(l.get(1));
            final int[] nn = new int[nhl];
            //the next @nHiddenLayer lines contain the number of neurons in each layer
            int i = 2;
            for (; i < 2 + nhl; i++) {
                nn[i - 2] = Integer.parseInt(l.get(i));
            }
            //create the network
            setInputOutput(features.length, 1);
            for (int j = 0; j < nhl; j++) {
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
            throw RankLibError.create("Error in RankNet::load(): ", ex);
        }
    }

    @Override
    public void printParameters() {
        logger.info(() -> "No. of epochs: " + nIteration);
        logger.info(() -> "No. of hidden layers: " + nHiddenLayer);
        logger.info(() -> "No. of hidden nodes per layer: " + nHiddenNodePerLayer);
        logger.info(() -> "Learning rate: " + learningRate);
    }

    @Override
    public String name() {
        return "RankNet";
    }

    /**
     * FOR DEBUGGING PURPOSE ONLY
     */
    protected void printNetworkConfig() {
        if (logger.isLoggable(Level.INFO)) {
            for (int i = 1; i < layers.size(); i++) {
                logger.info("Layer-" + (i + 1));
                for (int j = 0; j < layers.get(i).size(); j++) {
                    final Neuron n = layers.get(i).get(j);
                    logger.info("Neuron-" + (j + 1) + ": " + n.getInLinks().size() + " inputs\t");
                    for (int k = 0; k < n.getInLinks().size(); k++) {
                        logger.info(n.getInLinks().get(k).getWeight() + "\t");
                    }
                }
            }
        }
    }

    protected void printWeightVector() {
        if (logger.isLoggable(Level.INFO)) {
            final StringBuilder buf = new StringBuilder();
            for (int j = 0; j < outputLayer.get(0).getInLinks().size(); j++) {
                buf.append(outputLayer.get(0).getInLinks().get(j).getWeight()).append(' ');
            }
            logger.info(buf.toString());
        }
    }
}
