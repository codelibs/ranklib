package ciir.umass.edu.features;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import ciir.umass.edu.utilities.RankLibError;

/*
 * Calculate feature use statistics on saved model files that make use of only a subset
 * of all defined training features.  This may be useful for persons attempting to restrict
 * a feature set by possibly eliminating features that are dropped or rarely used by
 * ranking resulting models.  Experimentation is still required to confirm absent or rarely
 * used features have little or no effect on model effectiveness.
 */

public class FeatureStats {
    private static final Logger logger = Logger.getLogger(FeatureStats.class.getName());

    private String modelName;
    private final String modelFileName;
    private final File f;

    /**
     * Define the saved model file to be used.
     *
     * @param   model file name
     */
    protected FeatureStats(final String modelFileName) {
        this.f = new File(modelFileName);
        this.modelFileName = f.getAbsolutePath();
    } //- end constructor

    private TreeMap<Integer, Integer> getFeatureWeightFeatureFrequencies(final BufferedReader br) {

        final TreeMap<Integer, Integer> tm = new TreeMap<>();

        try {
            String line = null;
            while ((line = br.readLine()) != null) {
                line = line.trim().toLowerCase();

                if (line.length() == 0) {
                    continue;
                }

                //- Not interested in model comments
                else if (line.contains("##")) {
                    continue;
                }

                //- A RankBoost model contains one line with all the feature weights so
                //  we need to split the line up into an array of feature and their weights.
                else {
                    final String[] featureLines = line.split(" ");
                    int featureFreq = 0;

                    for (final String featureLine : featureLines) {
                        final Integer featureID = Integer.valueOf(featureLine.split(":")[0]);

                        if (tm.containsKey(featureID)) {
                            featureFreq = tm.get(featureID);
                            featureFreq++;
                            tm.put(featureID, featureFreq);
                        } else {
                            tm.put(featureID, 1);
                        }
                    }
                }
            } //- end while reading

            //br.close ();
        } //- end try
        catch (final Exception ex) {
            throw RankLibError.create("Exception: " + ex.toString(), ex);
        }

        return tm;
    } //- end method getFeatureWeightFeatureFrequencies

    private TreeMap<Integer, Integer> getTreeFeatureFrequencies(final BufferedReader br) {

        final TreeMap<Integer, Integer> tm = new TreeMap<>();

        try {
            String line = null;
            while ((line = br.readLine()) != null) {
                line = line.trim().toLowerCase();

                if (line.length() == 0 || line.contains("##")) {
                    continue;
                }

                //- Generate feature frequencies
                else if (line.contains("<feature>")) {
                    final int quote1 = line.indexOf('>', 0);
                    final int quote2 = line.indexOf('<', quote1 + 1);
                    final String featureIdStr = line.substring(quote1 + 1, quote2);
                    final Integer featureID = Integer.valueOf(featureIdStr.trim());

                    if (tm.containsKey(featureID)) {
                        int featureFreq = tm.get(featureID);
                        featureFreq++;
                        tm.put(featureID, featureFreq);
                    } else {
                        tm.put(featureID, 1);
                    }
                }
            } //- end while reading
        } //- end try
        catch (final Exception ex) {
            throw RankLibError.create("Exception: " + ex.toString(), ex);
        }

        return tm;

    } //- end method getTreeFeatureFrequencies

    public void writeFeatureStats() {
        TreeMap<Integer, Integer> featureTM = null;

        try (BufferedReader br = new BufferedReader(new FileReader(f))) {

            //- Remove leading ## from model name and handle multiple word names
            final String modelLine = br.readLine().trim();
            final String[] nameparts = modelLine.split(" ");
            final int len = nameparts.length;

            if (len == 2) {
                this.modelName = nameparts[1].trim();
            } else if (len == 3) {
                this.modelName = nameparts[1].trim() + " " + nameparts[2].trim();
            }

            //- There should be a model name in the file or something is screwy.
            if (modelName == null) {
                RankLibError.create("No model name defined.  Quitting.");
            }

            //- Can't do feature statistics on models that make use of every feature as it is
            //  then difficult to say the statistics mean anything.
            if (modelName.equals("Coordinate Ascent") || modelName.equals("LambdaRank") || modelName.equals("Linear Regression")
                    || modelName.equals("ListNet") || modelName.equals("RankNet")) {
                logger.info(() -> modelName + " uses all features.  Can't do selected model statistics for this algorithm.");
                return;
            }

            //- Feature:Weight models
            else if (modelName.equals("AdaRank") || modelName.equals("RankBoost")) {
                featureTM = getFeatureWeightFeatureFrequencies(br);
            }

            //- Tree models
            else if (modelName.equals("LambdaMART") || modelName.equals("MART") || modelName.equals("Random Forests")) {
                featureTM = getTreeFeatureFrequencies(br);
            }

        } catch (final IOException ioe) {
            throw RankLibError.create("IOException on file " + modelFileName, ioe);
        }

        //- How many features?
        final int featuresUsed = featureTM.size();

        //- Print the feature frequencies and statistics
        logger.info(() -> "Model File: " + modelFileName);
        logger.info(() -> "Algorithm : " + modelName);
        logger.info(() -> "Feature frequencies : ");

        final Set<Map.Entry<Integer, Integer>> s = featureTM.entrySet();
        final DescriptiveStatistics ds = new DescriptiveStatistics();

        final Iterator<Map.Entry<Integer, Integer>> it = s.iterator();
        while (it.hasNext()) {
            final Map.Entry<Integer, Integer> e = it.next();
            final int freqID = e.getKey();
            final int freq = e.getValue();
            logger.info(() -> String.format("\tFeature[%d] : %7d", freqID, freq));
            ds.addValue(freq);
        }

        //- Print out summary statistics
        logger.info(() -> String.format("Total Features Used: %d", featuresUsed));
        logger.info(() -> String.format("Min frequency    : %10.2f", ds.getMin()));
        logger.info(() -> String.format("Max frequency    : %10.2f", ds.getMax()));
        logger.info(() -> String.format("Median frequency : %10.2f", ds.getPercentile(50)));
        logger.info(() -> String.format("Avg frequency    : %10.2f", ds.getMean()));
        logger.info(() -> String.format("Variance         : %10.2f", ds.getVariance()));
        logger.info(() -> String.format("STD              : %10.2f", ds.getStandardDeviation()));
    } //- end writeFeatureStats

} //- end class FeatureStats
