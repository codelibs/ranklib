package ciir.umass.edu.stats;

import java.util.logging.Logger;

import ciir.umass.edu.utilities.RankLibError;

public class BasicStats {
    private static final Logger logger = Logger.getLogger(BasicStats.class.getName());

    public static double mean(final double[] values) {
        double mean = 0.0;
        if (values.length == 0) {
            RankLibError.create("Error in BasicStats::mean(): Empty input array.");
        }
        for (final double value : values) {
            mean += value;
        }
        return mean / values.length;
    }
}
