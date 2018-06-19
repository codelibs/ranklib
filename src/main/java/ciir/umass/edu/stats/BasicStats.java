package ciir.umass.edu.stats;

import ciir.umass.edu.utilities.RankLibError;

public class BasicStats {

    public static double mean(final double[] values) {
        double mean = 0.0;
        if (values.length == 0) {
            throw RankLibError.create("Error in BasicStats::mean(): Empty input array.");
        }
        for (final double value : values) {
            mean += value;
        }
        return mean / values.length;
    }
}
