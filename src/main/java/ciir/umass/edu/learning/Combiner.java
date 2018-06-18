/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.util.logging.Logger;

import ciir.umass.edu.learning.tree.Ensemble;
import ciir.umass.edu.learning.tree.RFRanker;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;

public class Combiner {
    private static final Logger logger = Logger.getLogger(Combiner.class.getName());

    public static void main(final String[] args) {
        final Combiner c = new Combiner();
        c.combine(args[0], args[1]);
    }

    public void combine(final String directory, final String outputFile) {
        final RankerFactory rf = new RankerFactory();
        final String[] fns = FileUtils.getAllFiles(directory);
        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputFile), "ASCII"))) {
            out.write("## " + (new RFRanker()).name() + "\n");
            for (final String fn2 : fns) {
                if (fn2.indexOf(".progress") != -1) {
                    continue;
                }
                final String fn = directory + fn2;
                final RFRanker r = (RFRanker) rf.loadRankerFromFile(fn);
                final Ensemble en = r.getEnsembles()[0];
                out.write(en.toString());
            }
        } catch (final Exception e) {
            throw RankLibError.create("Error in Combiner::combine(): " + e.toString(), e);
        }
    }
}
