/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.metric;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

import ciir.umass.edu.learning.RankList;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;
import ciir.umass.edu.utilities.Sorter;

/**
 * @author vdang
 */
public class NDCGScorer extends DCGScorer {
    private static final Logger logger = Logger.getLogger(NDCGScorer.class.getName());

    protected HashMap<String, Double> idealGains = null;

    public NDCGScorer() {
        super();
        idealGains = new HashMap<>();
    }

    public NDCGScorer(final int k) {
        super(k);
        idealGains = new HashMap<>();
    }

    @Override
    public MetricScorer copy() {
        return new NDCGScorer();
    }

    @Override
    public void loadExternalRelevanceJudgment(final String qrelFile) {
        //Queries with external relevance judgment will have their cached ideal gain value overridden
        try (BufferedReader in = FileUtils.smartReader(qrelFile)) {
            String content = "";
            String lastQID = "";
            final List<Integer> rel = new ArrayList<>();
            int nQueries = 0;
            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                final String[] s = content.split(" ");
                final String qid = s[0].trim();
                final int label = (int) Math.rint(Double.parseDouble(s[3].trim()));
                if (lastQID.compareTo("") != 0 && lastQID.compareTo(qid) != 0) {
                    final int size = (rel.size() > k) ? k : rel.size();
                    final int[] r = new int[rel.size()];
                    for (int i = 0; i < rel.size(); i++) {
                        r[i] = rel.get(i);
                    }
                    final double ideal = getIdealDCG(r, size);
                    idealGains.put(lastQID, ideal);
                    rel.clear();
                    nQueries++;
                }
                lastQID = qid;
                rel.add(label);
            }
            if (rel.size() > 0) {
                final int size = (rel.size() > k) ? k : rel.size();
                final int[] r = new int[rel.size()];
                for (int i = 0; i < rel.size(); i++) {
                    r[i] = rel.get(i);
                }
                final double ideal = getIdealDCG(r, size);
                idealGains.put(lastQID, ideal);
                rel.clear();
                nQueries++;
            }
            if (logger.isLoggable(Level.INFO)) {
                logger.info("Relevance judgment file loaded. [#q=" + nQueries + "]");
            }
        } catch (final IOException ex) {
            throw RankLibError.create("Error in NDCGScorer::loadExternalRelevanceJudgment(): ", ex);
        }
    }

    /**
     * Compute NDCG at k. NDCG(k) = DCG(k) / DCG_{perfect}(k). Note that the "perfect ranking" must be computed based on the whole list,
     * not just top-k portion of the list.
     */
    @Override
    public double score(final RankList rl) {
        if (rl.size() == 0) {
            return 0;
        }

        int size = k;
        if (k > rl.size() || k <= 0) {
            size = rl.size();
        }

        final int[] rel = getRelevanceLabels(rl);

        double ideal = 0;
        final Double d = idealGains.get(rl.getID());
        if (d != null) {
            ideal = d;
        } else {
            ideal = getIdealDCG(rel, size);
            idealGains.put(rl.getID(), ideal);
        }

        if (ideal <= 0.0) {
            return 0.0;
        }

        return getDCG(rel, size) / ideal;
    }

    @Override
    public double[][] swapChange(final RankList rl) {
        final int size = (rl.size() > k) ? k : rl.size();
        //compute the ideal ndcg
        final int[] rel = getRelevanceLabels(rl);
        double ideal = 0;
        final Double d = idealGains.get(rl.getID());
        if (d != null) {
            ideal = d;
        } else {
            ideal = getIdealDCG(rel, size);
            //idealGains.put(rl.getID(), ideal);//DO *NOT* do caching here. It's not thread-safe.
        }

        final double[][] changes = new double[rl.size()][];
        for (int i = 0; i < rl.size(); i++) {
            changes[i] = new double[rl.size()];
            Arrays.fill(changes[i], 0);
        }

        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < rl.size(); j++) {
                if (ideal > 0) {
                    changes[j][i] = changes[i][j] = (discount(i) - discount(j)) * (gain(rel[i]) - gain(rel[j])) / ideal;
                }
            }
        }

        return changes;
    }

    @Override
    public String name() {
        return "NDCG@" + k;
    }

    private double getIdealDCG(final int[] rel, final int topK) {
        final int[] idx = Sorter.sort(rel, false);
        double dcg = 0;
        for (int i = 0; i < topK; i++) {
            dcg += gain(rel[idx[i]]) * discount(i);
        }
        return dcg;
    }
}
