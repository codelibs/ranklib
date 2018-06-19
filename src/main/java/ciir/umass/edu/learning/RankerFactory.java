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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import ciir.umass.edu.learning.boosting.AdaRank;
import ciir.umass.edu.learning.boosting.RankBoost;
import ciir.umass.edu.learning.neuralnet.LambdaRank;
import ciir.umass.edu.learning.neuralnet.ListNet;
import ciir.umass.edu.learning.neuralnet.RankNet;
import ciir.umass.edu.learning.tree.LambdaMART;
import ciir.umass.edu.learning.tree.MART;
import ciir.umass.edu.learning.tree.RFRanker;
import ciir.umass.edu.metric.MetricScorer;
import ciir.umass.edu.utilities.FileUtils;
import ciir.umass.edu.utilities.RankLibError;

/**
 * @author vdang
 *
 * This class implements the Ranker factory. All ranking algorithms implemented have to be recognized in this class.
 */
public class RankerFactory {
    private static final Logger logger = Logger.getLogger(RankerFactory.class.getName());

    protected Ranker[] rFactory = new Ranker[] { new MART(), new RankBoost(), new RankNet(), new AdaRank(), new CoorAscent(),
            new LambdaRank(), new LambdaMART(), new ListNet(), new RFRanker(), new LinearRegRank() };
    protected Map<String, String> map = new HashMap<>();

    public RankerFactory() {
        map.put(createRanker(RankerType.MART).name().toUpperCase(), RankerType.MART.name());
        map.put(createRanker(RankerType.RANKNET).name().toUpperCase(), RankerType.RANKNET.name());
        map.put(createRanker(RankerType.RANKBOOST).name().toUpperCase(), RankerType.RANKBOOST.name());
        map.put(createRanker(RankerType.ADARANK).name().toUpperCase(), RankerType.ADARANK.name());
        map.put(createRanker(RankerType.COOR_ASCENT).name().toUpperCase(), RankerType.COOR_ASCENT.name());
        map.put(createRanker(RankerType.LAMBDARANK).name().toUpperCase(), RankerType.LAMBDARANK.name());
        map.put(createRanker(RankerType.LAMBDAMART).name().toUpperCase(), RankerType.LAMBDAMART.name());
        map.put(createRanker(RankerType.LISTNET).name().toUpperCase(), RankerType.LISTNET.name());
        map.put(createRanker(RankerType.RANDOM_FOREST).name().toUpperCase(), RankerType.RANDOM_FOREST.name());
        map.put(createRanker(RankerType.LINEAR_REGRESSION).name().toUpperCase(), RankerType.LINEAR_REGRESSION.name());
    }

    public void register(final String name, final String className) {
        map.put(name, className);
    }

    public Ranker createRanker(final RankerType type) {
        return rFactory[type.ordinal() - RankerType.MART.ordinal()].createNew();
    }

    public Ranker createRanker(final RankerType type, final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        final Ranker r = createRanker(type);
        r.setTrainingSet(samples);
        r.setFeatures(features);
        r.setMetricScorer(scorer);
        return r;
    }

    public Ranker createRanker(final String className) {
        try {
            final RankerType rankerType = RankerType.valueOf(className);
            return createRanker(rankerType);
        } catch (final Exception e) {
            // ignore
        }

        Ranker r = null;
        try {
            @SuppressWarnings("unchecked")
            final Class<Ranker> c = (Class<Ranker>) Class.forName(className);
            r = c.newInstance();
        } catch (final ClassNotFoundException e) {
            throw RankLibError
                    .create("Could find the class \"" + className + "\" you specified. Make sure the jar library is in your classpath.", e);
        } catch (final InstantiationException e) {
            throw RankLibError.create("Cannot create objects from the class \"" + className + "\" you specified.", e);
        } catch (final IllegalAccessException e) {
            throw RankLibError.create("The class \"" + className + "\" does not implement the Ranker interface.", e);
        }
        return r;
    }

    public Ranker createRanker(final String className, final List<RankList> samples, final int[] features, final MetricScorer scorer) {
        final Ranker r = createRanker(className);
        r.setTrainingSet(samples);
        r.setFeatures(features);
        r.setMetricScorer(scorer);
        return r;
    }

    public Ranker loadRankerFromFile(final String modelFile) {
        return loadRankerFromString(FileUtils.read(modelFile, "ASCII"));
    }

    public Ranker loadRankerFromString(final String fullText) {
        try (BufferedReader in = new BufferedReader(new StringReader(fullText))) {
            Ranker r;
            final String content = in.readLine().replace("## ", "").trim();//read the first line to get the name of the ranking algorithm
            logger.info(() -> "Model:\t\t" + content);
            r = createRanker(map.get(content.toUpperCase()));
            r.loadFromString(fullText);
            return r;
        } catch (final Exception ex) {
            throw RankLibError.create(ex);
        }
    }
}
