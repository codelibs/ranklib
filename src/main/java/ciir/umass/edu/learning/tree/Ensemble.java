/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.learning.tree;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;

import org.w3c.dom.Document;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import ciir.umass.edu.learning.DataPoint;
import ciir.umass.edu.utilities.RankLibError;

/**
 * @author vdang
 */
public class Ensemble {
    protected List<RegressionTree> trees = new ArrayList<>();
    protected List<Float> weights = new ArrayList<>();
    protected int[] features = null;

    public Ensemble() {
    }

    public Ensemble(final Ensemble e) {
        trees.addAll(e.trees);
        weights.addAll(e.weights);
    }

    public Ensemble(final String xmlRep) {
        try (final InputStream in = new ByteArrayInputStream(xmlRep.getBytes("UTF-8"))) {
            final DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
            final DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
            final Document doc = dBuilder.parse(in);
            final NodeList nl = doc.getElementsByTagName("tree");
            final Map<Integer, Integer> fids = new HashMap<>();
            for (int i = 0; i < nl.getLength(); i++) {
                final Node n = nl.item(i);//each node corresponds to a "tree" (tag)
                //create a regression tree from this node
                final Split root = create(n.getFirstChild(), fids);
                //get the weight for this tree
                final float weight = Float.parseFloat(n.getAttributes().getNamedItem("weight").getNodeValue());
                //add it to the ensemble
                trees.add(new RegressionTree(root));
                weights.add(weight);
            }
            features = new int[fids.keySet().size()];
            int i = 0;
            for (final Integer fid : fids.keySet()) {
                features[i++] = fid;
            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in Emsemble(xmlRepresentation): ", ex);
        }
    }

    public void add(final RegressionTree tree, final float weight) {
        trees.add(tree);
        weights.add(weight);
    }

    public RegressionTree getTree(final int k) {
        return trees.get(k);
    }

    public float getWeight(final int k) {
        return weights.get(k);
    }

    public double variance() {
        double var = 0;
        for (final RegressionTree tree : trees) {
            var += tree.variance();
        }
        return var;
    }

    public void remove(final int k) {
        trees.remove(k);
        weights.remove(k);
    }

    public int treeCount() {
        return trees.size();
    }

    public int leafCount() {
        int count = 0;
        for (final RegressionTree tree : trees) {
            count += tree.leaves().size();
        }
        return count;
    }

    public float eval(final DataPoint dp) {
        float s = 0;
        for (int i = 0; i < trees.size(); i++) {
            s += trees.get(i).eval(dp) * weights.get(i);
        }
        return s;
    }

    @Override
    public String toString() {
        final StringBuilder buf = new StringBuilder(1000);
        buf.append("<ensemble>\n");
        for (int i = 0; i < trees.size(); i++) {
            buf.append("\t<tree id=\"").append(Integer.toString(i + 1)).append("\" weight=\"").append(Float.toString(weights.get(i)))
                    .append("\">\n");
            buf.append(trees.get(i).toString("\t\t"));
            buf.append("\t</tree>\n");
        }
        buf.append("</ensemble>\n");
        return buf.toString();
    }

    public int[] getFeatures() {
        return features;
    }

    /**
     * Each input node @n corersponds to a <split> tag in the model file.
     * @param n
     * @return
     */
    private Split create(final Node n, final Map<Integer, Integer> fids) {
        Split s = null;
        if (n.getFirstChild().getNodeName().compareToIgnoreCase("feature") == 0)//this is a split
        {
            final NodeList nl = n.getChildNodes();
            final int fid = Integer.parseInt(nl.item(0).getFirstChild().getNodeValue().trim());//<feature>
            fids.put(fid, 0);
            final float threshold = Float.parseFloat(nl.item(1).getFirstChild().getNodeValue().trim());//<threshold>
            s = new Split(fid, threshold, 0);
            s.setLeft(create(nl.item(2), fids));
            s.setRight(create(nl.item(3), fids));
        } else//this is a stump
        {
            final float output = Float.parseFloat(n.getFirstChild().getFirstChild().getNodeValue().trim());
            s = new Split();
            s.setOutput(output);
        }
        return s;
    }
}
