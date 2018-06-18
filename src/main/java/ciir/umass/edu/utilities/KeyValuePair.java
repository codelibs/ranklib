/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.util.ArrayList;
import java.util.List;

/**
 * @author vdang
 */
public class KeyValuePair {

    protected List<String> keys = new ArrayList<>();;
    protected List<String> values = new ArrayList<>();;

    public KeyValuePair(String text) {
        try {
            final int idx = text.lastIndexOf('#');
            if (idx != -1) {
                text = text.substring(0, idx).trim();//remove the comment part at the end of the line
            }

            final String[] fs = text.split(" ");
            for (int i = 0; i < fs.length; i++) {
                fs[i] = fs[i].trim();
                if (fs[i].compareTo("") == 0) {
                    continue;
                }
                keys.add(getKey(fs[i]));
                values.add(getValue(fs[i]));

            }
        } catch (final Exception ex) {
            throw RankLibError.create("Error in KeyValuePair(text) constructor", ex);
        }
    }

    public List<String> keys() {
        return keys;
    }

    public List<String> values() {
        return values;
    }

    private String getKey(final String pair) {
        return pair.substring(0, pair.indexOf(':'));
    }

    private String getValue(final String pair) {
        return pair.substring(pair.lastIndexOf(':') + 1);
    }
}
