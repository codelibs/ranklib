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
 * This class contains the implementation of some simple sorting algorithms.
 * @author Van Dang
 * @version 1.3 (July 29, 2008)
 */
public class Sorter {

    /**
     * Sort a double array using Interchange sort.
     * @param sortVal The double array to be sorted.
     * @param asc TRUE to sort ascendingly, FALSE to sort descendingly.
     * @return The sorted indexes.
     */
    public static int[] sort(final double[] sortVal, final boolean asc) {
        final int[] freqIdx = new int[sortVal.length];
        for (int i = 0; i < sortVal.length; i++) {
            freqIdx[i] = i;
        }
        for (int i = 0; i < sortVal.length - 1; i++) {
            int max = i;
            for (int j = i + 1; j < sortVal.length; j++) {
                if (asc) {
                    if (sortVal[freqIdx[max]] > sortVal[freqIdx[j]]) {
                        max = j;
                    }
                } else {
                    if (sortVal[freqIdx[max]] < sortVal[freqIdx[j]]) {
                        max = j;
                    }
                }
            }
            //swap
            final int tmp = freqIdx[i];
            freqIdx[i] = freqIdx[max];
            freqIdx[max] = tmp;
        }
        return freqIdx;
    }

    public static int[] sort(final float[] sortVal, final boolean asc) {
        final int[] freqIdx = new int[sortVal.length];
        for (int i = 0; i < sortVal.length; i++) {
            freqIdx[i] = i;
        }
        for (int i = 0; i < sortVal.length - 1; i++) {
            int max = i;
            for (int j = i + 1; j < sortVal.length; j++) {
                if (asc) {
                    if (sortVal[freqIdx[max]] > sortVal[freqIdx[j]]) {
                        max = j;
                    }
                } else {
                    if (sortVal[freqIdx[max]] < sortVal[freqIdx[j]]) {
                        max = j;
                    }
                }
            }
            //swap
            final int tmp = freqIdx[i];
            freqIdx[i] = freqIdx[max];
            freqIdx[max] = tmp;
        }
        return freqIdx;
    }

    /**
     * Sort an integer array using Quick Sort.
     * @param sortVal The integer array to be sorted.
     * @param asc TRUE to sort ascendingly, FALSE to sort descendingly.
     * @return The sorted indexes.
     */
    public static int[] sort(final int[] sortVal, final boolean asc) {
        return qSort(sortVal, asc);
    }

    /**
     * Sort an integer array using Quick Sort.
     * @param sortVal The integer array to be sorted.
     * @param asc TRUE to sort ascendingly, FALSE to sort descendingly.
     * @return The sorted indexes.
     */
    public static int[] sort(final List<Integer> sortVal, final boolean asc) {
        return qSort(sortVal, asc);
    }

    public static int[] sortString(final List<String> sortVal, final boolean asc) {
        return qSortString(sortVal, asc);
    }

    /**
     * Sort an long array using Quick Sort.
     * @param sortVal The long array to be sorted.
     * @param asc TRUE to sort ascendingly, FALSE to sort descendingly.
     * @return The sorted indexes.
     */
    public static int[] sortLong(final List<Long> sortVal, final boolean asc) {
        return qSortLong(sortVal, asc);
    }

    /**
     * Sort an double array using Quick Sort.
     * @param sortVal The double array to be sorted.
     * @return The sorted indexes.
     */
    public static int[] sortDesc(final List<Double> sortVal) {
        return qSortDouble(sortVal, false);
    }

    /**
     * Quick sort internal
     * @param l The list to sort.
     * @param asc Ascending/Descendingly parameter.
     * @return The sorted indexes.
     */
    private static int[] qSort(final List<Integer> l, final boolean asc) {
        final int[] idx = new int[l.size()];
        List<Integer> idxList = new ArrayList<>();
        for (int i = 0; i < l.size(); i++) {
            idxList.add(i);
        }
        idxList = qSort(l, idxList, asc);
        for (int i = 0; i < l.size(); i++) {
            idx[i] = idxList.get(i);
        }
        return idx;
    }

    private static int[] qSortString(final List<String> l, final boolean asc) {
        final int[] idx = new int[l.size()];
        List<Integer> idxList = new ArrayList<>();
        for (int i = 0; i < l.size(); i++) {
            idxList.add(i);
        }
        idxList = qSortString(l, idxList, asc);
        for (int i = 0; i < l.size(); i++) {
            idx[i] = idxList.get(i);
        }
        return idx;
    }

    /**
     * Quick sort internal
     * @param l The list to sort.
     * @param asc Ascending/Descendingly parameter.
     * @return The sorted indexes.
     */
    private static int[] qSortLong(final List<Long> l, final boolean asc) {
        final int[] idx = new int[l.size()];
        List<Integer> idxList = new ArrayList<>();
        for (int i = 0; i < l.size(); i++) {
            idxList.add(i);
        }
        idxList = qSortLong(l, idxList, asc);
        for (int i = 0; i < l.size(); i++) {
            idx[i] = idxList.get(i);
        }
        return idx;
    }

    /**
     * Quick sort internal
     * @param l The list to sort.
     * @param asc Ascending/Descendingly parameter.
     * @return The sorted indexes.
     */
    private static int[] qSortDouble(final List<Double> l, final boolean asc) {
        final int[] idx = new int[l.size()];
        List<Integer> idxList = new ArrayList<>();
        for (int i = 0; i < l.size(); i++) {
            idxList.add(i);
        }
        idxList = qSortDouble(l, idxList, asc);
        for (int i = 0; i < l.size(); i++) {
            idx[i] = idxList.get(i);
        }
        return idx;
    }

    /**
     * Sort an integer array using Quick Sort.
     * @param l The integer array to be sorted.
     * @param asc TRUE to sort ascendingly, FALSE to sort descendingly.
     * @return The sorted indexes.
     */
    private static int[] qSort(final int[] l, final boolean asc) {
        final int[] idx = new int[l.length];
        List<Integer> idxList = new ArrayList<>();
        for (int i = 0; i < l.length; i++) {
            idxList.add(i);
        }
        idxList = qSort(l, idxList, asc);
        for (int i = 0; i < l.length; i++) {
            idx[i] = idxList.get(i);
        }
        return idx;
    }

    /**
     * Quick sort internal.
     * @param l
     * @param idxList
     * @param asc
     * @return  The sorted indexes.
     */
    private static List<Integer> qSort(final List<Integer> l, final List<Integer> idxList, final boolean asc) {
        final int mid = idxList.size() / 2;
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        final List<Integer> pivot = new ArrayList<>();
        for (int i = 0; i < idxList.size(); i++) {
            if (l.get(idxList.get(i)) > l.get(idxList.get(mid))) {
                if (asc) {
                    right.add(idxList.get(i));
                } else {
                    left.add(idxList.get(i));
                }
            } else if (l.get(idxList.get(i)) < l.get(idxList.get(mid))) {
                if (asc) {
                    left.add(idxList.get(i));
                } else {
                    right.add(idxList.get(i));
                }
            } else {
                pivot.add(idxList.get(i));
            }
        }
        if (left.size() > 1) {
            left = qSort(l, left, asc);
        }
        if (right.size() > 1) {
            right = qSort(l, right, asc);
        }
        final List<Integer> newIdx = new ArrayList<>();
        newIdx.addAll(left);
        newIdx.addAll(pivot);
        newIdx.addAll(right);
        return newIdx;
    }

    private static List<Integer> qSortString(final List<String> l, final List<Integer> idxList, final boolean asc) {
        final int mid = idxList.size() / 2;
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        final List<Integer> pivot = new ArrayList<>();
        for (int i = 0; i < idxList.size(); i++) {
            if (l.get(idxList.get(i)).compareTo(l.get(idxList.get(mid))) > 0) {
                if (asc) {
                    right.add(idxList.get(i));
                } else {
                    left.add(idxList.get(i));
                }
            } else if (l.get(idxList.get(i)).compareTo(l.get(idxList.get(mid))) < 0) {
                if (asc) {
                    left.add(idxList.get(i));
                } else {
                    right.add(idxList.get(i));
                }
            } else {
                pivot.add(idxList.get(i));
            }
        }
        if (left.size() > 1) {
            left = qSortString(l, left, asc);
        }
        if (right.size() > 1) {
            right = qSortString(l, right, asc);
        }
        final List<Integer> newIdx = new ArrayList<>();
        newIdx.addAll(left);
        newIdx.addAll(pivot);
        newIdx.addAll(right);
        return newIdx;
    }

    /**
     * Quick sort internal.
     * @param l
     * @param idxList
     * @param asc
     * @return The sorted indexes.
     */
    private static List<Integer> qSort(final int[] l, final List<Integer> idxList, final boolean asc) {
        final int mid = idxList.size() / 2;
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        final List<Integer> pivot = new ArrayList<>();
        for (int i = 0; i < idxList.size(); i++) {
            if (l[idxList.get(i)] > l[idxList.get(mid)]) {
                if (asc) {
                    right.add(idxList.get(i));
                } else {
                    left.add(idxList.get(i));
                }
            } else if (l[idxList.get(i)] < l[idxList.get(mid)]) {
                if (asc) {
                    left.add(idxList.get(i));
                } else {
                    right.add(idxList.get(i));
                }
            } else {
                pivot.add(idxList.get(i));
            }
        }
        if (left.size() > 1) {
            left = qSort(l, left, asc);
        }
        if (right.size() > 1) {
            right = qSort(l, right, asc);
        }
        final List<Integer> newIdx = new ArrayList<>();
        newIdx.addAll(left);
        newIdx.addAll(pivot);
        newIdx.addAll(right);
        return newIdx;
    }

    /**
     * Quick sort internal.
     * @param l
     * @param idxList
     * @param asc
     * @return  The sorted indexes.
     */
    private static List<Integer> qSortDouble(final List<Double> l, final List<Integer> idxList, final boolean asc) {
        final int mid = idxList.size() / 2;
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        final List<Integer> pivot = new ArrayList<>();
        for (int i = 0; i < idxList.size(); i++) {
            if (l.get(idxList.get(i)) > l.get(idxList.get(mid))) {
                if (asc) {
                    right.add(idxList.get(i));
                } else {
                    left.add(idxList.get(i));
                }
            } else if (l.get(idxList.get(i)) < l.get(idxList.get(mid))) {
                if (asc) {
                    left.add(idxList.get(i));
                } else {
                    right.add(idxList.get(i));
                }
            } else {
                pivot.add(idxList.get(i));
            }
        }
        if (left.size() > 1) {
            left = qSortDouble(l, left, asc);
        }
        if (right.size() > 1) {
            right = qSortDouble(l, right, asc);
        }
        final List<Integer> newIdx = new ArrayList<>();
        newIdx.addAll(left);
        newIdx.addAll(pivot);
        newIdx.addAll(right);
        return newIdx;
    }

    /**
     * Quick sort internal.
     * @param l
     * @param idxList
     * @param asc
     * @return The sorted indexes.
     */
    private static List<Integer> qSortLong(final List<Long> l, final List<Integer> idxList, final boolean asc) {
        final int mid = idxList.size() / 2;
        List<Integer> left = new ArrayList<>();
        List<Integer> right = new ArrayList<>();
        final List<Integer> pivot = new ArrayList<>();
        for (int i = 0; i < idxList.size(); i++) {
            if (l.get(idxList.get(i)) > l.get(idxList.get(mid))) {
                if (asc) {
                    right.add(idxList.get(i));
                } else {
                    left.add(idxList.get(i));
                }
            } else if (l.get(idxList.get(i)) < l.get(idxList.get(mid))) {
                if (asc) {
                    left.add(idxList.get(i));
                } else {
                    right.add(idxList.get(i));
                }
            } else {
                pivot.add(idxList.get(i));
            }
        }
        if (left.size() > 1) {
            left = qSortLong(l, left, asc);
        }
        if (right.size() > 1) {
            right = qSortLong(l, right, asc);
        }
        final List<Integer> newIdx = new ArrayList<>();
        newIdx.addAll(left);
        newIdx.addAll(pivot);
        newIdx.addAll(right);
        return newIdx;
    }
}
