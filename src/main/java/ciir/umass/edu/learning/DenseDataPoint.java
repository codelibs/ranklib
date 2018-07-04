package ciir.umass.edu.learning;

import ciir.umass.edu.utilities.RankLibError;

public class DenseDataPoint extends DataPoint {

    public DenseDataPoint(final String text) {
        super(text);
    }

    public DenseDataPoint(final DenseDataPoint dp) {
        label = dp.label;
        id = dp.id;
        description = dp.description;
        cached = dp.cached;
        fVals = new float[dp.fVals.length];
        System.arraycopy(dp.fVals, 0, fVals, 0, dp.fVals.length);
    }

    @Override
    public float getFeatureValue(final int fid) {
        if (fid <= 0 || fid >= fVals.length) {
            if (missingZero) {
                return 0f;
            }
            throw RankLibError.create("Error in DenseDataPoint::getFeatureValue(): requesting unspecified feature, fid=" + fid);
        }
        if (isUnknown(fVals[fid])) {
            return 0;
        }
        return fVals[fid];
    }

    @Override
    public void setFeatureValue(final int fid, final float fval) {
        if (fid <= 0 || fid >= fVals.length) {
            throw RankLibError.create("Error in DenseDataPoint::setFeatureValue(): feature (id=" + fid + ") not found.");
        }
        fVals[fid] = fval;
    }

    @Override
    public void setFeatureVector(final float[] dfVals) {
        fVals = dfVals;
    }

    @Override
    public float[] getFeatureVector() {
        return fVals;
    }
}
