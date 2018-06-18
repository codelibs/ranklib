package ciir.umass.edu.utilities;

/**
 * Instead of using random error types, use RankLibError exceptions throughout
 *   -- this means that clients can catch-all from us easily.
 * @author jfoley
 */
// TODO This class name should be changed to RankLibException
public class RankLibError extends RuntimeException {
    private static final long serialVersionUID = 1L;

    private RankLibError(final Exception e) {
        super(e);
    }

    private RankLibError(final String message) {
        super(message);
    }

    private RankLibError(final String message, final Exception cause) {
        super(message, cause);
    }

    /** Don't rewrap RankLibErrors in RankLibErrors */
    public static RankLibError create(final Exception e) {
        if (e instanceof RankLibError) {
            return (RankLibError) e;
        }
        return new RankLibError(e);
    }

    public static RankLibError create(final String message) {
        return new RankLibError(message);
    }

    /** Don't rewrap RankLibErrors in RankLibErrors */
    public static RankLibError create(final String message, final Exception cause) {
        if (cause instanceof RankLibError) {
            return (RankLibError) cause;
        }
        return new RankLibError(message, cause);
    }
}
