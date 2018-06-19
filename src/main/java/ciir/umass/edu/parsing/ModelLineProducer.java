package ciir.umass.edu.parsing;

import ciir.umass.edu.utilities.RankLibError;

/**
 * Created by doug on 5/24/17.
 */
public class ModelLineProducer {
    private static int CARRIAGE_RETURN = 13;
    private static int LINE_FEED = 10;

    private final StringBuilder model = new StringBuilder(1000);

    public interface LineConsumer {
        void nextLine(StringBuilder model, boolean maybeEndEns);
    }

    public StringBuilder getModel() {
        return model;
    }

    private boolean readUntil(final char[] fullTextChar, final int beginOfLineCursor, final int endOfLineCursor,
            final StringBuilder model) {

        // read line in, scan for probable Ensemble

        boolean ens = true;

        if (fullTextChar[beginOfLineCursor] != '#') {
            for (int j = beginOfLineCursor; j <= endOfLineCursor; j++) {
                model.append(fullTextChar[j]);
            }
        }

        // dumb quick hack to see if the reader should check for ensemble tag
        if (endOfLineCursor > 3) {
            ens = (fullTextChar[endOfLineCursor - 9] == '/' && fullTextChar[endOfLineCursor - 2] == 'l'
                    && fullTextChar[endOfLineCursor - 1] == 'e' && fullTextChar[endOfLineCursor] == '>');
        }
        return ens;
    }

    public void parse(final String fullText, final LineConsumer modelConsumer) {

        try {

            final char[] fullTextChar = fullText.toCharArray();

            int beginOfLineCursor = 0;
            for (int i = 0; i < fullTextChar.length; i++) {
                int charNum = fullTextChar[i];
                if (charNum == CARRIAGE_RETURN || charNum == LINE_FEED) {

                    // NEWLINE, read beginOfLineCursor -> i
                    if (fullTextChar[beginOfLineCursor] != '#') {
                        int eolCursor = i;
                        while (eolCursor > beginOfLineCursor && fullTextChar[eolCursor] <= 32) {
                            eolCursor--;
                        }

                        final boolean ens = readUntil(fullTextChar, beginOfLineCursor, eolCursor, model);

                        modelConsumer.nextLine(model, ens);
                    }

                    // readahead this new line up to the next space
                    while (charNum <= 32 & i < fullTextChar.length) {
                        charNum = fullTextChar[i];
                        beginOfLineCursor = i;
                        i++;
                    }
                }
            }

            final boolean ens = readUntil(fullTextChar, beginOfLineCursor, fullTextChar.length - 1, model);

            modelConsumer.nextLine(model, ens);

        } catch (final Exception ex) {
            throw RankLibError.create("Error in model loading ", ex);
        }
    }
}
