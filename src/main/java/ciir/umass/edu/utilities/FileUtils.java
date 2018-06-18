/*===============================================================================
 * Copyright (c) 2010-2015 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import java.util.zip.GZIPInputStream;

/**
 * This class provides some file processing utilities such as read/write files, obtain files in a
 * directory...
 * @author Van Dang
 * @version 1.3 (July 29, 2008)
 */
public class FileUtils {
    private static final Logger logger = Logger.getLogger(FileUtils.class.getName());

    public static BufferedReader smartReader(final String inputFile) throws IOException {
        return smartReader(inputFile, "UTF-8");
    }

    public static BufferedReader smartReader(final String inputFile, final String encoding) throws IOException {
        InputStream input = new FileInputStream(inputFile);
        if (inputFile.endsWith(".gz")) {
            input = new GZIPInputStream(input);
        }
        return new BufferedReader(new InputStreamReader(input, encoding));
    }

    /**
     * Read the content of a file.
     * @param filename The file to read.
     * @param encoding The encoding of the file.
     * @return The content of the input file.
     */
    public static String read(final String filename, final String encoding) {
        final StringBuilder content = new StringBuilder(1000);
        try (BufferedReader in = smartReader(filename, encoding)) {
            final char[] newContent = new char[40960];
            int numRead = -1;
            while ((numRead = in.read(newContent)) != -1) {
                content.append(new String(newContent, 0, numRead));
            }
        } catch (final Exception e) {
            throw RankLibError.create(e);
        }
        return content.toString();
    }

    public static List<String> readLine(final String filename, final String encoding) {
        final List<String> lines = new ArrayList<>();
        try (final BufferedReader in = smartReader(filename, encoding)) {
            String content = "";

            while ((content = in.readLine()) != null) {
                content = content.trim();
                if (content.length() == 0) {
                    continue;
                }
                lines.add(content);
            }
        } catch (final Exception ex) {
            throw RankLibError.create(ex);
        }
        return lines;
    }

    /**
     * Write a text to a file.
     * @param filename The output filename.
     * @param encoding The encoding of the file.
     * @param strToWrite The string to write.
     */
    public static void write(final String filename, final String encoding, final String strToWrite) {
        try (final BufferedWriter out = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), encoding))) {
            out.write(strToWrite);
        } catch (final Exception e) {
            throw RankLibError.create(e);
        }
    }

    /**
     * Get all file (non-recursively) from a directory.
     * @param directory The directory to read.
     * @return A list of filenames (without path) in the input directory.
     */
    public static String[] getAllFiles(final String directory) {
        final File dir = new File(directory);
        return dir.list();
    }

    /**
     * Get all file (non-recursively) from a directory.
     * @param directory The directory to read.
     * @return A list of filenames (without path) in the input directory.
     */
    public static List<String> getAllFiles2(final String directory) {
        final File dir = new File(directory);
        final String[] fns = dir.list();
        final List<String> files = new ArrayList<>();
        if (fns != null) {
            for (final String fn : fns) {
                files.add(fn);
            }
        }
        return files;
    }

    public static String getFileName(final String pathName) {
        final int idx1 = pathName.lastIndexOf("/");
        final int idx2 = pathName.lastIndexOf("\\");
        final int idx = (idx1 > idx2) ? idx1 : idx2;
        return pathName.substring(idx + 1);
    }

    public static String makePathStandard(final String directory) {
        String dir = directory;
        final char c = dir.charAt(dir.length() - 1);
        if (c != '/' && c != '\\') {
            //- I THINK we want File.separator (/ or \) instead of
            //  File.pathSeparator (: or ;) here.  Maybe needed for Analyzer?
            //dir += File.pathSeparator;
            dir += File.separator;
        }
        return dir;
    }
}
