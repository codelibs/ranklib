/*===============================================================================
 * Copyright (c) 2010-2015 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
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
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * This class provides some file processing utilities such as read/write files, obtain files in a
 * directory...
 * @author Van Dang
 * @version 1.3 (July 29, 2008)
 */
public class FileUtils {

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
            String content = null;

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

    /**
     * Test whether a file/directory exists.
     * @param file the file/directory to test.
     * @return TRUE if exists; FALSE otherwise.
     */
    public static boolean exists(final String file) {
        final File f = new File(file);
        return f.exists();
    }

    /**
     * Copy a file.
     * @param srcFile The source file.
     * @param dstFile The copied file.
     */
    public static void copyFile(final String srcFile, final String dstFile) {
        try (FileInputStream fis = new FileInputStream(new File(srcFile)); FileOutputStream fos = new FileOutputStream(new File(dstFile))) {
            final byte[] buf = new byte[40960];
            int i = 0;
            while ((i = fis.read(buf)) != -1) {
                fos.write(buf, 0, i);
            }
        } catch (final IOException e) {
            throw RankLibError.create("Error in FileUtils.copyFile: ", e);
        }
    }

    /**
     * Copy all files in the source directory to the target directory.
     * @param srcDir The source directory.
     * @param dstDir The target directory.
     * @param files The files to be copied. NOTE THAT this list contains only names (WITHOUT PATH).
     */
    public static void copyFiles(final String srcDir, final String dstDir, final List<String> files) {
        for (final String file : files) {
            FileUtils.copyFile(srcDir + file, dstDir + file);
        }
    }

    public static final int BUF_SIZE = 51200;

    /**
     * Gunzip an input file.
     * @param fileInput    Input file to gunzip.
     * @param dirOutput    Output directory to contain the ungzipped file (whose name = file_input - ".gz")
     * @return 1 if succeed
     */
    public static int gunzipFile(final File fileInput, final File dirOutput) {
        // Use the name of the archive for the output file name but
        // with ".gz" stripped off.
        final String fileInputName = fileInput.getName();
        final String fileOutputName = fileInputName.substring(0, fileInputName.length() - 3);

        // Create the decompressed output file.
        final File outputFile = new File(dirOutput, fileOutputName);

        // Decompress the gzipped file by reading it via
        // the GZIP input stream. Will need a buffer.
        final byte[] inputBuffer = new byte[BUF_SIZE];
        int len = 0;

        try (GZIPInputStream gin = new GZIPInputStream(new BufferedInputStream(new FileInputStream(fileInput)));
                BufferedOutputStream destination = new BufferedOutputStream(new FileOutputStream(outputFile), BUF_SIZE)) {

            //Now read from the gzip stream, which will decompress the data,
            //and write to the output stream.
            while ((len = gin.read(inputBuffer, 0, BUF_SIZE)) != -1) {
                destination.write(inputBuffer, 0, len);
            }
            destination.flush(); // Insure that all data is written to the output.

        } catch (final IOException e) {
            throw RankLibError.create("Error in gunzipFile(): " + e.toString(), e);
        }
        return 1;
    }

    /**
     * Gzip an input file.
     * @param inputFile The input file to gzip.
     * @param gzipFilename The gunzipped file's name.
     * @return 1 if succeeds
     */
    public static int gzipFile(final String inputFile, final String gzipFilename) {
        try (FileInputStream in = new FileInputStream(inputFile);
                GZIPOutputStream out = new GZIPOutputStream(new FileOutputStream(gzipFilename))) {

            // Transfer bytes from the input file
            // to the gzip output stream
            final byte[] buf = new byte[BUF_SIZE];
            int len;
            while ((len = in.read(buf)) > 0) {
                out.write(buf, 0, len);
            }

            // Finish creation of gzip file
            out.finish();
        } catch (final Exception ex) {
            throw RankLibError.create(ex);
        }
        return 1;
    }

    public static String getFileName(final String pathName) {
        final int idx1 = pathName.lastIndexOf('/');
        final int idx2 = pathName.lastIndexOf('\\');
        final int idx = (idx1 > idx2) ? idx1 : idx2;
        return pathName.substring(idx + 1);
    }

    public static String makePathStandard(final String directory) {
        final String dir = directory;
        final char c = dir.charAt(dir.length() - 1);
        if (c != '/' && c != '\\') {
            return dir + File.separator;
        }
        return dir;
    }

}
