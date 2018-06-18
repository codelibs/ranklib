/*===============================================================================
 * Copyright (c) 2010-2012 University of Massachusetts.  All Rights Reserved.
 *
 * Use of the RankLib package is subject to the terms of the software license set
 * forth in the LICENSE file included with this software, and also available at
 * http://people.cs.umass.edu/~vdang/ranklib_license.html
 *===============================================================================
 */

package ciir.umass.edu.utilities;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 *
 * @author vdang
 *
 */
public class MyThreadPool extends ThreadPoolExecutor {

    private final Semaphore semaphore;
    private int size = 0;

    private MyThreadPool(final int size) {
        super(size, size, 0, TimeUnit.MILLISECONDS, new LinkedBlockingQueue<Runnable>());
        semaphore = new Semaphore(size, true);
        this.size = size;
    }

    private static MyThreadPool singleton = null;

    public static MyThreadPool getInstance() {
        if (singleton == null) {
            init(Runtime.getRuntime().availableProcessors());
        }
        return singleton;
    }

    public static void init(final int poolSize) {
        singleton = new MyThreadPool(poolSize);
    }

    public int size() {
        return size;
    }

    public WorkerThread[] execute(final WorkerThread worker, final int nTasks) {
        final MyThreadPool p = MyThreadPool.getInstance();
        final int[] partition = p.partition(nTasks);
        final WorkerThread[] workers = new WorkerThread[partition.length - 1];
        for (int i = 0; i < partition.length - 1; i++) {
            final WorkerThread w = worker.clone();
            w.set(partition[i], partition[i + 1] - 1);
            workers[i] = w;
            p.execute(w);
        }
        await();
        return workers;
    }

    public void await() {
        for (int i = 0; i < size; i++) {
            try {
                semaphore.acquire();
            } catch (final Exception ex) {
                throw RankLibError.create("Error in MyThreadPool.await(): ", ex);
            }
        }
        for (int i = 0; i < size; i++) {
            semaphore.release();
        }
    }

    public int[] partition(final int listSize) {
        final int nChunks = Math.min(listSize, size);
        final int chunkSize = listSize / nChunks;
        final int mod = listSize % nChunks;
        final int[] partition = new int[nChunks + 1];
        partition[0] = 0;
        for (int i = 1; i <= nChunks; i++) {
            partition[i] = partition[i - 1] + chunkSize + ((i <= mod) ? 1 : 0);
        }
        return partition;
    }

    @Override
    public void execute(final Runnable task) {
        try {
            semaphore.acquire();
            super.execute(task);
        } catch (final Exception ex) {
            throw RankLibError.create("Error in MyThreadPool.execute(): ", ex);
        }
    }

    @Override
    protected void afterExecute(final Runnable r, final Throwable t) {
        super.afterExecute(r, t);
        semaphore.release();
    }
}
