using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace AdaptiveSort
{
    // Simple type traits used to detect blittable value types once per T.
    internal static class TypeTraits<T>
    {
        public static readonly bool IsBlittable =
            typeof(T).IsValueType && !RuntimeHelpers.IsReferenceOrContainsReferences<T>();
    }

    public static class AdaptiveSortExtensions
    {
        private const int INSERTION_SORT_THRESHOLD = 32;
        private const int STACKALLOC_BYTES_THRESHOLD = 4 * 1024; // tune as needed

        // --------- Public AdaptiveSort entry points ---------

        public static void AdaptiveSort<T>(this List<T> list)
            where T : IComparable<T>
        {
            int n = list.Count;
            if (n <= 1)
                return;

            Span<T> span = CollectionsMarshal.AsSpan(list);

            if (n < INSERTION_SORT_THRESHOLD)
            {
                InsertionSort(span, 0, n - 1);
            }
            else if (n > 100_000)
            {
                // Framework sort is highly optimized for large sizes.
                list.Sort();
            }
            else
            {
                Timsort(span);
            }
        }

        public static void AdaptiveSort<T>(this T[] array)
            where T : IComparable<T>
        {
            int n = array.Length;
            if (n <= 1)
                return;

            Span<T> span = array.AsSpan();

            if (n < INSERTION_SORT_THRESHOLD)
            {
                InsertionSort(span, 0, n - 1);
            }
            else if (n > 100_000)
            {
                Array.Sort(array);
            }
            else
            {
                Timsort(span);
            }
        }

        public static void AdaptiveSort<TKey, TValue>(this Dictionary<TKey, TValue> dict)
            where TKey : IComparable<TKey>
        {
            int count = dict.Count;
            if (count <= 1)
                return;

            // Single allocation, exact capacity.
            var entries = new List<KeyValuePair<TKey, TValue>>(count);
            foreach (var kvp in dict)
                entries.Add(kvp);

            entries.Sort((a, b) => a.Key.CompareTo(b.Key));

            dict.Clear();
            for (int i = 0; i < entries.Count; i++)
            {
                var kvp = entries[i];
                dict.Add(kvp.Key, kvp.Value);
            }
        }

        public static void AdaptiveSort<T>(this HashSet<T> hashSet)
            where T : IComparable<T>
        {
            int count = hashSet.Count;
            if (count <= 1)
                return;

            var list = new List<T>(count);
            foreach (var item in hashSet)
                list.Add(item);

            list.AdaptiveSort();

            hashSet.Clear();
            for (int i = 0; i < list.Count; i++)
                hashSet.Add(list[i]);
        }

        public static void AdaptiveSort(this Hashtable hashtable)
        {
            int count = hashtable.Count;
            if (count <= 1)
                return;

            var entries = new List<DictionaryEntry>(count);
            foreach (DictionaryEntry entry in hashtable)
                entries.Add(entry);

            entries.Sort((a, b) => Comparer.Default.Compare(a.Key, b.Key));

            hashtable.Clear();
            for (int i = 0; i < entries.Count; i++)
            {
                var entry = entries[i];
                hashtable.Add(entry.Key, entry.Value);
            }
        }

        // --------- Core Timsort over Span<T> ---------

private static void Timsort<T>(Span<T> span)
    where T : IComparable<T>
{
    int n = span.Length;
    if (n <= 1)
        return;

    // One reusable temporary buffer per sort.
    T[] temp = new T[n];

    // 1. Sort small runs with insertion sort.
    for (int i = 0; i < n; i += INSERTION_SORT_THRESHOLD)
    {
        int right = Math.Min(i + INSERTION_SORT_THRESHOLD - 1, n - 1);
        InsertionSort(span, i, right);
    }

    // 2. Bottom-up merge.
    for (int size = INSERTION_SORT_THRESHOLD; size < n; size <<= 1)
    {
        for (int left = 0; left < n - 1; left += (size << 1))
        {
            int mid = left + size - 1;
            if (mid >= n - 1)
                break;

            int right = Math.Min(left + (size << 1) - 1, n - 1);
            MergeRuns(span, left, mid, right, temp);
        }
    }
}

private static void MergeRuns<T>(Span<T> span, int left, int mid, int right, T[] temp)
    where T : IComparable<T>
{
    int leftSize = mid - left + 1;
    if (leftSize <= 0)
        return;

    // Reuse the shared temp buffer; we only touch the first leftSize elements.
    Span<T> leftSpan = temp.AsSpan(0, leftSize);

    // Copy left run into leftSpan.
    for (int i = 0; i < leftSize; i++)
        leftSpan[i] = span[left + i];

    int iLeft = 0;
    int iRight = mid + 1;
    int k = left;

    while (iLeft < leftSize && iRight <= right)
    {
        if (leftSpan[iLeft].CompareTo(span[iRight]) <= 0)
        {
            span[k++] = leftSpan[iLeft++];
        }
        else
        {
            span[k++] = span[iRight++];
        }
    }

    while (iLeft < leftSize)
    {
        span[k++] = leftSpan[iLeft++];
    }
}

        // --------- Insertion sort over Span<T> ---------

        private static void InsertionSort<T>(Span<T> span, int left, int right)
            where T : IComparable<T>
        {
            for (int i = left + 1; i <= right; i++)
            {
                T key = span[i];
                int j = i - 1;

                while (j >= left && span[j].CompareTo(key) > 0)
                {
                    span[j + 1] = span[j];
                    j--;
                }

                span[j + 1] = key;
            }
        }
    }

    // --------- GPU radix sort (int) with cached context and kernels ---------

    public static class GpuSortExtensions
    {
        private sealed class AcceleratorState
        {
            public Accelerator Accelerator { get; }
            public Action<Index1D, ArrayView<int>, int, ArrayView<int>> HistogramKernel { get; }
            public Action<Index1D, ArrayView<int>, ArrayView<int>> ExclusiveScanKernel { get; }
            public Action<Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>> ScatterKernel { get; }

            public AcceleratorState(Accelerator accelerator)
            {
                Accelerator = accelerator;
                HistogramKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>>(HistogramKernelImpl);
                ExclusiveScanKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>>(ExclusiveScanKernelImpl);
                ScatterKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>>(ScatterKernelImpl);
            }
        }

        private static readonly Context s_context;
        private static readonly AcceleratorState[] s_states;

        static GpuSortExtensions()
        {
            s_context = Context.Create(builder => builder.Cuda());

            // Collect CUDA accelerators without LINQ.
            var deviceList = new List<Device>();
            foreach (var device in s_context.Devices)
            {
                if (device.AcceleratorType == AcceleratorType.Cuda)
                    deviceList.Add(device);
            }

            if (deviceList.Count == 0)
            {
                s_states = Array.Empty<AcceleratorState>();
                return;
            }

            s_states = new AcceleratorState[deviceList.Count];
            for (int i = 0; i < deviceList.Count; i++)
            {
                var acc = s_context.CreateCudaAccelerator(i);
                s_states[i] = new AcceleratorState(acc);
            }
        }

        /// <summary>
        /// Performs an in-place GPU-based radix sort on an int array.
        /// </summary>
        public static void AdaptiveGpuSort(this int[] array)
        {
            if (s_states.Length == 0)
                throw new InvalidOperationException("No CUDA GPUs available.");

            if (s_states.Length == 1)
                OptimizedGpuSort(array, s_states[0]);
            else
                MultiGpuSort(array, s_states);
        }

        /// <summary>
        /// Single-device radix sort using a cached accelerator.
        /// 4 passes (8 bits per pass).
        /// </summary>
        private static void OptimizedGpuSort(int[] array, AcceleratorState state)
        {
            int length = array.Length;
            if (length == 0)
                return;

            var accelerator = state.Accelerator;

            // Allocate device buffers.
            var inputBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(length, default);
            var outputBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(length, default);
            var histogramBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(256, default);
            var prefixBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(256, default);

            try
            {
                inputBuffer.CopyFromCPU(array);

                var histogramKernel = state.HistogramKernel;
                var exclusiveScanKernel = state.ExclusiveScanKernel;
                var scatterKernel = state.ScatterKernel;

                for (int shift = 0; shift < 32; shift += 8)
                {
                    histogramBuffer.MemSetToZero();

                    histogramKernel((int)inputBuffer.Length, inputBuffer.View, shift, histogramBuffer.View);
                    accelerator.Synchronize();

                    // Exclusive scan is small; use a single work-item.
                    exclusiveScanKernel(1, histogramBuffer.View, prefixBuffer.View);
                    accelerator.Synchronize();

                    scatterKernel((int)inputBuffer.Length, inputBuffer.View, shift, prefixBuffer.View, outputBuffer.View);
                    accelerator.Synchronize();

                    SwapBuffers(ref inputBuffer, ref outputBuffer);
                }

                inputBuffer.CopyToCPU(array);
            }
            finally
            {
                inputBuffer.Dispose();
                outputBuffer.Dispose();
                histogramBuffer.Dispose();
                prefixBuffer.Dispose();
            }
        }

        /// <summary>
        /// Multi-device radix sort: split array into chunks, sort each on a GPU, then k-way merge.
        /// </summary>
        private static void MultiGpuSort(int[] array, AcceleratorState[] states)
        {
            int length = array.Length;
            if (length == 0)
                return;

            int gpuCount = states.Length;
            int chunkSize = length / gpuCount;
            var tasks = new Task[gpuCount];
            var sortedChunks = new int[gpuCount][];

            for (int i = 0; i < gpuCount; i++)
            {
                int start = i * chunkSize;
                int end = (i == gpuCount - 1) ? length : start + chunkSize;
                int len = end - start;

                var chunk = new int[len];
                Array.Copy(array, start, chunk, 0, len);

                AcceleratorState state = states[i];

                tasks[i] = Task.Run(() =>
                {
                    OptimizedGpuSort(chunk, state);
                    sortedChunks[i] = chunk;
                });
            }

            Task.WaitAll(tasks);

            int[] merged = KWayMerge(sortedChunks);
            Array.Copy(merged, array, length);
        }

        /// <summary>
        /// k-way merge of multiple sorted arrays.
        /// </summary>
        private static int[] KWayMerge(int[][] chunks)
        {
            int total = 0;
            for (int i = 0; i < chunks.Length; i++)
                total += chunks[i].Length;

            int[] merged = new int[total];
            var pq = new PriorityQueue<(int value, int chunkIndex, int indexInChunk), int>();

            for (int i = 0; i < chunks.Length; i++)
            {
                var chunk = chunks[i];
                if (chunk.Length > 0)
                {
                    pq.Enqueue((chunk[0], i, 0), chunk[0]);
                }
            }

            int pos = 0;
            while (pq.Count > 0)
            {
                var (value, chunkIndex, indexInChunk) = pq.Dequeue();
                merged[pos++] = value;

                int nextIdx = indexInChunk + 1;
                var chunk = chunks[chunkIndex];
                if (nextIdx < chunk.Length)
                {
                    int nextVal = chunk[nextIdx];
                    pq.Enqueue((nextVal, chunkIndex, nextIdx), nextVal);
                }
            }

            return merged;
        }

        /// <summary>
        /// Swaps two MemoryBuffer1D references.
        /// </summary>
        private static void SwapBuffers(ref MemoryBuffer1D<int, Stride1D.Dense> a,
                                        ref MemoryBuffer1D<int, Stride1D.Dense> b)
        {
            var temp = a;
            a = b;
            b = temp;
        }

        // --------- GPU kernels (ILGPU) ---------

        private static void ExclusiveScanKernelImpl(Index1D index, ArrayView<int> histogram, ArrayView<int> prefix)
        {
            if (index == 0)
            {
                int sum = 0;
                for (int i = 0; i < histogram.Length; i++)
                {
                    prefix[i] = sum;
                    sum += histogram[i];
                }
            }
        }

        private static void HistogramKernelImpl(Index1D index, ArrayView<int> input, int shift, ArrayView<int> histogram)
        {
            if (index < input.Length)
            {
                int value = input[index];
                int digit = (value >> shift) & 0xFF;

                // ILGPU will map this to a device-side atomic. If needed,
                // we can replace it with ILGPU.Algorithms.Atomic.Add.
                Interlocked.Add(ref histogram[digit], 1);
            }
        }

        private static void ScatterKernelImpl(Index1D index, ArrayView<int> input, int shift, ArrayView<int> prefix, ArrayView<int> output)
        {
            if (index < input.Length)
            {
                int value = input[index];
                int digit = (value >> shift) & 0xFF;
                int pos = Interlocked.Add(ref prefix[digit], 1);
                output[pos] = value;
            }
        }
    }

    // --------- FPGA interop ---------

    public static class FPGASortInterop
    {
        [DllImport("FpgaSortLib.dll", EntryPoint = "fpgaSort", CallingConvention = CallingConvention.Cdecl)]
        public static extern void FpgaSort([In, Out] uint[] array);
    }

    // --------- SmartSort orchestration (FPGA / GPU / CPU) ---------

    public static class SmartSortExtensions
    {
        /// <summary>
        /// SmartSort for List&lt;T&gt; chooses between FPGA (if flagged and valid) or CPU sort.
        /// </summary>
        public static void SmartSort<T>(this List<T> list, bool useFpga = false)
            where T : IComparable<T>
        {
            if (useFpga && typeof(T) == typeof(int))
            {
                Span<T> spanT = CollectionsMarshal.AsSpan(list);
                int length = spanT.Length;
                if (length == 1024)
                {
                    // Convert to uint, call FPGA, convert back in-place.
                    uint[] uArray = new uint[length];
                    for (int i = 0; i < length; i++)
                        uArray[i] = (uint)(int)(object)spanT[i];

                    FPGASortInterop.FpgaSort(uArray);

                    for (int i = 0; i < length; i++)
                        spanT[i] = (T)(object)(int)uArray[i];

                    return;
                }
            }

            // Fallback to CPU adaptive sort.
            list.AdaptiveSort();
        }

        /// <summary>
        /// SmartSort for arrays chooses between FPGA (if flagged and valid), GPU (for int[]), or CPU sort.
        /// </summary>
        public static void SmartSort<T>(this T[] array, bool useFpga = false)
            where T : IComparable<T>
        {
            int length = array.Length;

            // FPGA path for int[1024].
            if (useFpga && typeof(T) == typeof(int) && array is int[] intArrayFpga && length == 1024)
            {
                uint[] uArray = new uint[length];
                for (int i = 0; i < length; i++)
                    uArray[i] = (uint)intArrayFpga[i];

                FPGASortInterop.FpgaSort(uArray);

                for (int i = 0; i < length; i++)
                    intArrayFpga[i] = (int)uArray[i];

                return;
            }

            // GPU path for int[].
            if (typeof(T) == typeof(int) && array is int[] intArray)
            {
                try
                {
                    intArray.AdaptiveGpuSort();
                    return;
                }
                catch
                {
                    // Fall back to CPU on any GPU/ILGPU failure.
                }
            }
            // CPU fallback.
            array.AdaptiveSort();
        }
    }
}
