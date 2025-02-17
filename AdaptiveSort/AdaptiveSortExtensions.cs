using System.Collections;
using System.ComponentModel.DataAnnotations;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace AdaptiveSort
{
    public static class AdaptiveSortExtensions
    {
        private const int INSERTION_SORT_THRESHOLD = 32;

        public static void AdaptiveSort<T>(this List<T> list) where T : IComparable<T>
        {
            if (list.Count < INSERTION_SORT_THRESHOLD)
                InsertionSort(list, 0, list.Count - 1);
            else if (list.Count > 100_000)
                list.Sort();
            else
                Timsort(list);
        }

        public static void AdaptiveSort<T>(this T[] array) where T : IComparable<T>
        {
            if (array.Length < INSERTION_SORT_THRESHOLD)
                InsertionSort(array, 0, array.Length - 1);
            else if (array.Length > 100_000)
                Array.Sort(array);
            else
                Timsort(array);
        }

        public static void AdaptiveSort<TKey, TValue>(this Dictionary<TKey, TValue> dict) where TKey : IComparable<TKey>
        {
            var sortedKeys = dict.Keys.ToList();
            sortedKeys.AdaptiveSort();
            var sortedDict = sortedKeys.ToDictionary(key => key, key => dict[key]);
            dict.Clear();
            foreach (var kvp in sortedDict)
                dict.Add(kvp.Key, kvp.Value);
        }

        public static void AdaptiveSort<T>(this HashSet<T> hashSet) where T : IComparable<T>
        {
            var sorted = hashSet.ToList();
            sorted.AdaptiveSort();
            hashSet.Clear();
            foreach (var item in sorted)
                hashSet.Add(item);
        }

        public static void AdaptiveSort(this Hashtable hashtable)
        {
            var sortedKeys = hashtable.Keys.Cast<object>().ToList();
            sortedKeys.Sort();
            Hashtable sortedTable = new Hashtable();
            foreach (var key in sortedKeys)
                sortedTable[key] = hashtable[key];
            hashtable.Clear();
            foreach (DictionaryEntry entry in sortedTable)
                hashtable.Add(entry.Key, entry.Value);
        }
        /// <summary>
        /// MergeInPlace dispatches to an unmanaged-optimized version if possible.
        /// </summary>
        internal static void MergeInPlace<T>(IList<T> array, int left, int mid, int right)
            where T : IComparable<T>
        {
            if (IsUnmanaged<T>())
            {
                // Use reflection to call the unmanaged version.
                MethodInfo method = typeof(AdaptiveSortExtensions)
                    .GetMethod(nameof(MergeInPlace_UnmanagedCore), BindingFlags.NonPublic | BindingFlags.Static)
                    ?? throw new InvalidOperationException("Could not locate MergeInPlace_UnmanagedCore.");

                MethodInfo generic = method.MakeGenericMethod(typeof(T));
                generic.Invoke(null, new object[] { array, left, mid, right });
            }
            else
            {
                MergeInPlace_Managed(array, left, mid, right);
            }
        }

        /// <summary>
        /// Returns true if T is a value type that does not contain any managed references.
        /// </summary>
        private static bool IsUnmanaged<T>()
        {
            return typeof(T).IsValueType && !RuntimeHelpers.IsReferenceOrContainsReferences<T>();
        }

        /// <summary>
        /// Unmanaged version of merge using stack allocation.
        /// </summary>
        private static void MergeInPlace_UnmanagedCore<T>(IList<T> array, int left, int mid, int right) where T : unmanaged, IComparable<T>
        {
            int leftSize = mid - left + 1;
            const int STACKALLOC_THRESHOLD = 128;
            Span<T> leftSpan = leftSize <= STACKALLOC_THRESHOLD
                ? stackalloc T[leftSize]
                : new T[leftSize];

            for (int i = 0; i < leftSize; i++)
                leftSpan[i] = array[left + i];

            int iLeft = 0, iRight = mid + 1, k = left;
            while (iLeft < leftSize && iRight <= right)
            {
                array[k++] = leftSpan[iLeft].CompareTo(array[iRight]) <= 0
                    ? leftSpan[iLeft++]
                    : array[iRight++];
            }
            while (iLeft < leftSize)
                array[k++] = leftSpan[iLeft++];
        }


        /// <summary>
        /// Managed fallback merge using heap allocation.
        /// </summary>
        private static void MergeInPlace_Managed<T>(IList<T> array, int left, int mid, int right)
            where T : IComparable<T>
        {
            int leftSize = mid - left + 1;
            T[] leftArray = new T[leftSize];
            for (int i = 0; i < leftSize; i++)
                leftArray[i] = array[left + i];

            int iLeft = 0, iRight = mid + 1, k = left;
            while (iLeft < leftSize && iRight <= right)
            {
                array[k++] = leftArray[iLeft].CompareTo(array[iRight]) <= 0
                    ? leftArray[iLeft++]
                    : array[iRight++];
            }
            while (iLeft < leftSize)
                array[k++] = leftArray[iLeft++];
        }

        // Timsort calls MergeInPlace.
        private static void Timsort<T>(IList<T> array) where T : IComparable<T>
        {
            int n = array.Count;
            for (int i = 0; i < n; i += INSERTION_SORT_THRESHOLD)
                InsertionSort(array, i, Math.Min(i + INSERTION_SORT_THRESHOLD - 1, n - 1));

            for (int size = INSERTION_SORT_THRESHOLD; size < n; size *= 2)
            {
                for (int left = 0; left < n - 1; left += 2 * size)
                {
                    int mid = left + size - 1;
                    int right = Math.Min(left + 2 * size - 1, n - 1);
                    if (mid < right)
                        MergeInPlace(array, left, mid, right);
                }
            }
        }

        private static void InsertionSort<T>(IList<T> array, int left, int right) where T : IComparable<T>
        {
            for (int i = left + 1; i <= right; i++)
            {
                T key = array[i];
                int j = i - 1;
                while (j >= left && array[j].CompareTo(key) > 0)
                {
                    array[j + 1] = array[j];
                    j--;
                }
                array[j + 1] = key;
            }
        }
    }


    public static class GpuSortExtensions
    {
        // Cached ILGPU context and accelerator instances.
        private static readonly Context s_context;
        private static readonly Accelerator[] s_accelerators;

        static GpuSortExtensions()
        {
            // Create a single ILGPU context with CUDA support.
            s_context = Context.Create(builder => builder.Cuda());
            // Retrieve and cache all available CUDA devices.
            var devices = s_context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToArray();
            if (devices.Length == 0)
            {
                s_accelerators = Array.Empty<Accelerator>();
            }
            else
            {
                s_accelerators = new Accelerator[devices.Length];
                for (int i = 0; i < devices.Length; i++)
                {
                    s_accelerators[i] = s_context.CreateCudaAccelerator(i);
                }
            }
        }

        /// <summary>
        /// Performs an in-place GPU-based radix sort on an int array.
        /// </summary>
        public static void AdaptiveGpuSort(this int[] array)
        {
            if (s_accelerators.Length == 0)
                throw new Exception("No CUDA GPUs available.");

            if (s_accelerators.Length == 1)
                OptimizedGpuSort(array, s_accelerators[0]);
            else
                MultiGpuSort(array, s_accelerators);
        }

        /// <summary>
        /// Single-device radix sort using a cached accelerator.
        /// Implements a 4-pass (8 bits per pass) radix sort.
        /// </summary>
        private static void OptimizedGpuSort(int[] array, Accelerator accelerator)
        {
            int length = array.Length;
            // Allocate two buffers for in-place sorting.
            var inputBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(length, default(Stride1D.Dense));
            var outputBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(length, default(Stride1D.Dense));
            // Allocate (once) buffers for histogram (256 buckets) and its prefix.
            var histogramBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(256, default(Stride1D.Dense));
            var prefixBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(256, default(Stride1D.Dense));

            try
            {
                // Copy the host array into the device buffer.
                inputBuffer.CopyFromCPU(array);

                // Load the kernels.
                var histogramKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>>(HistogramKernel);
                var scatterKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, int, ArrayView<int>, ArrayView<int>>(ScatterKernel);
                var exclusiveScanKernel = accelerator.LoadAutoGroupedStreamKernel<
                    Index1D, ArrayView<int>, ArrayView<int>>(ExclusiveScanKernel);

                // Process 4 passes (for shift = 0, 8, 16, 24).
                for (int shift = 0; shift < 32; shift += 8)
                {
                    // Clear the histogram buffer.
                    histogramBuffer.MemSetToZero();

                    // Build the histogram for the current digit.
                    histogramKernel((int)inputBuffer.Length, inputBuffer.View, shift, histogramBuffer.View);
                    accelerator.Synchronize();

                    // Compute the exclusive scan (prefix sum) on the GPU.
                    // Launch with 1 thread since the problem is small.
                    exclusiveScanKernel(1, histogramBuffer.View, prefixBuffer.View);
                    accelerator.Synchronize();

                    // Scatter the elements into their new positions based on the prefix.
                    scatterKernel((int)inputBuffer.Length, inputBuffer.View, shift, prefixBuffer.View, outputBuffer.View);
                    accelerator.Synchronize();

                    // Swap input and output buffers for the next pass.
                    SwapBuffers(ref inputBuffer, ref outputBuffer);
                }

                // Copy the sorted result back to the host.
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
        /// Multi-device radix sort.
        /// Each accelerator sorts a chunk concurrently; then a k-way merge combines the sorted chunks.
        /// </summary>
        private static void MultiGpuSort(int[] array, Accelerator[] accelerators)
        {
            int length = array.Length;
            int chunkSize = length / accelerators.Length;
            var tasks = new Task[accelerators.Length];
            int[][] sortedChunks = new int[accelerators.Length][];

            for (int i = 0; i < accelerators.Length; i++)
            {
                int start = i * chunkSize;
                int end = (i == accelerators.Length - 1) ? length : start + chunkSize;
                int len = end - start;
                int[] chunk = new int[len];
                Array.Copy(array, start, chunk, 0, len);
                Accelerator accelerator = accelerators[i];

                tasks[i] = Task.Run(() =>
                {
                    OptimizedGpuSort(chunk, accelerator);
                    sortedChunks[i] = chunk;
                });
            }

            Task.WaitAll(tasks);

            // Merge the sorted chunks using a k-way merge.
            int[] merged = KWayMerge(sortedChunks);
            Array.Copy(merged, array, length);
        }

        /// <summary>
        /// Performs a k-way merge on multiple sorted arrays.
        /// </summary>
        private static int[] KWayMerge(int[][] chunks)
        {
            int total = chunks.Sum(c => c.Length);
            int[] merged = new int[total];
            var pq = new PriorityQueue<(int value, int chunkIndex, int indexInChunk), int>();

            // Enqueue the first element of each chunk.
            for (int i = 0; i < chunks.Length; i++)
            {
                if (chunks[i].Length > 0)
                    pq.Enqueue((chunks[i][0], i, 0), chunks[i][0]);
            }

            int pos = 0;
            while (pq.Count > 0)
            {
                var (value, chunkIndex, indexInChunk) = pq.Dequeue();
                merged[pos++] = value;
                if (indexInChunk + 1 < chunks[chunkIndex].Length)
                {
                    int nextVal = chunks[chunkIndex][indexInChunk + 1];
                    pq.Enqueue((nextVal, chunkIndex, indexInChunk + 1), nextVal);
                }
            }
            return merged;
        }

        /// <summary>
        /// A GPU kernel that computes an exclusive scan on a 1D array.
        /// Since the array is small (256 elements), a single thread performs a sequential scan.
        /// </summary>
        private static void ExclusiveScanKernel(Index1D index, ArrayView<int> histogram, ArrayView<int> prefix)
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

        /// <summary>
        /// Swaps two MemoryBuffer1D references.
        /// </summary>
        private static void SwapBuffers(ref MemoryBuffer1D<int, Stride1D.Dense> a, ref MemoryBuffer1D<int, Stride1D.Dense> b)
        {
            var temp = a;
            a = b;
            b = temp;
        }

        /// <summary>
        /// GPU histogram kernel.
        /// For each element in the input, extract the digit (using the provided shift) and atomically increment the corresponding histogram bucket.
        /// </summary>
        private static void HistogramKernel(Index1D index, ArrayView<int> input, int shift, ArrayView<int> histogram)
        {
            if (index < input.Length)
            {
                int value = input[index];
                int digit = (value >> shift) & 0xFF;
                // Atomically increment the histogram bucket.
                Interlocked.Add(ref histogram[digit], 1);
            }
        }

        /// <summary>
        /// GPU scatter kernel.
        /// For each element, determine its digit, then atomically fetch and increment the corresponding prefix value,
        /// and write the element into the output array at the computed position.
        /// </summary>
        private static void ScatterKernel(Index1D index, ArrayView<int> input, int shift, ArrayView<int> prefix, ArrayView<int> output)
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

    public static class FPGASortInterop
    {
        [DllImport("FpgaSortLib.dll", EntryPoint = "fpgaSort", CallingConvention = CallingConvention.Cdecl)]
        public static extern void FpgaSort([In, Out] uint[] array);
    }

    public static class SmartSortExtensions
    {
        /// <summary>
        /// SmartSort for List&lt;T&gt; chooses between FPGA (if flagged and valid), GPU (for int data) or CPU sort.
        /// </summary>
        public static void SmartSort<T>(this List<T> list, bool useFpga = false) where T : IComparable<T>
        {
            if (useFpga && typeof(T) == typeof(int))
            {
                var array = list.Cast<int>().ToArray();
                if (array.Length == 1024)
                {
                    // Convert to uint, call FPGA sort, then convert back.
                    uint[] uArray = Array.ConvertAll(array, x => (uint)x);
                    FPGASortInterop.FpgaSort(uArray);
                    int[] sorted = Array.ConvertAll(uArray, x => (int)x);
                    list.Clear();
                    list.AddRange(sorted.Cast<T>());
                    return;
                }
            }
            list.AdaptiveSort();
        }

        /// <summary>
        /// SmartSort for arrays chooses between FPGA (if flagged and valid), GPU (if int) or CPU sort.
        /// </summary>
        public static void SmartSort<T>(this T[] array, bool useFpga = false) where T : IComparable<T>
        {
            if (useFpga && typeof(T) == typeof(int) && array.Length == 1024)
            {
                uint[] uArray = Array.ConvertAll(array as int[], x => (uint)x);
                FPGASortInterop.FpgaSort(uArray);
                int[] sorted = Array.ConvertAll(uArray, x => (int)x);
                Array.Copy(sorted, array, sorted.Length);
                return;
            }
            if (typeof(T) == typeof(int) && array is int[] intArray)
            {
                try
                {
                    using var context = Context.Create(builder => builder.Cuda());
                    var devices = context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToArray();
                    if (devices.Length > 0)
                    {
                        intArray.AdaptiveGpuSort();
                        return;
                    }
                }
                catch { /* Fallback to CPU sort */ }
            }
            array.AdaptiveSort();
        }
    }
}
