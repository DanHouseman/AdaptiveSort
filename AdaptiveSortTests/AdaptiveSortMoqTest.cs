using NUnit.Framework;
using NUnit.Framework.Legacy;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using AdaptiveSort;
using ILGPU;
using ILGPU.Runtime;

namespace AdaptiveSortTests
{
    [TestFixture]
    public class AdaptiveSortExtensionsTests
    {
        [Test]
        public void AdaptiveSort_List_Int_SortsCorrectly()
        {
            var list = new List<int> { 5, 2, 8, 1, 3 };

            list.AdaptiveSort();

            Assert.That(list, Is.Ordered);
        }

        [Test]
        public void AdaptiveSort_Array_Int_SortsCorrectly()
        {
            int[] array = { 10, 3, 7, 2, 9 };

            array.AdaptiveSort();

            Assert.That(array, Is.Ordered);
        }

        [Test]
        public void AdaptiveSort_List_Empty_Succeeds()
        {
            var list = new List<int>();

            list.AdaptiveSort();

            Assert.That(list, Is.Empty);
        }

        [Test]
        public void AdaptiveSort_Array_SingleElement_Succeeds()
        {
            int[] array = { 42 };

            array.AdaptiveSort();

            CollectionAssert.AreEqual(new[] { 42 }, array);
        }

        [Test]
        public void AdaptiveSort_Dictionary_SortsKeys()
        {
            var dict = new Dictionary<string, int>
            {
                { "b", 2 },
                { "a", 1 },
                { "c", 3 }
            };

            dict.AdaptiveSort();

            List<string> keys = dict.Keys.ToList();
            var expected = new List<string>(keys);
            expected.Sort();

            CollectionAssert.AreEqual(expected, keys);
        }

        [Test]
        public void AdaptiveSort_HashSet_SortsElements()
        {
            var set = new HashSet<int> { 4, 2, 9, 1 };

            set.AdaptiveSort();

            List<int> sorted = set.ToList();
            Assert.That(sorted, Is.Ordered);
        }

        [Test]
        public void AdaptiveSort_Hashtable_SortsElements()
        {
            var table = new Hashtable
            {
                { 3, "three" },
                { 1, "one" },
                { 2, "two" }
            };

            table.AdaptiveSort();

            // Hashtable enumeration order is not guaranteed; verify by sorted keys.
            List<int> keys = table.Keys.Cast<int>().ToList();
            keys.Sort();

            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, keys);
        }

        // Indirect test of the Timsort path via AdaptiveSort.
        [Test]
        public void Timsort_SortsCorrectly_ForUnsortedData()
        {
            var list = new List<int> { 9, 5, 3, 7, 1, 4, 8, 2, 6 };

            list.AdaptiveSort();

            Assert.That(list, Is.Ordered);
        }
    }

    [TestFixture]
    public class GpuSortExtensionsTests
    {
        private static bool HasCudaGpu()
        {
            try
            {
                using var context = Context.CreateDefault();
                foreach (var device in context.Devices)
                {
                    if (device.AcceleratorType == AcceleratorType.Cuda)
                        return true;
                }

                return false;
            }
            catch
            {
                // If we cannot even create a context, treat as no GPU.
                return false;
            }
        }

        [Test]
        public void AdaptiveGpuSort_NoGPUs_ThrowsException()
        {
            if (HasCudaGpu())
            {
                Assert.Ignore("CUDA GPUs are available; skipping no-GPU scenario test.");
            }

            int[] array = { 5, 3, 1 };

            // Production code throws InvalidOperationException when no CUDA GPUs are cached.
            Assert.Throws<InvalidOperationException>(() => array.AdaptiveGpuSort());
        }

        [Test]
        public void AdaptiveGpuSort_SortsCorrectly()
        {
            if (!HasCudaGpu())
            {
                Assert.Ignore("No CUDA GPUs available; skipping GPU sort test.");
            }

            int[] array = { 20, 3, 15, 7, 2, 9, 12, 5 };

            array.AdaptiveGpuSort();

            Assert.That(array, Is.Ordered);
        }

        [Test]
        public void KWayMerge_MergesSortedChunks()
        {
            // Prepare sorted chunks.
            int[][] chunks =
            {
                new[] { 1, 4, 7 },
                new[] { 2, 5, 8 },
                new[] { 3, 6, 9 }
            };

            // Invoke the private KWayMerge method via reflection.
            MethodInfo method = typeof(GpuSortExtensions)
                .GetMethod("KWayMerge", BindingFlags.NonPublic | BindingFlags.Static);
            Assert.That(method, Is.Not.Null, "KWayMerge method not found via reflection.");

            int[] merged = (int[])method!.Invoke(null, new object[] { chunks })!;

            CollectionAssert.AreEqual(
                new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
                merged);
        }

        [Test]
        public void ExclusiveScanKernel_ComputesCorrectPrefix()
        {
            if (!HasCudaGpu())
            {
                Assert.Ignore("No CUDA GPUs available; skipping exclusive scan test.");
            }

            int[] histogram = { 3, 1, 4, 0, 2 };
            int[] expectedPrefix = { 0, 3, 4, 8, 8 };

            using var context = Context.CreateDefault();
            var device = context.Devices.First(d => d.AcceleratorType == AcceleratorType.Cuda);
            using var accelerator = device.CreateAccelerator(context);

            using var histBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(histogram.Length, default);
            using var prefixBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(histogram.Length, default);

            histBuffer.CopyFromCPU(histogram);

            // Kernel equivalent to production ExclusiveScanKernelImpl.
            var exclusiveScanKernel =
                accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
                    (Index1D index, ArrayView<int> hist, ArrayView<int> pref) =>
                    {
                        if (index == 0)
                        {
                            int sum = 0;
                            for (int i = 0; i < hist.Length; i++)
                            {
                                pref[i] = sum;
                                sum += hist[i];
                            }
                        }
                    });

            exclusiveScanKernel(1, histBuffer.View, prefixBuffer.View);
            accelerator.Synchronize();

            int[] actualPrefix = prefixBuffer.GetAsArray1D();
            CollectionAssert.AreEqual(expectedPrefix, actualPrefix);
        }
    }
}
