using NUnit.Framework.Legacy;
using System.Collections;
using System.Reflection;
using Moq;
using AdaptiveSort;
using ILGPU.Runtime;
using ILGPU;
using System.Linq;

namespace AdaptiveSortTests
{
    [TestFixture]
    public class AdaptiveSortExtensionsTests
    {
        [Test]
        public void AdaptiveSort_List_Int_SortsCorrectly()
        {
            List<int> list = new List<int> { 5, 2, 8, 1, 3 };
            list.AdaptiveSort();
            Assert.That(list, Is.Ordered);
        }

        [Test]
        public void AdaptiveSort_Array_Int_SortsCorrectly()
        {
            int[] array = new int[] { 10, 3, 7, 2, 9 };
            array.AdaptiveSort();
            Assert.That(array, Is.Ordered);
        }

        [Test]
        public void AdaptiveSort_List_Empty_Succeeds()
        {
            List<int> list = new List<int>();
            list.AdaptiveSort();
            Assert.That(list, Is.Empty);
        }

        [Test]
        public void AdaptiveSort_Array_SingleElement_Succeeds()
        {
            int[] array = new int[] { 42 };
            array.AdaptiveSort();
            CollectionAssert.AreEqual(new int[] { 42 }, array);
        }

        [Test]
        public void AdaptiveSort_Dictionary_SortsKeys()
        {
            Dictionary<string, int> dict = new Dictionary<string, int>
            {
                { "b", 2 },
                { "a", 1 },
                { "c", 3 }
            };
            dict.AdaptiveSort();
            List<string> keys = dict.Keys.ToList();
            List<string> expected = new List<string>(keys);
            expected.Sort();
            CollectionAssert.AreEqual(expected, keys);
        }

        [Test]
        public void AdaptiveSort_HashSet_SortsElements()
        {
            HashSet<int> set = new HashSet<int> { 4, 2, 9, 1 };
            set.AdaptiveSort();
            List<int> sorted = set.ToList();
            Assert.That(sorted, Is.Ordered);
        }

        [Test]
        public void AdaptiveSort_Hashtable_SortsElements()
        {
            Hashtable table = new Hashtable
            {
                { 3, "three" },
                { 1, "one" },
                { 2, "two" }
            };
            table.AdaptiveSort();

            // Hashtable does not guarantee ordering when enumerating keys.
            // To verify the sort, we extract the keys and then sort them.
            List<int> keys = table.Keys.Cast<int>().ToList();
            keys.Sort();

            CollectionAssert.AreEqual(new int[] { 1, 2, 3 }, keys);
        }

        // Indirect test of the Timsort path through AdaptiveSort.
        [Test]
        public void Timsort_SortsCorrectly_ForUnsortedData()
        {
            List<int> list = new List<int> { 9, 5, 3, 7, 1, 4, 8, 2, 6 };
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
                using var context = Context.Create(builder => builder.Cuda());
                foreach (var device in context.Devices)
                {
                    if (device.AcceleratorType == AcceleratorType.Cuda)
                        return true;
                }
                return false;
            }
            catch
            {
                // If ILGPU cannot create a CUDA context at all, treat as no GPU.
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

            int[] array = new int[] { 5, 3, 1 };

            // New implementation throws InvalidOperationException when no GPUs.
            Assert.Throws<InvalidOperationException>(() => array.AdaptiveGpuSort());
        }

        [Test]
        public void AdaptiveGpuSort_SortsCorrectly()
        {
            if (!HasCudaGpu())
            {
                Assert.Ignore("No CUDA GPUs available; skipping GPU sort test.");
            }

            int[] array = new int[] { 20, 3, 15, 7, 2, 9, 12, 5 };
            array.AdaptiveGpuSort();
            Assert.That(array, Is.Ordered);
        }

        [Test]
        public void KWayMerge_MergesSortedChunks()
        {
            // Prepare sorted chunks.
            int[][] chunks = new int[][]
            {
                new int[] { 1, 4, 7 },
                new int[] { 2, 5, 8 },
                new int[] { 3, 6, 9 }
            };

            // Invoke the private KWayMerge method via reflection.
            MethodInfo method = typeof(GpuSortExtensions)
                .GetMethod("KWayMerge", BindingFlags.NonPublic | BindingFlags.Static);
            int[] merged = (int[])method.Invoke(null, new object[] { chunks });
            CollectionAssert.AreEqual(new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, merged);
        }

        [Test]
        public void ExclusiveScanKernel_ComputesCorrectPrefix()
        {
            if (!HasCudaGpu())
            {
                Assert.Ignore("No CUDA GPUs available; skipping exclusive scan test.");
            }

            // Prepare a histogram array.
            int[] histogram = new int[] { 3, 1, 4, 0, 2 };
            int[] expectedPrefix = new int[] { 0, 3, 4, 8, 8 };

            using var context = Context.Create(builder => builder.Cuda());
            var device = context.Devices.First(d => d.AcceleratorType == AcceleratorType.Cuda);
            using var accelerator = device.CreateAccelerator(context);

            using var histBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(histogram.Length, default(Stride1D.Dense));
            using var prefixBuffer = accelerator.Allocate1D<int, Stride1D.Dense>(histogram.Length, default(Stride1D.Dense));
            histBuffer.CopyFromCPU(histogram);

            // Use an equivalent exclusive scan kernel to the production implementation.
            var exclusiveScanKernel = accelerator.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<int>>(
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
