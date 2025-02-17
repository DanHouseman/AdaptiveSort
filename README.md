## Overview
This adaptive sorting library intelligently selects among various sorting algorithms and hardware accelerators. Its design, backed by mathematically grounded performance models, offers efficient CPU-based sorting (via Timsort with adaptive merging), GPU-accelerated radix sort with multi-device support, and even FPGA integration via SmartSort. The “How To Use” section provides clear examples for integrating these capabilities into your projects, while the design adheres to modern software engineering principles (SOLID, DRY) ensuring extensibility and maintainability.

The adaptive sorting library is designed to select the optimal sorting algorithm (and hardware backend) based on the input data’s type and size. It is divided into three main components:

1. **CPU Sorting (AdaptiveSortExtensions):**  
   Implements adaptive sorting for common collection types (lists, arrays, dictionaries, hash sets, and hash tables) using a combination of insertion sort, Timsort (a hybrid merge sort), and built‑in sorting when appropriate.

2. **GPU Sorting (GpuSortExtensions):**  
   Uses ILGPU with CUDA support to perform a 4‑pass least-significant–digit (LSD) radix sort on int arrays. It supports single‑GPU and multi‑GPU modes, partitioning the data and merging sorted chunks using a k‑way merge.

3. **FPGA Integration and SmartSort (FPGASortInterop & SmartSortExtensions):**  
   Provides an interop layer to call an external FPGA sort function and exposes a SmartSort method that automatically chooses between FPGA, GPU, or CPU sorting.

---

## CPU Sorting – AdaptiveSortExtensions

### Insertion Sort

**Description:**  
Insertion sort iterates over the collection, inserting each element into its correct position within the already‑sorted portion.

**Mathematical Equation:**  
$C(n) = \frac{n(n-1)}{2} \quad \text{and} \quad T(n) \in O(n^2)$

---

### Timsort

**Description:**  
Timsort divides the collection into small runs (typically below a threshold of 32 elements), sorts each run using insertion sort, and then efficiently merges these runs.

**Mathematical Estimates:**  
- **Run sorting:**  
$T_{\text{runs}}(n) \in O\left(\frac{n}{r} \cdot r^2\right) = O(nr)$
- **Merging:**  
$T_{\text{merge}}(n) \in O\left(n \log \frac{n}{r}\right)$
- **Overall:**  
$T(n) \in O(n \log n)$

---

### Merge Operation

The merge operation is implemented in two variants:
- **Unmanaged Merge:**  
  When $T$ is an unmanaged type (with no managed references), a stack‑allocated temporary buffer is used.  
  *Time Complexity:* $O(n)$ (approximately $n-1$ comparisons in the worst case).
- **Managed Merge:**  
  For types that cannot use stack allocation, a temporary heap array is allocated.

---

## GPU Sorting – GpuSortExtensions

### GPU Radix Sort

**Description:**  
A 4‑pass LSD radix sort is implemented on the GPU:
- **Histogram Kernel:** For each element, the algorithm extracts an 8‑bit digit and atomically increments a corresponding bucket in a 256‑element histogram.
- **Exclusive Scan Kernel:** Computes a prefix sum on the histogram directly on the GPU (using a single thread, given the small problem size).
- **Scatter Kernel:** Uses the prefix to scatter elements into their correct positions in an output buffer.

**Mathematical Model:**  
- **Per-pass cost:**  
$T_{\text{pass}}(n) \in O(n)$
- **Total (4 passes):**  
  $T(n) = 4 \cdot O(n) \in O(n)$

### K‑Way Merge for Multi‑GPU

**Description:**  
When multiple GPUs are available, the array is partitioned and sorted on each device independently. A k‑way merge (using a priority queue) then combines these sorted chunks.

**Mathematical Complexity:**
$T(n, k) \in O(n \log k)$
  
---

## FPGA Integration and SmartSort

### FPGA Sort

**Description:**  
A thin P/Invoke layer (FPGASortInterop) calls a native FPGA sort function. Although the FPGA implementation is external, many FPGA sorts leverage massive parallelism.

**Theoretical Complexity (if parallelized):**  
$T(n) \in O\left(\frac{n}{p} + \log n\right)$
where $p$ is the number of parallel processing units.

### SmartSort

**Description:**  
SmartSort examines the type and size of the data:
- For int arrays of exactly 1024 elements and if FPGA sorting is enabled, it calls the FPGA sort.
- For int data when GPUs are available, it uses the GPU radix sort.
- Otherwise, it falls back to the adaptive CPU sort.

This approach abstracts hardware decisions away from the client.

---

## How To Use

### CPU Sorting

- **Sort a List:**  
  ```csharp
  List<int> numbers = new List<int> { 5, 2, 9, 1, 3 };
  numbers.AdaptiveSort();  // Uses insertion sort, built-in sort, or Timsort as appropriate.
  ```

- **Sort an Array:**  
  ```csharp
  int[] data = { 10, 3, 7, 2, 9 };
  data.AdaptiveSort();
  ```

- **Sort a Dictionary by Keys:**  
  ```csharp
  Dictionary<string, int> dict = new Dictionary<string, int>
  {
      { "b", 2 },
      { "a", 1 },
      { "c", 3 }
  };
  dict.AdaptiveSort();
  ```

- **Sort a HashSet:**  
  ```csharp
  HashSet<int> set = new HashSet<int> { 4, 2, 9, 1 };
  set.AdaptiveSort();
  ```

- **Sort a Hashtable:**  
  ```csharp
  Hashtable table = new Hashtable
  {
      { 3, "three" },
      { 1, "one" },
      { 2, "two" }
  };
  table.AdaptiveSort();
  // Note: The keys are reinserted in sorted order.
  ```

### GPU Sorting

- **Sort an int Array on the GPU:**  
  ```csharp
  int[] gpuData = { 20, 3, 15, 7, 2, 9, 12, 5 };
  gpuData.AdaptiveGpuSort();  // Automatically selects the GPU radix sort.
  ```

*Note:* GPU methods require an available CUDA-enabled device. If no GPUs are detected, an exception is thrown or the code falls back to CPU sorting in SmartSort.

### FPGA & SmartSort

- **SmartSort a List:**  
  ```csharp
  List<int> smartList = new List<int> { 10, 5, 7, 3, 8 };
  // If FPGA sort conditions are met, it is used; otherwise, GPU or CPU sort is applied.
  smartList.SmartSort(useFpga: true);
  ```

- **SmartSort an Array:**  
  ```csharp
  int[] smartArray = { 10, 5, 7, 3, 8 };
  smartArray.SmartSort(useFpga: true);
  ```

---

## Mathematical Equations Summary

- **Insertion Sort:**  
$C(n) = \frac{n(n-1)}{2} \quad T(n) \in O(n^2)$

- **Timsort:**  
$T_{\text{runs}}(n) \in O(nr), \quad T_{\text{merge}}(n) \in O\left(n \log \frac{n}{r}\right), \quad T(n) \in O(n \log n)$

- **Merge Operation:**  
  $T_{\text{merge}}(n) \in O(n)$

- **GPU Radix Sort (4 passes):**  
  $T(n) = 4 \cdot O(n) \in O(n)$

- **K‑Way Merge:**  
$T(n, k) \in O(n \log k)$

- **FPGA Sort (Parallel Model):**  
$T(n) \in O\left(\frac{n}{p} + \log n\right)$

---

## Efficiency and Design Principles

- **CPU Efficiency:**  
  - For small collections, insertion sort minimizes overhead.
  - Timsort minimizes overall comparisons by combining insertion sort (for runs) and merge sort.
  - Unmanaged merges use stack allocation to reduce heap pressure.

- **GPU Efficiency:**  
  - Cached ILGPU contexts and accelerators reduce initialization overhead.
  - Buffer reuse and an on‑device exclusive scan reduce host–device communication.
  - Multi-GPU partitioning and a k‑way merge maximize throughput.

- **FPGA Integration:**  
  - The P/Invoke wrapper provides a path for highly parallel FPGA sorting.

- **SOLID and DRY:**  
  - Each module has a single responsibility: CPU sorting, GPU sorting, or hardware integration.
  - Common code (e.g., insertion sort, merge, k‑way merge) is reused.
  - The library is designed for easy extension, allowing new hardware backends without altering core logic.

