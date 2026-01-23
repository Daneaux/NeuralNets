using MatrixLibrary;
using MatrixLibrary.Gpu;

namespace CublasTests
{
    [TestClass]
    public sealed class CudaContextTests
    {
        [TestMethod]
        public void TestIsGpuAvailable_DoesNotThrow()
        {
            // Should never throw — returns true or false
            bool available = BackendSelector.IsGPUAvailable();
            // Just verify it returns without exception
            Assert.IsTrue(available || !available);
        }

        [TestMethod]
        public void TestCudaContext_InitializesCorrectly()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available.");

            var ctx = CudaContext.Instance;
            Assert.IsTrue(ctx.IsInitialized);
            Assert.AreNotEqual(IntPtr.Zero, ctx.CublasHandle);
        }

        [TestMethod]
        public void TestCudaContext_Singleton()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available.");

            var ctx1 = CudaContext.Instance;
            var ctx2 = CudaContext.Instance;
            Assert.AreSame(ctx1, ctx2);
        }

        [TestMethod]
        public void TestMemoryRoundTrip_1D()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available.");

            float[] original = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
            IntPtr devicePtr = GpuMemoryManager.AllocateAndUpload(original);

            try
            {
                float[] downloaded = new float[5];
                GpuMemoryManager.CopyFromDevice(devicePtr, downloaded, 5);

                CollectionAssert.AreEqual(original, downloaded);
            }
            finally
            {
                GpuMemoryManager.Free(devicePtr);
            }
        }

        [TestMethod]
        public void TestMemoryRoundTrip_2D()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available.");

            float[,] original = { { 1, 2, 3 }, { 4, 5, 6 } };
            IntPtr devicePtr = GpuMemoryManager.AllocateAndUpload(original, 2, 3);

            try
            {
                float[,] downloaded = new float[2, 3];
                GpuMemoryManager.CopyFromDevice(devicePtr, downloaded, 2, 3);

                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 3; c++)
                        Assert.AreEqual(original[r, c], downloaded[r, c]);
            }
            finally
            {
                GpuMemoryManager.Free(devicePtr);
            }
        }

        [TestMethod]
        public void TestMemoryRoundTrip_LargeArray()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available.");

            const int size = 100000;
            float[] original = new float[size];
            var rnd = new Random(42);
            for (int i = 0; i < size; i++)
                original[i] = (float)(rnd.NextDouble() * 200 - 100);

            IntPtr devicePtr = GpuMemoryManager.AllocateAndUpload(original);

            try
            {
                float[] downloaded = new float[size];
                GpuMemoryManager.CopyFromDevice(devicePtr, downloaded, size);

                for (int i = 0; i < size; i++)
                    Assert.AreEqual(original[i], downloaded[i], 0f,
                        $"Mismatch at [{i}]");
            }
            finally
            {
                GpuMemoryManager.Free(devicePtr);
            }
        }

        [TestMethod]
        public void TestFreeZeroPointer_DoesNotThrow()
        {
            // Freeing IntPtr.Zero should be a no-op
            GpuMemoryManager.Free(IntPtr.Zero);
        }

        [TestMethod]
        public void TestBackendSelector_ReportsGpu()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available.");

            // With GPU available, recommended backend for large matrices should be GPU
            var backend = BackendSelector.GetRecommendedBackendForSize(100000);
            Assert.AreEqual(MatrixBackend.GPU, backend);
        }
    }
}
