using MatrixLibrary;

namespace CublasTests
{
    [TestClass]
    public sealed class GpuColumnVectorTests
    {
        private const float Tolerance = 1e-3f;

        [TestInitialize]
        public void CheckGpuAvailable()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available — skipping GPU tests.");
        }

        private static float[] MakeRandomArray(int size, int seed = 42)
        {
            var rnd = new Random(seed);
            var arr = new float[size];
            for (int i = 0; i < size; i++)
                arr[i] = (float)(rnd.NextDouble() * 20 - 10);
            return arr;
        }

        // ============================================================
        // Add (Vector)
        // ============================================================

        [TestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(57)]
        [DataRow(1024)]
        public void TestAdd_Vector(int size)
        {
            var dataA = MakeRandomArray(size, 42);
            var dataB = MakeRandomArray(size, 43);

            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);
            var swA = new ColumnVector(dataA);
            var swB = new ColumnVector(dataB);

            var gpuResult = gpuA.Add(gpuB);
            var swResult = swA.Add(swB);

            Assert.AreEqual(swResult.Size, gpuResult.Size);
            for (int i = 0; i < size; i++)
                Assert.AreEqual(swResult[i], gpuResult[i], Tolerance,
                    $"Mismatch at [{i}]");
        }

        [TestMethod]
        public void TestAdd_Vector_KnownValues()
        {
            var a = new GpuColumnVector(new float[] { 1, 2, 3, 4 });
            var b = new GpuColumnVector(new float[] { 5, 6, 7, 8 });

            var result = a.Add(b);

            Assert.AreEqual(6f, result[0], Tolerance);
            Assert.AreEqual(8f, result[1], Tolerance);
            Assert.AreEqual(10f, result[2], Tolerance);
            Assert.AreEqual(12f, result[3], Tolerance);
        }

        // ============================================================
        // Subtract (Vector)
        // ============================================================

        [TestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(57)]
        [DataRow(1024)]
        public void TestSubtract_Vector(int size)
        {
            var dataA = MakeRandomArray(size, 42);
            var dataB = MakeRandomArray(size, 43);

            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);
            var swA = new ColumnVector(dataA);
            var swB = new ColumnVector(dataB);

            var gpuResult = gpuA.Subtract(gpuB);
            var swResult = swA.Subtract(swB);

            for (int i = 0; i < size; i++)
                Assert.AreEqual(swResult[i], gpuResult[i], Tolerance,
                    $"Mismatch at [{i}]");
        }

        [TestMethod]
        public void TestSubtract_Vector_KnownValues()
        {
            var a = new GpuColumnVector(new float[] { 5, 6, 7, 8 });
            var b = new GpuColumnVector(new float[] { 1, 2, 3, 4 });

            var result = a.Subtract(b);

            Assert.AreEqual(4f, result[0], Tolerance);
            Assert.AreEqual(4f, result[1], Tolerance);
            Assert.AreEqual(4f, result[2], Tolerance);
            Assert.AreEqual(4f, result[3], Tolerance);
        }

        // ============================================================
        // Multiply (Scalar)
        // ============================================================

        [TestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(57)]
        [DataRow(1024)]
        public void TestMultiply_Scalar(int size)
        {
            var data = MakeRandomArray(size, 42);
            float scalar = 3.5f;

            var gpuVec = new GpuColumnVector(data);
            var avxVec = new AvxColumnVector(data);

            var gpuResult = gpuVec.Multiply(scalar);
            var avxResult = avxVec.Multiply(scalar);

            for (int i = 0; i < size; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance,
                    $"Mismatch at [{i}]");
        }

        [TestMethod]
        public void TestMultiply_Scalar_KnownValues()
        {
            var a = new GpuColumnVector(new float[] { 1, 2, 3, 4 });
            var result = a.Multiply(2.5f);

            Assert.AreEqual(2.5f, result[0], Tolerance);
            Assert.AreEqual(5.0f, result[1], Tolerance);
            Assert.AreEqual(7.5f, result[2], Tolerance);
            Assert.AreEqual(10.0f, result[3], Tolerance);
        }

        // ============================================================
        // Multiply (Element-wise vector) — fallback
        // ============================================================

        [TestMethod]
        [DataRow(16)]
        [DataRow(57)]
        public void TestMultiply_Elementwise(int size)
        {
            var dataA = MakeRandomArray(size, 42);
            var dataB = MakeRandomArray(size, 43);

            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.Multiply(gpuB);
            var avxResult = avxA.Multiply(avxB);

            for (int i = 0; i < size; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance,
                    $"Mismatch at [{i}]");
        }

        // ============================================================
        // Add (Scalar) — fallback
        // ============================================================

        [TestMethod]
        public void TestAdd_Scalar()
        {
            var data = MakeRandomArray(16, 42);
            float scalar = 5.5f;

            var gpuVec = new GpuColumnVector(data);
            var avxVec = new AvxColumnVector(data);

            var gpuResult = gpuVec.Add(scalar);
            var avxResult = avxVec.Add(scalar);

            for (int i = 0; i < 16; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance);
        }

        // ============================================================
        // Subtract (Scalar) — fallback
        // ============================================================

        [TestMethod]
        public void TestSubtract_Scalar()
        {
            var data = MakeRandomArray(16, 42);
            float scalar = 3.0f;

            var gpuVec = new GpuColumnVector(data);
            var avxVec = new AvxColumnVector(data);

            var gpuResult = gpuVec.Subtract(scalar);
            var avxResult = avxVec.Subtract(scalar);

            for (int i = 0; i < 16; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance);
        }

        // ============================================================
        // Sum — fallback
        // ============================================================

        [TestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(1024)]
        public void TestSum(int size)
        {
            var data = MakeRandomArray(size, 42);

            var gpuVec = new GpuColumnVector(data);
            var avxVec = new AvxColumnVector(data);

            float gpuSum = gpuVec.Sum();
            float avxSum = avxVec.Sum();

            Assert.AreEqual(avxSum, gpuSum, Tolerance);
        }

        [TestMethod]
        public void TestSum_KnownValues()
        {
            var vec = new GpuColumnVector(new float[] { 1, 2, 3, 4, 5 });
            Assert.AreEqual(15f, vec.Sum(), Tolerance);
        }

        // ============================================================
        // Log — fallback
        // ============================================================

        [TestMethod]
        public void TestLog()
        {
            float[] data = { 1, 2, 3, 4, 5 };

            var gpuVec = new GpuColumnVector(data);
            var avxVec = new AvxColumnVector(data);

            var gpuResult = gpuVec.Log();
            var avxResult = avxVec.Log();

            for (int i = 0; i < data.Length; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance);
        }

        // ============================================================
        // OuterProduct
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(3, 5)]
        [DataRow(16, 8)]
        [DataRow(57, 41)]
        public void TestOuterProduct(int sizeA, int sizeB)
        {
            var dataA = MakeRandomArray(sizeA, 42);
            var dataB = MakeRandomArray(sizeB, 43);

            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.OuterProduct(gpuB);
            var avxResult = avxA.OuterProduct(avxB);

            Assert.AreEqual(avxResult.Rows, gpuResult.Rows);
            Assert.AreEqual(avxResult.Cols, gpuResult.Cols);

            for (int r = 0; r < sizeA; r++)
                for (int c = 0; c < sizeB; c++)
                    Assert.AreEqual(avxResult[r, c], gpuResult[r, c], Tolerance,
                        $"Mismatch at [{r},{c}]");
        }

        [TestMethod]
        public void TestOuterProduct_KnownValues()
        {
            var a = new GpuColumnVector(new float[] { 1, 2, 3 });
            var b = new GpuColumnVector(new float[] { 4, 5 });

            var result = a.OuterProduct(b);

            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(4f, result[0, 0], Tolerance);  // 1*4
            Assert.AreEqual(5f, result[0, 1], Tolerance);  // 1*5
            Assert.AreEqual(8f, result[1, 0], Tolerance);  // 2*4
            Assert.AreEqual(10f, result[1, 1], Tolerance); // 2*5
            Assert.AreEqual(12f, result[2, 0], Tolerance); // 3*4
            Assert.AreEqual(15f, result[2, 1], Tolerance); // 3*5
        }

        // ============================================================
        // Chained operations
        // ============================================================

        [TestMethod]
        public void TestChainedOperations()
        {
            var dataA = MakeRandomArray(32, 42);
            var dataB = MakeRandomArray(32, 43);

            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);
            var swA = new ColumnVector(dataA);
            var swB = new ColumnVector(dataB);

            // Chain: (A + B) * 2.0
            var gpuResult = gpuA.Add(gpuB).Multiply(2.0f);
            var swResult = swA.Add(swB).Multiply(2.0f);

            for (int i = 0; i < 32; i++)
                Assert.AreEqual(swResult[i], gpuResult[i], Tolerance);
        }

        // ============================================================
        // Dispose
        // ============================================================

        [TestMethod]
        public void TestDispose_NoThrow()
        {
            var gpu = new GpuColumnVector(new float[] { 1, 2, 3 });

            // Force device allocation
            gpu.Multiply(2.0f);

            gpu.Dispose();
            gpu.Dispose(); // double dispose should be safe
        }

        // ============================================================
        // Dimension mismatch
        // ============================================================

        [TestMethod]
        public void TestAdd_DimensionMismatch()
        {
            var a = new GpuColumnVector(new float[] { 1, 2, 3 });
            var b = new GpuColumnVector(new float[] { 1, 2, 3, 4 });
            
            Assert.ThrowsExactly<ArgumentException>(() => a.Add(b));
        }

        [TestMethod]
        public void TestSubtract_DimensionMismatch()
        {
            var a = new GpuColumnVector(new float[] { 1, 2, 3 });
            var b = new GpuColumnVector(new float[] { 1, 2 });
            //a.Subtract(b);

            Assert.ThrowsExactly<ArgumentException>(() => a.Subtract(b));
        }
    }
}
