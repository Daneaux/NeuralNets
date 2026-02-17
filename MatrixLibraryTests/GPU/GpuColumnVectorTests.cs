using MatrixLibrary;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using MatrixLibrary.Gpu;

namespace MatrixLibraryTests.Gpu
{
    /// <summary>
    /// Comprehensive tests for GpuColumnVector operations with DataRow parameterization
    /// Tests GPU vector implementations against AVX baseline
    /// </summary>
    [TestClass]
    public class GpuColumnVectorTests
    {
        private const float Tolerance = 1e-4f;
        private static readonly Random Random = new Random(42);

        #region Helper Methods

        private static float[] GenerateRandomVector(int size)
        {
            float[] data = new float[size];
            for (int i = 0; i < size; i++)
                data[i] = (float)(Random.NextDouble() * 10 - 5);
            return data;
        }

        private static void AssertVectorsEqual(ColumnVectorBase expected, ColumnVectorBase actual, string message = "")
        {
            Assert.AreEqual(expected.Size, actual.Size, $"{message} - Size mismatch");

            for (int i = 0; i < expected.Size; i++)
            {
                Assert.AreEqual(expected[i], actual[i], Tolerance,
                    $"{message} - Mismatch at index {i}: expected {expected[i]}, got {actual[i]}");
            }
        }

        private static bool IsGpuAvailable()
        {
            try
            {
                return BackendSelector.IsGPUAvailable();
            }
            catch
            {
                return false;
            }
        }

        #endregion

        #region Constructor and Property Tests

        [TestMethod]
        public void GpuColumnVector_Constructor_Empty_CreatesVector()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            using var vector = new GpuColumnVector(100);
            
            Assert.AreEqual(100, vector.Size);
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        [DataRow(1000)]
        public void GpuColumnVector_Constructor_WithData_CreatesVector(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(size);
            using var vector = new GpuColumnVector(data);
            
            Assert.AreEqual(size, vector.Size);
            
            // Verify data is accessible
            for (int i = 0; i < size; i++)
                Assert.AreEqual(data[i], vector[i], Tolerance);
        }

        [TestMethod]
        public void GpuColumnVector_Indexer_Get_ReturnsCorrectValue()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = { 1, 2, 3, 4, 5 };
            using var vector = new GpuColumnVector(data);
            
            Assert.AreEqual(1, vector[0], Tolerance);
            Assert.AreEqual(3, vector[2], Tolerance);
            Assert.AreEqual(5, vector[4], Tolerance);
        }

        [TestMethod]
        public void GpuColumnVector_Column_ReturnsArray()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = { 10, 20, 30 };
            using var vector = new GpuColumnVector(data);
            
            float[] column = vector.Column;
            
            Assert.AreEqual(3, column.Length);
            Assert.AreEqual(10, column[0], Tolerance);
            Assert.AreEqual(20, column[1], Tolerance);
            Assert.AreEqual(30, column[2], Tolerance);
        }

        #endregion

        #region Addition Tests

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        [DataRow(512)]
        public void GpuColumnVector_Add_Vector_MatchesAvx(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            using var gpuA = new GpuColumnVector(dataA);
            using var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.Add(gpuB);
            var avxResult = avxA.Add(avxB);

            AssertVectorsEqual(avxResult, gpuResult, $"Vector addition size {size}");
        }

        [DataTestMethod]
        [DataRow(1, 5.5f)]
        [DataRow(16, 10.0f)]
        [DataRow(32, -3.5f)]
        [DataRow(64, 0.0f)]
        [DataRow(128, 100.0f)]
        public void GpuColumnVector_Add_Scalar_MatchesAvx(int size, float scalar)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(size);
            
            using var gpu = new GpuColumnVector(data);
            var avx = new AvxColumnVector(data);

            var gpuResult = gpu.Add(scalar);
            var avxResult = avx.Add(scalar);

            AssertVectorsEqual(avxResult, gpuResult, $"Scalar addition size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        public void GpuColumnVector_Add_MixedTypes_MatchesAvx(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            using var gpuA = new GpuColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);

            var gpuResult = gpuA.Add(avxB);
            var avxResult = avxA.Add(avxB);

            AssertVectorsEqual(avxResult, gpuResult, $"Mixed type addition size {size}");
        }

        #endregion

        #region Subtraction Tests

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        [DataRow(512)]
        public void GpuColumnVector_Subtract_Vector_MatchesAvx(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            using var gpuA = new GpuColumnVector(dataA);
            using var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.Subtract(gpuB);
            var avxResult = avxA.Subtract(avxB);

            AssertVectorsEqual(avxResult, gpuResult, $"Vector subtraction size {size}");
        }

        [DataTestMethod]
        [DataRow(1, 5.0f)]
        [DataRow(16, 2.5f)]
        [DataRow(32, -1.0f)]
        public void GpuColumnVector_Subtract_Scalar_MatchesAvx(int size, float scalar)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(size);
            
            using var gpu = new GpuColumnVector(data);
            var avx = new AvxColumnVector(data);

            var gpuResult = gpu.Subtract(scalar);
            var avxResult = avx.Subtract(scalar);

            AssertVectorsEqual(avxResult, gpuResult, $"Scalar subtraction size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(16)]
        [DataRow(64)]
        [DataRow(256)]
        public void GpuColumnVector_Subtract_Self_ReturnsZero(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(size);
            
            using var gpu = new GpuColumnVector(data);
            var result = gpu.Subtract(gpu);

            for (int i = 0; i < size; i++)
                Assert.AreEqual(0, result[i], Tolerance);
        }

        #endregion

        #region Multiplication Tests

        [DataTestMethod]
        [DataRow(1, 2.5f)]
        [DataRow(16, 0.0f)]
        [DataRow(32, 1.0f)]
        [DataRow(64, -1.0f)]
        [DataRow(128, 3.5f)]
        [DataRow(256, -2.0f)]
        [DataRow(512, 0.5f)]
        public void GpuColumnVector_Multiply_Scalar_MatchesAvx(int size, float scalar)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(size);
            
            using var gpu = new GpuColumnVector(data);
            var avx = new AvxColumnVector(data);

            var gpuResult = gpu.Multiply(scalar);
            var avxResult = avx.Multiply(scalar);

            AssertVectorsEqual(avxResult, gpuResult, $"Scalar multiplication size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        public void GpuColumnVector_Multiply_Vector_MatchesAvx(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            using var gpuA = new GpuColumnVector(dataA);
            using var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.Multiply(gpuB);
            var avxResult = avxA.Multiply(avxB);

            AssertVectorsEqual(avxResult, gpuResult, $"Element-wise multiplication size {size}");
        }

        [TestMethod]
        public void GpuColumnVector_Multiply_ZeroScalar_ReturnsZero()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(100);
            
            using var gpu = new GpuColumnVector(data);
            var result = gpu.Multiply(0);

            for (int i = 0; i < 100; i++)
                Assert.AreEqual(0, result[i], Tolerance);
            (result as IDisposable)?.Dispose();
        }

        #endregion

        #region Operator Tests

        [DataTestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        public void GpuColumnVector_OperatorPlus_MatchesMethod(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);

            var opResult = gpuA + gpuB;
            using (var gpuA2 = new GpuColumnVector(dataA))
            using (var gpuB2 = new GpuColumnVector(dataB))
            {
                var methodResult = gpuA2.Add(gpuB2);
                AssertVectorsEqual(methodResult, opResult, "Operator + should match Add");
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        [DataTestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        public void GpuColumnVector_OperatorMinus_MatchesMethod(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);

            var opResult = gpuA - gpuB;
            using (var gpuA2 = new GpuColumnVector(dataA))
            using (var gpuB2 = new GpuColumnVector(dataB))
            {
                var methodResult = gpuA2.Subtract(gpuB2);
                AssertVectorsEqual(methodResult, opResult, "Operator - should match Subtract");
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        [DataTestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        public void GpuColumnVector_OperatorMultiply_MatchesMethod(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            var gpuA = new GpuColumnVector(dataA);
            var gpuB = new GpuColumnVector(dataB);

            var opResult = gpuA * gpuB;
            using (var gpuA2 = new GpuColumnVector(dataA))
            using (var gpuB2 = new GpuColumnVector(dataB))
            {
                var methodResult = gpuA2.Multiply(gpuB2);
                AssertVectorsEqual(methodResult, opResult, "Operator * should match Multiply");
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        #endregion

        #region Special Operations

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        [DataRow(512)]
        public void GpuColumnVector_Sum_MatchesAvx(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = GenerateRandomVector(size);
            
            using var gpu = new GpuColumnVector(data);
            var avx = new AvxColumnVector(data);

            float gpuSum = gpu.Sum();
            float avxSum = avx.Sum();

            Assert.AreEqual(avxSum, gpuSum, Tolerance, $"Sum mismatch for size {size}");
        }

        [TestMethod]
        public void GpuColumnVector_Sum_KnownValues()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] data = { 1, 2, 3, 4, 5 };
            
            using var gpu = new GpuColumnVector(data);
            float sum = gpu.Sum();

            Assert.AreEqual(15, sum, Tolerance); // 1+2+3+4+5 = 15
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void GpuColumnVector_OuterProduct_MatchesAvx(int sizeA, int sizeB)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(sizeA);
            float[] dataB = GenerateRandomVector(sizeB);
            
            using var gpuA = new GpuColumnVector(dataA);
            using var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.OuterProduct(gpuB);
            var avxResult = avxA.OuterProduct(avxB);

            Assert.AreEqual(avxResult.Rows, gpuResult.Rows);
            Assert.AreEqual(avxResult.Cols, gpuResult.Cols);
            for (int i = 0; i < avxResult.Rows; i++)
                for (int j = 0; j < avxResult.Cols; j++)
                    Assert.AreEqual(avxResult[i, j], gpuResult[i, j], Tolerance,
                        $"Mismatch at [{i},{j}]");
        }

        #endregion

        #region Disposal Tests

        [TestMethod]
        public void GpuColumnVector_Dispose_DoesNotThrow()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            var vector = new GpuColumnVector(100);
            vector.Dispose();
            
            Assert.IsTrue(true);
        }

        #endregion
    }
}
