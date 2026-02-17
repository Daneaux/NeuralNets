using MatrixLibrary;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using MatrixLibrary.Gpu;

namespace MatrixLibraryTests.Gpu
{
    /// <summary>
    /// Comprehensive tests for GpuMatrix basic arithmetic operations using DataRow parameterization
    /// Tests GPU implementations against AVX baseline with various matrix sizes
    /// </summary>
    [TestClass]
    public class GpuMatrixBasicOperationsTests
    {
        private const float Tolerance = 1e-4f;
        private static readonly Random Random = new Random(42);

        #region Helper Methods

        private static float[,] GenerateRandomMatrix(int rows, int cols)
        {
            float[,] data = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i, j] = (float)(Random.NextDouble() * 10 - 5);
            return data;
        }

        private static void AssertMatricesEqual(MatrixBase expected, MatrixBase actual, string message = "")
        {
            Assert.AreEqual(expected.Rows, actual.Rows, $"{message} - Row count mismatch");
            Assert.AreEqual(expected.Cols, actual.Cols, $"{message} - Column count mismatch");

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], actual[i, j], Tolerance,
                        $"{message} - Mismatch at [{i},{j}]: expected {expected[i, j]}, got {actual[i, j]}");
                }
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
        public void GpuMatrix_Constructor_Empty_CreatesMatrix()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            using var matrix = new GpuMatrix(10, 10);
            
            Assert.AreEqual(10, matrix.Rows);
            Assert.AreEqual(10, matrix.Cols);
        }

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        [DataRow(100, 100)]
        [DataRow(256, 256)]
        public void GpuMatrix_Constructor_WithData_CreatesMatrix(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            using var matrix = new GpuMatrix(data);
            
            Assert.AreEqual(rows, matrix.Rows);
            Assert.AreEqual(cols, matrix.Cols);
            
            // Verify data is accessible
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.AreEqual(data[i, j], matrix[i, j], Tolerance);
        }

        [TestMethod]
        public void GpuMatrix_Indexer_GetAndSet_Works()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = { { 1, 2 }, { 3, 4 } };
            using var matrix = new GpuMatrix(data);
            
            // Get
            Assert.AreEqual(1, matrix[0, 0], Tolerance);
            Assert.AreEqual(4, matrix[1, 1], Tolerance);
            
            // Set
            matrix[0, 0] = 100;
            Assert.AreEqual(100, matrix[0, 0], Tolerance);
        }

        [TestMethod]
        public void GpuMatrix_IsSquare_DetectsCorrectly()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            using var square = new GpuMatrix(5, 5);
            using var nonSquare = new GpuMatrix(3, 4);
            
            Assert.IsTrue(square.IsSquare());
            Assert.IsFalse(nonSquare.IsSquare());
        }

        [TestMethod]
        public void GpuMatrix_TotalSize_ReturnsCorrectValue()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            using var matrix = new GpuMatrix(10, 20);
            Assert.AreEqual(200, matrix.TotalSize);
        }

        #endregion

        #region Addition Tests

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        [DataRow(100, 100)]
        [DataRow(128, 256)]
        [DataRow(256, 128)]
        public void GpuMatrix_Add_Matrix_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuB = new GpuMatrix(dataB);
            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);

            var gpuResult = gpuA.Add(gpuB);
            var avxResult = avxA.Add(avxB);

            AssertMatricesEqual(avxResult, gpuResult, $"Matrix addition {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void GpuMatrix_Add_MixedTypes_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);
            
            using var gpuA = new GpuMatrix(dataA);
            var avxB = new AvxMatrix(dataB);  // Different type
            var avxA = new AvxMatrix(dataA);

            var gpuResult = gpuA.Add(avxB);
            var avxResult = avxA.Add(avxB);

            AssertMatricesEqual(avxResult, gpuResult, $"Mixed type addition {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(1, 1, 5.5f)]
        [DataRow(4, 4, 10.0f)]
        [DataRow(16, 16, -3.5f)]
        [DataRow(32, 32, 0.0f)]
        [DataRow(64, 64, 100.0f)]
        public void GpuMatrix_Add_Scalar_MatchesAvx(int rows, int cols, float scalar)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var avx = new AvxMatrix(data);

            var gpuResult = gpu.Add(scalar);
            var avxResult = avx.Add(scalar);

            AssertMatricesEqual(avxResult, gpuResult, $"Scalar addition {rows}x{cols} scalar={scalar}");
        }

        #endregion

        #region Subtraction Tests

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        [DataRow(100, 100)]
        [DataRow(128, 256)]
        [DataRow(256, 128)]
        public void GpuMatrix_Subtract_Matrix_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuB = new GpuMatrix(dataB);
            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);

            var gpuResult = gpuA.Subtract(gpuB);
            var avxResult = avxA.Subtract(avxB);

            AssertMatricesEqual(avxResult, gpuResult, $"Matrix subtraction {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(4, 4)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void GpuMatrix_Subtract_Self_ReturnsZero(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var result = gpu.Subtract(gpu);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.AreEqual(0, result[i, j], Tolerance);
        }

        #endregion

        #region Scalar Multiplication Tests

        [DataTestMethod]
        [DataRow(1, 1, 2.5f)]
        [DataRow(4, 4, 0.0f)]
        [DataRow(8, 8, 1.0f)]
        [DataRow(16, 16, -1.0f)]
        [DataRow(32, 32, 3.5f)]
        [DataRow(64, 64, -2.0f)]
        [DataRow(128, 128, 0.5f)]
        public void GpuMatrix_Multiply_Scalar_MatchesAvx(int rows, int cols, float scalar)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var avx = new AvxMatrix(data);

            var gpuResult = gpu.Multiply(scalar);
            var avxResult = avx.Multiply(scalar);

            AssertMatricesEqual(avxResult, gpuResult, $"Scalar multiplication {rows}x{cols} scalar={scalar}");
        }

        [TestMethod]
        public void GpuMatrix_Multiply_ZeroScalar_ReturnsZeroMatrix()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(32, 32);
            
            using var gpu = new GpuMatrix(data);
            var result = gpu.Multiply(0);

            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                    Assert.AreEqual(0, result[i, j], Tolerance);
        }

        #endregion

        #region Operator Tests

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        public void GpuMatrix_OperatorPlus_MatchesMethod(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);
            
            var gpuA = new GpuMatrix(dataA);
            var gpuB = new GpuMatrix(dataB);

            var opResult = gpuA + gpuB;
            using (var gpuA2 = new GpuMatrix(dataA))
            using (var gpuB2 = new GpuMatrix(dataB))
            {
                var methodResult = gpuA2.Add(gpuB2);
                AssertMatricesEqual(methodResult, opResult, "Operator + should match Add method");
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        public void GpuMatrix_OperatorMinus_MatchesMethod(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);
            
            var gpuA = new GpuMatrix(dataA);
            var gpuB = new GpuMatrix(dataB);

            var opResult = gpuA - gpuB;
            using (var gpuA2 = new GpuMatrix(dataA))
            using (var gpuB2 = new GpuMatrix(dataB))
            {
                var methodResult = gpuA2.Subtract(gpuB2);
                AssertMatricesEqual(methodResult, opResult, "Operator - should match Subtract method");
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        #endregion

        #region Immutability Tests

        [TestMethod]
        public void GpuMatrix_Operations_DoNotModifyOriginal()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = { { 1, 2 }, { 3, 4 } };
            
            using var original = new GpuMatrix(data);
            float originalValue = original[0, 0];

            var added = original.Add(100);
            Assert.AreEqual(originalValue, original[0, 0], Tolerance, "Original should not be modified by Add");
            (added as IDisposable)?.Dispose();

            var multiplied = original.Multiply(10);
            Assert.AreEqual(originalValue, original[0, 0], Tolerance, "Original should not be modified by Multiply");
            (multiplied as IDisposable)?.Dispose();
        }

        #endregion

        #region Disposal Tests

        [TestMethod]
        public void GpuMatrix_Dispose_DoesNotThrow()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            var matrix = new GpuMatrix(10, 10);
            matrix.Dispose();
            
            // Test passes if no exception thrown
            Assert.IsTrue(true);
        }

        [TestMethod]
        public void GpuMatrix_MultipleDisposal_DoesNotThrow()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            var matrix = new GpuMatrix(10, 10);
            matrix.Dispose();
            matrix.Dispose();  // Should not throw
            
            Assert.IsTrue(true);
        }

        #endregion
    }
}
