using MatrixLibrary;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using MatrixLibrary.Gpu;

namespace MatrixLibraryTests.Gpu
{
    /// <summary>
    /// Comprehensive tests for GpuMatrix advanced operations (multiplication, transpose, etc.)
    /// Tests GPU implementations against AVX baseline with DataRow parameterization
    /// </summary>
    [TestClass]
    public class GpuMatrixAdvancedOperationsTests
    {
        private const float Tolerance = 1e-3f;
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

        #region Matrix Multiplication Tests

        [DataTestMethod]
        [DataRow(2, 2, 2)]
        [DataRow(4, 4, 4)]
        [DataRow(8, 8, 8)]
        [DataRow(16, 16, 16)]
        [DataRow(32, 32, 32)]
        [DataRow(64, 64, 64)]
        [DataRow(128, 128, 128)]
        public void GpuMatrix_Multiply_SquareMatrices_MatchesAvx(int rows, int aCols, int bCols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, aCols);
            float[,] dataB = GenerateRandomMatrix(aCols, bCols);
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuB = new GpuMatrix(dataB);
            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);

            var gpuResult = gpuA.Multiply(gpuB);
            var avxResult = avxA.Multiply(avxB);

            AssertMatricesEqual(avxResult, gpuResult, $"Matrix multiply {rows}x{aCols} * {aCols}x{bCols}");
        }

        [DataTestMethod]
        [DataRow(2, 3, 4)]
        [DataRow(4, 8, 2)]
        [DataRow(8, 16, 8)]
        [DataRow(16, 32, 16)]
        [DataRow(32, 64, 32)]
        [DataRow(100, 200, 50)]
        public void GpuMatrix_Multiply_NonSquare_MatchesAvx(int aRows, int aCols, int bCols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(aRows, aCols);
            float[,] dataB = GenerateRandomMatrix(aCols, bCols);
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuB = new GpuMatrix(dataB);
            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);

            var gpuResult = gpuA.Multiply(gpuB);
            var avxResult = avxA.Multiply(avxB);

            AssertMatricesEqual(avxResult, gpuResult, $"Non-square multiply {aRows}x{aCols} * {aCols}x{bCols}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        public void GpuMatrix_Multiply_IdentityMatrix_ReturnsOriginal(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            float[,] identity = new float[cols, cols];
            for (int i = 0; i < cols; i++)
                identity[i, i] = 1.0f;
            
            using var gpu = new GpuMatrix(data);
            using var gpuIdentity = new GpuMatrix(identity);
            
            var result = gpu.Multiply(gpuIdentity);

            AssertMatricesEqual(gpu, result, "Multiply by identity should return original");
        }

        [TestMethod]
        public void GpuMatrix_Multiply_ZeroMatrix_ReturnsZero()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(32, 32);
            float[,] zero = new float[32, 32];
            
            using var gpu = new GpuMatrix(data);
            using var gpuZero = new GpuMatrix(zero);
            
            var result = gpu.Multiply(gpuZero);

            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                    Assert.AreEqual(0, result[i, j], Tolerance);
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        public void GpuMatrix_OperatorMultiply_MatchesMethod(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(cols, cols);
            
            var gpuA = new GpuMatrix(dataA);
            var gpuB = new GpuMatrix(dataB);

            var opResult = gpuA * gpuB;
            using (var gpuA2 = new GpuMatrix(dataA))
            using (var gpuB2 = new GpuMatrix(dataB))
            {
                var methodResult = gpuA2.Multiply(gpuB2);
                AssertMatricesEqual(methodResult, opResult, "Operator * should match Multiply method");
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        #endregion

        #region Matrix-Vector Multiplication Tests

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        [DataRow(128, 128)]
        public void GpuMatrix_MatrixTimesColumn_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] matrixData = GenerateRandomMatrix(rows, cols);
            float[] vectorData = new float[cols];
            for (int i = 0; i < cols; i++)
                vectorData[i] = (float)(Random.NextDouble() * 10 - 5);
            
            using var gpuMatrix = new GpuMatrix(matrixData);
            using var gpuVector = new GpuColumnVector(vectorData);
            var avxMatrix = new AvxMatrix(matrixData);
            var avxVector = new AvxColumnVector(vectorData);

            var gpuResult = gpuMatrix.MatrixTimesColumn(gpuVector);
            var avxResult = avxMatrix.MatrixTimesColumn(avxVector);

            Assert.AreEqual(avxResult.Size, gpuResult.Size);
            for (int i = 0; i < avxResult.Size; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance, $"Mismatch at index {i}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        public void GpuMatrix_OperatorMultiply_Vector_MatchesMethod(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] matrixData = GenerateRandomMatrix(rows, cols);
            float[] vectorData = new float[cols];
            for (int i = 0; i < cols; i++)
                vectorData[i] = (float)(Random.NextDouble() * 10);
            
            var gpuMatrix = new GpuMatrix(matrixData);
            var gpuVector = new GpuColumnVector(vectorData);

            var opResult = gpuMatrix * gpuVector;
            using (var gpuMatrix2 = new GpuMatrix(matrixData))
            using (var gpuVector2 = new GpuColumnVector(vectorData))
            {
                var methodResult = gpuMatrix2.MatrixTimesColumn(gpuVector2);
                Assert.AreEqual(methodResult.Size, opResult.Size);
                for (int i = 0; i < methodResult.Size; i++)
                    Assert.AreEqual(methodResult[i], opResult[i], Tolerance);
                (methodResult as IDisposable)?.Dispose();
            }
            (opResult as IDisposable)?.Dispose();
        }

        #endregion

        #region Transpose Tests

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        [DataRow(128, 128)]
        public void GpuMatrix_Transpose_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var avx = new AvxMatrix(data);

            var gpuResult = gpu.GetTransposedMatrix();
            var avxResult = avx.GetTransposedMatrix();

            AssertMatricesEqual(avxResult, gpuResult, $"Transpose {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(2, 3)]
        [DataRow(4, 8)]
        [DataRow(8, 16)]
        [DataRow(16, 32)]
        [DataRow(32, 16)]
        [DataRow(100, 200)]
        public void GpuMatrix_Transpose_NonSquare_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var avx = new AvxMatrix(data);

            var gpuResult = gpu.GetTransposedMatrix();
            var avxResult = avx.GetTransposedMatrix();

            AssertMatricesEqual(avxResult, gpuResult, $"Non-square transpose {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        public void GpuMatrix_Transpose_TwiceReturnsOriginal(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var transposed = gpu.GetTransposedMatrix();
            var doubleTransposed = transposed.GetTransposedMatrix();

            AssertMatricesEqual(gpu, doubleTransposed, "Double transpose should return original");
        }

        #endregion

        #region Hadamard Product Tests

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void GpuMatrix_HadamardProduct_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuB = new GpuMatrix(dataB);
            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);

            var gpuResult = gpuA.HadamardProduct(gpuB);
            var avxResult = avxA.HadamardProduct(avxB);

            AssertMatricesEqual(avxResult, gpuResult, $"Hadamard product {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        public void GpuMatrix_HadamardProduct_ZeroMatrix_ReturnsZero(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] zero = new float[rows, cols];
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuZero = new GpuMatrix(zero);
            
            var result = gpuA.HadamardProduct(gpuZero);

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    Assert.AreEqual(0, result[i, j], Tolerance);
        }

        #endregion

        #region Sum Tests

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void GpuMatrix_Sum_MatchesAvx(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var avx = new AvxMatrix(data);

            float gpuSum = gpu.Sum();
            float avxSum = avx.Sum();

            Assert.AreEqual(avxSum, gpuSum, Tolerance, $"Sum mismatch for {rows}x{cols}");
        }

        [TestMethod]
        public void GpuMatrix_Sum_KnownValues()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = { { 1, 2 }, { 3, 4 } };
            
            using var gpu = new GpuMatrix(data);
            float sum = gpu.Sum();

            Assert.AreEqual(10, sum, Tolerance); // 1+2+3+4 = 10
        }

        #endregion

        #region Convolution Tests

        [DataTestMethod]
        [DataRow(5, 3)]
        [DataRow(8, 3)]
        [DataRow(16, 3)]
        [DataRow(32, 3)]
        public void GpuMatrix_Convolution_Valid_MatchesAvx(int matrixSize, int kernelSize)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(matrixSize, matrixSize);
            float[,] kernel = GenerateRandomMatrix(kernelSize, kernelSize);
            
            using var gpu = new GpuMatrix(data);
            using var gpuKernel = new GpuMatrix(kernel);
            var avx = new AvxMatrix(data);
            var avxKernel = new AvxMatrix(kernel);

            var gpuResult = gpu.Convolution(gpuKernel);
            var avxResult = avx.Convolution(avxKernel);

            AssertMatricesEqual(avxResult, gpuResult, $"Convolution {matrixSize}x{matrixSize} with {kernelSize}x{kernelSize}");
        }

        [DataTestMethod]
        [DataRow(5, 3)]
        [DataRow(8, 4)]
        [DataRow(16, 5)]
        public void GpuMatrix_ConvolutionFull_MatchesAvx(int matrixSize, int kernelSize)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(matrixSize, matrixSize);
            float[,] kernel = GenerateRandomMatrix(kernelSize, kernelSize);
            
            using var gpu = new GpuMatrix(data);
            using var gpuKernel = new GpuMatrix(kernel);
            var avx = new AvxMatrix(data);
            var avxKernel = new AvxMatrix(kernel);

            var gpuResult = gpu.ConvolutionFull(gpuKernel);
            var avxResult = avx.ConvolutionFull(avxKernel);

            AssertMatricesEqual(avxResult, gpuResult, $"Full convolution {matrixSize}x{matrixSize} with {kernelSize}x{kernelSize}");
        }

        #endregion

        #region Chained Operations Tests

        [TestMethod]
        public void GpuMatrix_ChainedOperations_DeviceStaysResident()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] dataA = GenerateRandomMatrix(32, 32);
            float[,] dataB = GenerateRandomMatrix(32, 32);
            float[,] dataC = GenerateRandomMatrix(32, 32);
            
            using var gpuA = new GpuMatrix(dataA);
            using var gpuB = new GpuMatrix(dataB);
            using var gpuC = new GpuMatrix(dataC);

            // Chain: A + B * scalar - C
            var multiplied = gpuB.Multiply(2.0f);
            var added = gpuA.Add(multiplied);
            var result = added.Subtract(gpuC);

            // Calculate expected on CPU
            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);
            var avxC = new AvxMatrix(dataC);
            var avxMultiplied = avxB.Multiply(2.0f);
            var avxAdded = avxA.Add(avxMultiplied);
            var avxResult = avxAdded.Subtract(avxC);

            AssertMatricesEqual(avxResult, result, "Chained operations");
        }

        #endregion
    }
}
