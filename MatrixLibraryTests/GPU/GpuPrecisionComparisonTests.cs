using MatrixLibrary;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using MatrixLibrary.Gpu;

namespace MatrixLibraryTests.Gpu
{
    /// <summary>
    /// Precision comparison tests between GPU and AVX implementations
    /// Verifies that GPU produces identical results to CPU implementations
    /// </summary>
    [TestClass]
    public class GpuPrecisionComparisonTests
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

        private static float[] GenerateRandomVector(int size)
        {
            float[] data = new float[size];
            for (int i = 0; i < size; i++)
                data[i] = (float)(Random.NextDouble() * 10 - 5);
            return data;
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

        #region Matrix Precision Tests

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        [DataRow(128, 128)]
        public void GpuVsAvx_MatrixAddition_Precision(int rows, int cols)
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

            float maxError = 0;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    maxError = Math.Max(maxError, Math.Abs(gpuResult[i, j] - avxResult[i, j]));
            
            Assert.IsTrue(maxError < Tolerance, $"Max error {maxError} exceeds tolerance for {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(2, 2, 2)]
        [DataRow(4, 4, 4)]
        [DataRow(8, 8, 8)]
        [DataRow(16, 16, 16)]
        [DataRow(32, 32, 32)]
        [DataRow(64, 64, 64)]
        public void GpuVsAvx_MatrixMultiplication_Precision(int aRows, int aCols, int bCols)
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

            float maxError = 0;
            for (int i = 0; i < aRows; i++)
                for (int j = 0; j < bCols; j++)
                    maxError = Math.Max(maxError, Math.Abs(gpuResult[i, j] - avxResult[i, j]));
            
            Assert.IsTrue(maxError < Tolerance, $"Max error {maxError} exceeds tolerance for {aRows}x{aCols}*{aCols}x{bCols}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void GpuVsAvx_MatrixTranspose_Precision(int rows, int cols)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] data = GenerateRandomMatrix(rows, cols);
            
            using var gpu = new GpuMatrix(data);
            var avx = new AvxMatrix(data);

            var gpuResult = gpu.GetTransposedMatrix();
            var avxResult = avx.GetTransposedMatrix();

            float maxError = 0;
            for (int i = 0; i < cols; i++)
                for (int j = 0; j < rows; j++)
                    maxError = Math.Max(maxError, Math.Abs(gpuResult[i, j] - avxResult[i, j]));
            
            Assert.IsTrue(maxError < Tolerance, $"Max error {maxError} exceeds tolerance for {rows}x{cols}");
        }

        #endregion

        #region Vector Precision Tests

        [DataTestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        [DataRow(512)]
        public void GpuVsAvx_VectorAddition_Precision(int size)
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

            float maxError = 0;
            for (int i = 0; i < size; i++)
                maxError = Math.Max(maxError, Math.Abs(gpuResult[i] - avxResult[i]));
            
            Assert.IsTrue(maxError < Tolerance, $"Max error {maxError} exceeds tolerance for size {size}");
        }

        [DataTestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(256)]
        public void GpuVsAvx_VectorDotProduct_Precision(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            using var gpuA = new GpuColumnVector(dataA);
            using var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            // Element-wise multiply then sum
            var gpuMul = gpuA.Multiply(gpuB);
            var avxMul = avxA.Multiply(avxB);
            
            float gpuSum = gpuMul.Sum();
            float avxSum = avxMul.Sum();

            Assert.AreEqual(avxSum, gpuSum, Tolerance, $"Dot product mismatch for size {size}");
        }

        [DataTestMethod]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(32)]
        [DataRow(64)]
        public void GpuVsAvx_VectorOuterProduct_Precision(int size)
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);
            
            using var gpuA = new GpuColumnVector(dataA);
            using var gpuB = new GpuColumnVector(dataB);
            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);

            var gpuResult = gpuA.OuterProduct(gpuB);
            var avxResult = avxA.OuterProduct(avxB);

            float maxError = 0;
            for (int i = 0; i < size; i++)
                for (int j = 0; j < size; j++)
                    maxError = Math.Max(maxError, Math.Abs(gpuResult[i, j] - avxResult[i, j]));
            
            Assert.IsTrue(maxError < Tolerance, $"Max error {maxError} exceeds tolerance for size {size}");
        }

        #endregion

        #region Special Value Tests

        [TestMethod]
        public void GpuVsAvx_ZeroValues_Match()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] matrixData = new float[32, 32];
            float[] vectorData = new float[64];

            using var gpuMatrix = new GpuMatrix(matrixData);
            using var gpuVector = new GpuColumnVector(vectorData);
            var avxMatrix = new AvxMatrix(matrixData);
            var avxVector = new AvxColumnVector(vectorData);

            var gpuMatrixAdd = gpuMatrix.Add(gpuMatrix);
            var avxMatrixAdd = avxMatrix.Add(avxMatrix);
            
            var gpuVectorAdd = gpuVector.Add(gpuVector);
            var avxVectorAdd = avxVector.Add(avxVector);

            Assert.AreEqual(0, gpuMatrixAdd[0, 0], Tolerance);
            Assert.AreEqual(0, gpuVectorAdd[0], Tolerance);
        }

        [TestMethod]
        public void GpuVsAvx_NegativeValues_Match()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];
            
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    matrixData[i, j] = -((i * 16 + j + 1) * 0.5f);
            
            for (int i = 0; i < 64; i++)
                vectorData[i] = -(i + 1) * 0.25f;

            using var gpuMatrix = new GpuMatrix(matrixData);
            using var gpuVector = new GpuColumnVector(vectorData);
            var avxMatrix = new AvxMatrix(matrixData);
            var avxVector = new AvxColumnVector(vectorData);

            var gpuMatrixMul = gpuMatrix.Multiply(2.0f);
            var avxMatrixMul = avxMatrix.Multiply(2.0f);
            
            var gpuVectorMul = gpuVector.Multiply(2.0f);
            var avxVectorMul = avxVector.Multiply(2.0f);

            float maxMatrixError = 0;
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    maxMatrixError = Math.Max(maxMatrixError, Math.Abs(gpuMatrixMul[i, j] - avxMatrixMul[i, j]));
            
            float maxVectorError = 0;
            for (int i = 0; i < 64; i++)
                maxVectorError = Math.Max(maxVectorError, Math.Abs(gpuVectorMul[i] - avxVectorMul[i]));

            Assert.IsTrue(maxMatrixError < Tolerance, $"Matrix error {maxMatrixError} exceeds tolerance");
            Assert.IsTrue(maxVectorError < Tolerance, $"Vector error {maxVectorError} exceeds tolerance");
        }

        [TestMethod]
        public void GpuVsAvx_LargeValues_Match()
        {
            if (!IsGpuAvailable()) Assert.Inconclusive("GPU not available");

            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];
            
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    matrixData[i, j] = 1000.0f * (i + j);
            
            for (int i = 0; i < 64; i++)
                vectorData[i] = 1000.0f * i;

            using var gpuMatrix = new GpuMatrix(matrixData);
            using var gpuVector = new GpuColumnVector(vectorData);
            var avxMatrix = new AvxMatrix(matrixData);
            var avxVector = new AvxColumnVector(vectorData);

            var gpuMatrixAdd = gpuMatrix.Add(gpuMatrix);
            var avxMatrixAdd = avxMatrix.Add(avxMatrix);
            
            var gpuVectorAdd = gpuVector.Add(gpuVector);
            var avxVectorAdd = avxVector.Add(avxVector);

            float maxMatrixError = 0;
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    maxMatrixError = Math.Max(maxMatrixError, Math.Abs(gpuMatrixAdd[i, j] - avxMatrixAdd[i, j]));
            
            float maxVectorError = 0;
            for (int i = 0; i < 64; i++)
                maxVectorError = Math.Max(maxVectorError, Math.Abs(gpuVectorAdd[i] - avxVectorAdd[i]));

            Assert.IsTrue(maxMatrixError < Tolerance, $"Matrix error {maxMatrixError} exceeds tolerance");
            Assert.IsTrue(maxVectorError < Tolerance, $"Vector error {maxVectorError} exceeds tolerance");
        }

        #endregion
    }
}
