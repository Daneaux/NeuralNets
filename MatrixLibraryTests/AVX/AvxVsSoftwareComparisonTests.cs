using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests.Avx
{
    /// <summary>
    /// Systematic comparison tests between AVX and Software implementations
    /// Verifies that AvxMatrix and AvxColumnVector produce identical results to software versions
    /// </summary>
    [TestClass]
    public class AvxVsSoftwareComparisonTests
    {
        private const float Tolerance = 1e-4f;
        private Random _random = new Random(42);

        #region Matrix Comparison Tests

        [DataTestMethod]
        [DataRow(1, 1)]      // Single element
        [DataRow(2, 2)]      // Small square
        [DataRow(3, 4)]      // Non-square
        [DataRow(4, 4)]      // Common CNN size
        [DataRow(8, 8)]      // AVX2-friendly
        [DataRow(16, 16)]    // One full AVX-512 vector
        [DataRow(17, 17)]    // Remainder handling
        [DataRow(32, 32)]    // Two AVX-512 vectors
        [DataRow(64, 64)]    // Large
        [DataRow(100, 100)]  // Stress test
        public void AvxMatrix_Addition_MatchesSoftware_AllSizes(int rows, int cols)
        {
            // Arrange
            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);

            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);
            var softwareA = new Matrix2D(dataA);
            var softwareB = new Matrix2D(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, $"Addition {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(16, 16)]
        [DataRow(17, 17)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void AvxMatrix_Subtraction_MatchesSoftware_AllSizes(int rows, int cols)
        {
            // Arrange
            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);

            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);
            var softwareA = new Matrix2D(dataA);
            var softwareB = new Matrix2D(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, $"Subtraction {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(1, 1)]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(16, 16)]
        [DataRow(17, 17)]
        [DataRow(32, 32)]
        [DataRow(64, 64)]
        public void AvxMatrix_ScalarMultiply_MatchesSoftware_AllSizes(int rows, int cols)
        {
            // Arrange
            float[,] data = GenerateRandomMatrix(rows, cols);
            float scalar = (float)(_random.NextDouble() * 10 - 5);

            var avx = new AvxMatrix(data);
            var software = new Matrix2D(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, $"Scalar multiply {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(2, 2, 2)]      // Small square
        [DataRow(3, 4, 4)]      // Non-square
        [DataRow(4, 4, 4)]      // Common size
        [DataRow(8, 8, 8)]      // AVX-friendly
        [DataRow(16, 16, 16)]   // Full vector
        [DataRow(32, 32, 32)]   // Large
        public void AvxMatrix_MatrixMultiply_MatchesSoftware_AllSizes(int aRows, int aCols, int bCols)
        {
            // Arrange
            float[,] dataA = GenerateRandomMatrix(aRows, aCols);
            float[,] dataB = GenerateRandomMatrix(aCols, bCols);

            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);
            var softwareA = new Matrix2D(dataA);
            var softwareB = new Matrix2D(dataB);

            // Act
            var avxResult = avxA.Multiply(avxB);
            var softwareResult = softwareA.Multiply(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, $"Matrix multiply {aRows}x{aCols} * {aCols}x{bCols}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        public void AvxMatrix_Transpose_MatchesSoftware_AllSizes(int rows, int cols)
        {
            // Arrange
            float[,] data = GenerateRandomMatrix(rows, cols);

            var avx = new AvxMatrix(data);
            var software = new Matrix2D(data);

            // Act
            var avxResult = avx.GetTransposedMatrix();
            var softwareResult = software.GetTransposedMatrix();

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, $"Transpose {rows}x{cols}");
        }

        [DataTestMethod]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        public void AvxMatrix_HadamardProduct_MatchesSoftware_AllSizes(int rows, int cols)
        {
            // Arrange
            float[,] dataA = GenerateRandomMatrix(rows, cols);
            float[,] dataB = GenerateRandomMatrix(rows, cols);

            var avxA = new AvxMatrix(dataA);
            var avxB = new AvxMatrix(dataB);
            var softwareA = new Matrix2D(dataA);
            var softwareB = new Matrix2D(dataB);

            // Act
            var avxResult = avxA.HadamardProduct(avxB);
            var softwareResult = softwareA.HadamardProduct(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, $"Hadamard product {rows}x{cols}");
        }

        #endregion

        #region Vector Comparison Tests

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(2)]
        [DataRow(4)]
        [DataRow(8)]
        [DataRow(15)]
        [DataRow(16)]    // One AVX-512 vector
        [DataRow(17)]    // Remainder
        [DataRow(31)]
        [DataRow(32)]    // Two AVX-512 vectors
        [DataRow(33)]
        [DataRow(64)]    // Four AVX-512 vectors
        [DataRow(128)]   // Eight AVX-512 vectors
        [DataRow(1000)]  // Large
        public void AvxColumnVector_Addition_MatchesSoftware_AllSizes(int size)
        {
            // Arrange
            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, $"Vector addition size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(17)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        public void AvxColumnVector_Subtraction_MatchesSoftware_AllSizes(int size)
        {
            // Arrange
            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, $"Vector subtraction size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(17)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        public void AvxColumnVector_ElementWiseMultiply_MatchesSoftware_AllSizes(int size)
        {
            // Arrange
            float[] dataA = GenerateRandomVector(size);
            float[] dataB = GenerateRandomVector(size);

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Multiply(avxB);
            var softwareResult = softwareA.Multiply(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, $"Element-wise multiply size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(17)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        public void AvxColumnVector_ScalarMultiply_MatchesSoftware_AllSizes(int size)
        {
            // Arrange
            float[] data = GenerateRandomVector(size);
            float scalar = (float)(_random.NextDouble() * 10 - 5);

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, $"Scalar multiply size {size}");
        }

        [DataTestMethod]
        [DataRow(1)]
        [DataRow(4)]
        [DataRow(16)]
        [DataRow(17)]
        [DataRow(32)]
        [DataRow(64)]
        [DataRow(128)]
        [DataRow(1000)]
        public void AvxColumnVector_Sum_MatchesSoftware_AllSizes(int size)
        {
            // Arrange
            float[] data = GenerateRandomVector(size);

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            float avxResult = avx.Sum();
            float softwareResult = software.Sum();

            // Assert
            Assert.AreEqual(softwareResult, avxResult, Tolerance,
                $"Sum operation size {size}");
        }

        [DataTestMethod]
        [DataRow(2, 2)]
        [DataRow(4, 4)]
        [DataRow(8, 8)]
        [DataRow(16, 16)]
        [DataRow(32, 32)]
        public void AvxColumnVector_OuterProduct_MatchesSoftware_AllSizes(int sizeA, int sizeB)
        {
            // Arrange
            float[] dataA = GenerateRandomVector(sizeA);
            float[] dataB = GenerateRandomVector(sizeB);

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.OuterProduct(avxB);
            var softwareResult = softwareA.OuterProduct(softwareB);

            // Assert
            Assert.AreEqual(softwareResult.Rows, avxResult.Rows);
            Assert.AreEqual(softwareResult.Cols, avxResult.Cols);
            for (int i = 0; i < softwareResult.Rows; i++)
                for (int j = 0; j < softwareResult.Cols; j++)
                    Assert.AreEqual(softwareResult[i, j], avxResult[i, j], Tolerance,
                        $"Mismatch at [{i},{j}]");
        }

        #endregion

        #region Special Value Tests

        [TestMethod]
        public void AvxVsSoftware_ZeroValues_Match()
        {
            // Test with all zeros
            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];

            TestMatrixOperations(matrixData, "Zero matrix");
            TestVectorOperations(vectorData, "Zero vector");
        }

        [TestMethod]
        public void AvxVsSoftware_NegativeValues_Match()
        {
            // Test with negative values
            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];
            
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    matrixData[i, j] = -((i * 16 + j + 1) * 0.5f);
            
            for (int i = 0; i < 64; i++)
                vectorData[i] = -(i + 1) * 0.25f;

            TestMatrixOperations(matrixData, "Negative matrix");
            TestVectorOperations(vectorData, "Negative vector");
        }

        [TestMethod]
        public void AvxVsSoftware_SmallValues_Match()
        {
            // Test with very small values (underflow check)
            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];
            
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    matrixData[i, j] = 1e-6f * (i + j);
            
            for (int i = 0; i < 64; i++)
                vectorData[i] = 1e-6f * i;

            TestMatrixOperations(matrixData, "Small value matrix");
            TestVectorOperations(vectorData, "Small value vector");
        }

        [TestMethod]
        public void AvxVsSoftware_LargeValues_Match()
        {
            // Test with large values (overflow check)
            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];
            
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    matrixData[i, j] = 100.0f * (i + j);
            
            for (int i = 0; i < 64; i++)
                vectorData[i] = 100.0f * i;

            TestMatrixOperations(matrixData, "Large value matrix");
            TestVectorOperations(vectorData, "Large value vector");
        }

        [TestMethod]
        public void AvxVsSoftware_MixedSignValues_Match()
        {
            // Test with alternating signs
            float[,] matrixData = new float[16, 16];
            float[] vectorData = new float[64];
            
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 16; j++)
                    matrixData[i, j] = ((i + j) % 2 == 0 ? 1 : -1) * (i + j + 1);
            
            for (int i = 0; i < 64; i++)
                vectorData[i] = (i % 2 == 0 ? 1 : -1) * (i + 1);

            TestMatrixOperations(matrixData, "Mixed sign matrix");
            TestVectorOperations(vectorData, "Mixed sign vector");
        }

        #endregion

        #region Helper Methods

        private float[,] GenerateRandomMatrix(int rows, int cols)
        {
            float[,] data = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[i, j] = (float)(_random.NextDouble() * 10 - 5);
            return data;
        }

        private float[] GenerateRandomVector(int size)
        {
            float[] data = new float[size];
            for (int i = 0; i < size; i++)
                data[i] = (float)(_random.NextDouble() * 10 - 5);
            return data;
        }

        private void AssertMatricesEqual(MatrixBase expected, MatrixBase actual, string message)
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

        private void AssertVectorsEqual(ColumnVectorBase expected, ColumnVectorBase actual, string message)
        {
            Assert.AreEqual(expected.Size, actual.Size, $"{message} - Size mismatch");

            for (int i = 0; i < expected.Size; i++)
            {
                Assert.AreEqual(expected[i], actual[i], Tolerance,
                    $"{message} - Mismatch at index {i}: expected {expected[i]}, got {actual[i]}");
            }
        }

        private void TestMatrixOperations(float[,] data, string testName)
        {
            float[,] dataB = GenerateRandomMatrix(data.GetLength(0), data.GetLength(1));

            var avx = new AvxMatrix(data);
            var avxB = new AvxMatrix(dataB);
            var software = new Matrix2D(data);
            var softwareB = new Matrix2D(dataB);

            // Test addition
            var avxAdd = avx.Add(avxB);
            var softwareAdd = software.Add(softwareB);
            AssertMatricesEqual(softwareAdd, avxAdd, $"{testName} - Addition");

            // Test subtraction
            var avxSub = avx.Subtract(avxB);
            var softwareSub = software.Subtract(softwareB);
            AssertMatricesEqual(softwareSub, avxSub, $"{testName} - Subtraction");

            // Test scalar multiply
            var avxMul = avx.Multiply(2.5f);
            var softwareMul = software.Multiply(2.5f);
            AssertMatricesEqual(softwareMul, avxMul, $"{testName} - Scalar Multiply");
        }

        private void TestVectorOperations(float[] data, string testName)
        {
            float[] dataB = GenerateRandomVector(data.Length);

            var avx = new AvxColumnVector(data);
            var avxB = new AvxColumnVector(dataB);
            var software = new ColumnVector(data);
            var softwareB = new ColumnVector(dataB);

            // Test addition
            var avxAdd = avx.Add(avxB);
            var softwareAdd = software.Add(softwareB);
            AssertVectorsEqual(softwareAdd, avxAdd, $"{testName} - Addition");

            // Test subtraction
            var avxSub = avx.Subtract(avxB);
            var softwareSub = software.Subtract(softwareB);
            AssertVectorsEqual(softwareSub, avxSub, $"{testName} - Subtraction");

            // Test element-wise multiply
            var avxMul = avx.Multiply(avxB);
            var softwareMul = software.Multiply(softwareB);
            AssertVectorsEqual(softwareMul, avxMul, $"{testName} - Element-wise Multiply");

            // Test scalar multiply
            var avxScalar = avx.Multiply(3.0f);
            var softwareScalar = software.Multiply(3.0f);
            AssertVectorsEqual(softwareScalar, avxScalar, $"{testName} - Scalar Multiply");

            // Test sum
            float avxSum = avx.Sum();
            float softwareSum = software.Sum();
            Assert.AreEqual(softwareSum, avxSum, Tolerance, $"{testName} - Sum");
        }

        #endregion
    }
}
