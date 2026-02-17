using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests.Avx
{
    /// <summary>
    /// Comprehensive tests for AvxColumnVector operations
    /// Tests AVX-512 optimized vector implementations against software baseline
    /// </summary>
    [TestClass]
    public class AvxColumnVectorTests
    {
        private const float Tolerance = 1e-4f;

        #region Helper Methods

        private void AssertVectorsEqual(ColumnVectorBase expected, ColumnVectorBase actual, string message = "")
        {
            Assert.AreEqual(expected.Size, actual.Size, $"{message} - Size mismatch");

            for (int i = 0; i < expected.Size; i++)
            {
                Assert.AreEqual(expected[i], actual[i], Tolerance,
                    $"{message} - Mismatch at index {i}: expected {expected[i]}, got {actual[i]}");
            }
        }

        #endregion

        #region Addition Tests

        [TestMethod]
        public void AvxColumnVector_Add_Vector_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
            float[] dataB = { 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 };

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Vector addition");
        }

        [TestMethod]
        public void AvxColumnVector_Add_LargeVector_MatchesSoftware()
        {
            // Arrange - 64 elements (4 AVX vectors of 16 floats each)
            var random = new Random(42);
            float[] dataA = new float[64];
            float[] dataB = new float[64];
            for (int i = 0; i < 64; i++)
            {
                dataA[i] = (float)random.NextDouble() * 100;
                dataB[i] = (float)random.NextDouble() * 100;
            }

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Large vector addition");
        }

        [TestMethod]
        public void AvxColumnVector_Add_Scalar_MatchesSoftware()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };
            float scalar = 100.5f;

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Add(scalar);
            var softwareResult = software.Add(scalar);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Scalar addition");
        }

        [TestMethod]
        public void AvxColumnVector_Add_NonMultipleOf16_MatchesSoftware()
        {
            // Arrange - 17 elements to test remainder handling
            var random = new Random(42);
            float[] dataA = new float[17];
            float[] dataB = new float[17];
            for (int i = 0; i < 17; i++)
            {
                dataA[i] = (float)random.NextDouble() * 10;
                dataB[i] = (float)random.NextDouble() * 10;
            }

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Non-multiple of 16 vector addition");
        }

        #endregion

        #region Subtraction Tests

        [TestMethod]
        public void AvxColumnVector_Subtract_Vector_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { 100, 200, 300, 400 };
            float[] dataB = { 10, 20, 30, 40 };

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Vector subtraction");
        }

        [TestMethod]
        public void AvxColumnVector_Subtract_Scalar_MatchesSoftware()
        {
            // Arrange
            float[] data = { 100, 200, 300, 400, 500 };
            float scalar = 50;

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Subtract(scalar);
            var softwareResult = software.Subtract(scalar);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Scalar subtraction");
        }

        [TestMethod]
        public void AvxColumnVector_Subtract_NegativeValues_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { -10, -20, -30 };
            float[] dataB = { -5, -15, -25 };

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Negative value subtraction");
        }

        #endregion

        #region Multiplication Tests

        [TestMethod]
        public void AvxColumnVector_Multiply_Vector_MatchesSoftware()
        {
            // Arrange - element-wise multiplication
            float[] dataA = { 2, 3, 4, 5 };
            float[] dataB = { 5, 6, 7, 8 };

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Multiply(avxB);
            var softwareResult = softwareA.Multiply(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Element-wise multiplication");
        }

        [TestMethod]
        public void AvxColumnVector_Multiply_Scalar_MatchesSoftware()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };
            float scalar = 2.5f;

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Scalar multiplication");
        }

        [TestMethod]
        public void AvxColumnVector_Multiply_LargeVector_MatchesSoftware()
        {
            // Arrange - 128 elements (8 AVX vectors)
            var random = new Random(123);
            float[] dataA = new float[128];
            float[] dataB = new float[128];
            for (int i = 0; i < 128; i++)
            {
                dataA[i] = (float)random.NextDouble() * 10;
                dataB[i] = (float)random.NextDouble() * 10;
            }

            var avxA = new AvxColumnVector(dataA);
            var avxB = new AvxColumnVector(dataB);
            var softwareA = new ColumnVector(dataA);
            var softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA.Multiply(avxB);
            var softwareResult = softwareA.Multiply(softwareB);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Large vector multiplication");
        }

        [TestMethod]
        public void AvxColumnVector_Multiply_ZeroScalar_MatchesSoftware()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Multiply(0);
            var softwareResult = software.Multiply(0);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Zero scalar multiplication");
        }

        [TestMethod]
        public void AvxColumnVector_Multiply_NegativeScalar_MatchesSoftware()
        {
            // Arrange
            float[] data = { 10, 20, 30 };
            float scalar = -2;

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Negative scalar multiplication");
        }

        #endregion

        #region Operator Tests

        [TestMethod]
        public void AvxColumnVector_OperatorPlus_Vector_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { 1, 2, 3, 4 };
            float[] dataB = { 10, 20, 30, 40 };

            ColumnVectorBase avxA = new AvxColumnVector(dataA);
            ColumnVectorBase avxB = new AvxColumnVector(dataB);
            ColumnVectorBase softwareA = new ColumnVector(dataA);
            ColumnVectorBase softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA + avxB;
            var softwareResult = softwareA + softwareB;

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Operator + (vector)");
        }

        [TestMethod]
        public void AvxColumnVector_OperatorMinus_Vector_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { 100, 200, 300 };
            float[] dataB = { 10, 20, 30 };

            ColumnVectorBase avxA = new AvxColumnVector(dataA);
            ColumnVectorBase avxB = new AvxColumnVector(dataB);
            ColumnVectorBase softwareA = new ColumnVector(dataA);
            ColumnVectorBase softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA - avxB;
            var softwareResult = softwareA - softwareB;

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Operator - (vector)");
        }

        [TestMethod]
        public void AvxColumnVector_OperatorMultiply_Vector_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { 2, 4, 6 };
            float[] dataB = { 3, 5, 7 };

            ColumnVectorBase avxA = new AvxColumnVector(dataA);
            ColumnVectorBase avxB = new AvxColumnVector(dataB);
            ColumnVectorBase softwareA = new ColumnVector(dataA);
            ColumnVectorBase softwareB = new ColumnVector(dataB);

            // Act
            var avxResult = avxA * avxB;
            var softwareResult = softwareA * softwareB;

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Operator * (vector)");
        }

        [TestMethod]
        public void AvxColumnVector_OperatorMultiply_Scalar_MatchesSoftware()
        {
            // Arrange
            float[] data = { 5, 10, 15 };
            float scalar = 3;

            ColumnVectorBase avx = new AvxColumnVector(data);
            ColumnVectorBase software = new ColumnVector(data);

            // Act
            var avxResult = avx * scalar;
            var softwareResult = software * scalar;

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Operator * (scalar right)");
        }

        [TestMethod]
        public void AvxColumnVector_OperatorMultiply_ScalarLeft_MatchesSoftware()
        {
            // Arrange
            float[] data = { 3, 6, 9 };
            float scalar = 4;

            ColumnVectorBase avx = new AvxColumnVector(data);
            ColumnVectorBase software = new ColumnVector(data);

            // Act
            var avxResult = scalar * avx;
            var softwareResult = scalar * software;

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Operator * (scalar left)");
        }

        #endregion

        #region Special Operations

        [TestMethod]
        public void AvxColumnVector_Sum_MatchesSoftware()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            float avxResult = avx.Sum();
            float softwareResult = software.Sum();

            // Assert
            Assert.AreEqual(softwareResult, avxResult, Tolerance, "Sum operation");
            Assert.AreEqual(136, avxResult, Tolerance); // 1+2+...+16 = 136
        }

        [TestMethod]
        public void AvxColumnVector_Sum_LargeVector_MatchesSoftware()
        {
            // Arrange - 128 elements
            var random = new Random(42);
            float[] data = new float[128];
            for (int i = 0; i < 128; i++)
                data[i] = (float)random.NextDouble() * 10;

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            float avxResult = avx.Sum();
            float softwareResult = software.Sum();

            // Assert
            Assert.AreEqual(softwareResult, avxResult, Tolerance, "Large vector sum");
        }

        [TestMethod]
        public void AvxColumnVector_OuterProduct_MatchesSoftware()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] dataB = { 4, 5 };

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

        [TestMethod]
        public void AvxColumnVector_OuterProduct_Large_MatchesSoftware()
        {
            // Arrange - 32x32 outer product
            var random = new Random(42);
            float[] dataA = new float[32];
            float[] dataB = new float[32];
            for (int i = 0; i < 32; i++)
            {
                dataA[i] = (float)random.NextDouble() * 10;
                dataB[i] = (float)random.NextDouble() * 10;
            }

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

        [TestMethod]
        public void AvxColumnVector_GetMax_MatchesSoftware()
        {
            // Arrange
            float[] data = { 3, 1, 4, 1, 5, 9, 2, 6, 8, 7 };

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            float avxResult = avx.GetMax();
            float softwareResult = software.GetMax();

            // Assert
            Assert.AreEqual(softwareResult, avxResult, Tolerance, "Max operation");
            Assert.AreEqual(9, avxResult, Tolerance);
        }

        [TestMethod]
        public void AvxColumnVector_GetMax_LargeVector_MatchesSoftware()
        {
            // Arrange - 128 elements with max in the middle
            float[] data = new float[128];
            for (int i = 0; i < 128; i++)
                data[i] = i < 64 ? (float)i : (float)(128 - i);
            data[65] = 1000; // Set max

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            float avxResult = avx.GetMax();
            float softwareResult = software.GetMax();

            // Assert
            Assert.AreEqual(softwareResult, avxResult, Tolerance, "Large vector max");
            Assert.AreEqual(1000, avxResult, Tolerance);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void AvxColumnVector_EmptyVector_MatchesSoftware()
        {
            // Arrange
            float[] data = new float[0];

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            float avxSum = avx.Sum();
            float softwareSum = software.Sum();

            // Assert
            Assert.AreEqual(softwareSum, avxSum, Tolerance, "Empty vector sum");
        }

        [TestMethod]
        public void AvxColumnVector_SingleElement_MatchesSoftware()
        {
            // Arrange
            float[] data = { 42 };

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Multiply(2);
            var softwareResult = software.Multiply(2);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Single element operations");
        }

        [TestMethod]
        public void AvxColumnVector_NegativeValues_MatchesSoftware()
        {
            // Arrange
            float[] data = { -1, -2, -3, -4, -5 };

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Add(10);
            var softwareResult = software.Add(10);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Negative values");
        }

        [TestMethod]
        public void AvxColumnVector_ZeroVector_MatchesSoftware()
        {
            // Arrange
            float[] data = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

            var avx = new AvxColumnVector(data);
            var software = new ColumnVector(data);

            // Act
            var avxResult = avx.Multiply(100);
            var softwareResult = software.Multiply(100);

            // Assert
            AssertVectorsEqual(softwareResult, avxResult, "Zero vector");
        }

        #endregion
    }
}
