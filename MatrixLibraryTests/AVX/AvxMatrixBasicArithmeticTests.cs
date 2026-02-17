using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests.Avx
{
    /// <summary>
    /// Comprehensive tests for AvxMatrix basic arithmetic operations
    /// Tests AVX-512 optimized implementations against software baseline
    /// </summary>
    [TestClass]
    public class AvxMatrixBasicArithmeticTests
    {
        private const float Tolerance = 1e-4f;

        #region Helper Methods

        private AvxMatrix CreateAvxMatrix(float[,] data)
        {
            return new AvxMatrix(data);
        }

        private Matrix2D CreateSoftwareMatrix(float[,] data)
        {
            return new Matrix2D(data);
        }

        private void AssertMatricesEqual(MatrixBase expected, MatrixBase actual, string message = "")
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

        #endregion

        #region Addition Tests

        [TestMethod]
        public void AvxMatrix_Add_Matrix_MatchesSoftware()
        {
            // Arrange
            float[,] dataA = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 }, { 13, 14, 15, 16 } };
            float[,] dataB = { { 16, 15, 14, 13 }, { 12, 11, 10, 9 }, { 8, 7, 6, 5 }, { 4, 3, 2, 1 } };

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Matrix addition");
        }

        [TestMethod]
        public void AvxMatrix_Add_Scalar_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            float scalar = 100.5f;

            var avx = CreateAvxMatrix(data);
            var software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = avx.Add(scalar);
            var softwareResult = software.Add(scalar);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Scalar addition");
        }

        [TestMethod]
        public void AvxMatrix_Add_LargeMatrix_MatchesSoftware()
        {
            // Arrange - 32x32 to test multiple AVX vectors (16 floats each)
            var random = new Random(42);
            float[,] dataA = new float[32, 32];
            float[,] dataB = new float[32, 32];
            for (int i = 0; i < 32; i++)
                for (int j = 0; j < 32; j++)
                {
                    dataA[i, j] = (float)random.NextDouble() * 100;
                    dataB[i, j] = (float)random.NextDouble() * 100;
                }

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Large matrix addition");
        }

        [TestMethod]
        public void AvxMatrix_Add_NonMultipleOf16_MatchesSoftware()
        {
            // Arrange - 17x17 to test remainder handling (17 % 16 = 1)
            var random = new Random(42);
            float[,] dataA = new float[17, 17];
            float[,] dataB = new float[17, 17];
            for (int i = 0; i < 17; i++)
                for (int j = 0; j < 17; j++)
                {
                    dataA[i, j] = (float)random.NextDouble() * 10;
                    dataB[i, j] = (float)random.NextDouble() * 10;
                }

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Non-multiple of 16 matrix addition");
        }

        [TestMethod]
        public void AvxMatrix_Add_SmallMatrix_MatchesSoftware()
        {
            // Arrange - 3x3 small matrix
            float[,] dataA = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            float[,] dataB = { { 9, 8, 7 }, { 6, 5, 4 }, { 3, 2, 1 } };

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Small matrix addition");
        }

        #endregion

        #region Subtraction Tests

        [TestMethod]
        public void AvxMatrix_Subtract_Matrix_MatchesSoftware()
        {
            // Arrange
            float[,] dataA = { { 100, 200, 300 }, { 400, 500, 600 } };
            float[,] dataB = { { 10, 20, 30 }, { 40, 50, 60 } };

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Matrix subtraction");
        }

        [TestMethod]
        public void AvxMatrix_Subtract_LargeMatrix_MatchesSoftware()
        {
            // Arrange - 64x64 matrix
            var random = new Random(123);
            float[,] dataA = new float[64, 64];
            float[,] dataB = new float[64, 64];
            for (int i = 0; i < 64; i++)
                for (int j = 0; j < 64; j++)
                {
                    dataA[i, j] = (float)random.NextDouble() * 1000;
                    dataB[i, j] = (float)random.NextDouble() * 1000;
                }

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Large matrix subtraction");
        }

        [TestMethod]
        public void AvxMatrix_Subtract_NegativeValues_MatchesSoftware()
        {
            // Arrange
            float[,] dataA = { { -10, -20 }, { -30, -40 } };
            float[,] dataB = { { -5, -15 }, { -25, -35 } };

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Subtract(avxB);
            var softwareResult = softwareA.Subtract(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Negative value subtraction");
        }

        #endregion

        #region Scalar Multiplication Tests

        [TestMethod]
        public void AvxMatrix_Multiply_Scalar_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } };
            float scalar = 3.5f;

            var avx = CreateAvxMatrix(data);
            var software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Scalar multiplication");
        }

        [TestMethod]
        public void AvxMatrix_Multiply_ZeroScalar_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 }, { 5, 6 }, { 7, 8 } };

            var avx = CreateAvxMatrix(data);
            var software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = avx.Multiply(0);
            var softwareResult = software.Multiply(0);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Zero scalar multiplication");
        }

        [TestMethod]
        public void AvxMatrix_Multiply_NegativeScalar_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 10, 20 }, { 30, 40 } };
            float scalar = -2.5f;

            var avx = CreateAvxMatrix(data);
            var software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Negative scalar multiplication");
        }

        [TestMethod]
        public void AvxMatrix_Multiply_LargeScalar_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 0.1f, 0.2f }, { 0.3f, 0.4f } };
            float scalar = 1000.0f;

            var avx = CreateAvxMatrix(data);
            var software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = avx.Multiply(scalar);
            var softwareResult = software.Multiply(scalar);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Large scalar multiplication");
        }

        #endregion

        #region Operator Tests

        [TestMethod]
        public void AvxMatrix_OperatorPlus_MatchesSoftware()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 10, 20 }, { 30, 40 } };

            MatrixBase avxA = CreateAvxMatrix(dataA);
            MatrixBase avxB = CreateAvxMatrix(dataB);
            MatrixBase softwareA = CreateSoftwareMatrix(dataA);
            MatrixBase softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA + avxB;
            var softwareResult = softwareA + softwareB;

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Operator + (matrix)");
        }

        [TestMethod]
        public void AvxMatrix_OperatorMinus_MatchesSoftware()
        {
            // Arrange
            float[,] dataA = { { 100, 200 }, { 300, 400 } };
            float[,] dataB = { { 10, 20 }, { 30, 40 } };

            MatrixBase avxA = CreateAvxMatrix(dataA);
            MatrixBase avxB = CreateAvxMatrix(dataB);
            MatrixBase softwareA = CreateSoftwareMatrix(dataA);
            MatrixBase softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA - avxB;
            var softwareResult = softwareA - softwareB;

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Operator - (matrix)");
        }

        [TestMethod]
        public void AvxMatrix_OperatorMultiplyScalar_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 5, 10 }, { 15, 20 } };
            float scalar = 2.5f;

            MatrixBase avx = CreateAvxMatrix(data);
            MatrixBase software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = avx * scalar;
            var softwareResult = software * scalar;

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Operator * (scalar right)");
        }

        [TestMethod]
        public void AvxMatrix_OperatorMultiplyScalarLeft_MatchesSoftware()
        {
            // Arrange
            float[,] data = { { 3, 6 }, { 9, 12 } };
            float scalar = 3.0f;

            MatrixBase avx = CreateAvxMatrix(data);
            MatrixBase software = CreateSoftwareMatrix(data);

            // Act
            var avxResult = scalar * avx;
            var softwareResult = scalar * software;

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Operator * (scalar left)");
        }

        #endregion

        #region Immutability Tests

        [TestMethod]
        public void AvxMatrix_Add_DoesNotModifyOriginal()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var avx = CreateAvxMatrix(data);
            float originalValue = avx[0, 0];

            // Act
            var result = avx.Add(1000);

            // Assert
            Assert.AreEqual(originalValue, avx[0, 0], Tolerance, "Original matrix should not be modified");
            Assert.AreNotEqual(originalValue, result[0, 0], Tolerance, "Result should be different");
        }

        [TestMethod]
        public void AvxMatrix_Multiply_DoesNotModifyOriginal()
        {
            // Arrange
            float[,] data = { { 5, 10 }, { 15, 20 } };
            var avx = CreateAvxMatrix(data);
            float originalValue = avx[1, 1];

            // Act
            var result = avx.Multiply(100);

            // Assert
            Assert.AreEqual(originalValue, avx[1, 1], Tolerance, "Original matrix should not be modified");
            Assert.AreNotEqual(originalValue, result[1, 1], Tolerance, "Result should be different");
        }

        #endregion

        #region Edge Case Tests

        [TestMethod]
        public void AvxMatrix_SingleRow_MatchesSoftware()
        {
            // Arrange - 1x16 row vector
            float[,] dataA = new float[1, 16];
            float[,] dataB = new float[1, 16];
            for (int j = 0; j < 16; j++)
            {
                dataA[0, j] = j + 1;
                dataB[0, j] = (j + 1) * 2;
            }

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Single row matrix");
        }

        [TestMethod]
        public void AvxMatrix_SingleColumn_MatchesSoftware()
        {
            // Arrange - 16x1 column vector as matrix
            float[,] dataA = new float[16, 1];
            float[,] dataB = new float[16, 1];
            for (int i = 0; i < 16; i++)
            {
                dataA[i, 0] = i + 1;
                dataB[i, 0] = (i + 1) * 0.5f;
            }

            var avxA = CreateAvxMatrix(dataA);
            var avxB = CreateAvxMatrix(dataB);
            var softwareA = CreateSoftwareMatrix(dataA);
            var softwareB = CreateSoftwareMatrix(dataB);

            // Act
            var avxResult = avxA.Add(avxB);
            var softwareResult = softwareA.Add(softwareB);

            // Assert
            AssertMatricesEqual(softwareResult, avxResult, "Single column matrix");
        }

        #endregion
    }
}
