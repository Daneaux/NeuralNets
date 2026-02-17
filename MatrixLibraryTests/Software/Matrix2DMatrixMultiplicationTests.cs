using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for Matrix2D matrix multiplication operations
    /// Phase 1: Core Matrix Operations - Matrix multiplication
    /// </summary>
    [TestClass]
    public class Matrix2DMatrixMultiplicationTests
    {
        private const float Tolerance = 1e-5f;

        #region Matrix Multiplication Tests

        [TestMethod]
        public void Matrix2D_Multiply_Matrix_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 5, 6 }, { 7, 8 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA.Multiply(matrixB);

            // Assert
            // [1 2]   [5 6]   [1*5+2*7  1*6+2*8]   [19 22]
            // [3 4] * [7 8] = [3*5+4*7  3*6+4*8] = [43 50]
            Assert.AreEqual(19, result[0, 0], Tolerance);
            Assert.AreEqual(22, result[0, 1], Tolerance);
            Assert.AreEqual(43, result[1, 0], Tolerance);
            Assert.AreEqual(50, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorMultiply_Matrix_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 5, 6 }, { 7, 8 } };
            MatrixBase matrixA = new Matrix2D(dataA);
            MatrixBase matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA * matrixB;

            // Assert
            Assert.AreEqual(19, result[0, 0], Tolerance);
            Assert.AreEqual(22, result[0, 1], Tolerance);
            Assert.AreEqual(43, result[1, 0], Tolerance);
            Assert.AreEqual(50, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_NonSquare_ReturnsCorrectProduct()
        {
            // Arrange - 2x3 * 3x2 = 2x2
            float[,] dataA = { { 1, 2, 3 }, { 4, 5, 6 } };
            float[,] dataB = { { 7, 8 }, { 9, 10 }, { 11, 12 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA.Multiply(matrixB);

            // Assert
            Assert.AreEqual(2, result.Rows);
            Assert.AreEqual(2, result.Cols);
            // [1 2 3]   [7  8 ]   [1*7+2*9+3*11   1*8+2*10+3*12 ]   [58  64 ]
            // [4 5 6] * [9  10] = [4*7+5*9+6*11   4*8+5*10+6*12 ] = [139 154]
            //           [11 12]
            Assert.AreEqual(58, result[0, 0], Tolerance);
            Assert.AreEqual(64, result[0, 1], Tolerance);
            Assert.AreEqual(139, result[1, 0], Tolerance);
            Assert.AreEqual(154, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_Identity_ReturnsSameMatrix()
        {
            // Arrange
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            float[,] identity = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            var matrix = new Matrix2D(data);
            var identityMatrix = new Matrix2D(identity);

            // Act
            var result = matrix.Multiply(identityMatrix);

            // Assert
            Assert.AreEqual(matrix.Rows, result.Rows);
            Assert.AreEqual(matrix.Cols, result.Cols);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    Assert.AreEqual(matrix[i, j], result[i, j], Tolerance);
                }
            }
        }

        [TestMethod]
        public void Matrix2D_Multiply_ZeroMatrix_ReturnsZeroMatrix()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] zeroData = { { 0, 0 }, { 0, 0 } };
            var matrixA = new Matrix2D(dataA);
            var zeroMatrix = new Matrix2D(zeroData);

            // Act
            var result = matrixA.Multiply(zeroMatrix);

            // Assert
            Assert.AreEqual(0, result[0, 0], Tolerance);
            Assert.AreEqual(0, result[0, 1], Tolerance);
            Assert.AreEqual(0, result[1, 0], Tolerance);
            Assert.AreEqual(0, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_IncompatibleDimensions_ThrowsException()
        {
            // Arrange - 2x3 * 2x2 (incompatible)
            float[,] dataA = { { 1, 2, 3 }, { 4, 5, 6 } };
            float[,] dataB = { { 1, 2 }, { 3, 4 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => matrixA.Multiply(matrixB));
        }

        #endregion

        #region Matrix-Vector Multiplication Tests

        [TestMethod]
        public void Matrix2D_MatrixTimesColumn_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] matrixData = { { 1, 2 }, { 3, 4 }, { 5, 6 } };
            float[] vectorData = { 7, 8 };
            var matrix = new Matrix2D(matrixData);
            var vector = new ColumnVector(vectorData);

            // Act
            var result = matrix.MatrixTimesColumn(vector);

            // Assert
            // [1 2]   [7]   [1*7+2*8]   [23]
            // [3 4] * [8] = [3*7+4*8] = [53]
            // [5 6]           [5*7+6*8]   [83]
            Assert.AreEqual(3, result.Size);
            Assert.AreEqual(23, result[0], Tolerance);
            Assert.AreEqual(53, result[1], Tolerance);
            Assert.AreEqual(83, result[2], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorMultiply_ColumnVector_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] matrixData = { { 1, 2 }, { 3, 4 } };
            float[] vectorData = { 5, 6 };
            MatrixBase matrix = new Matrix2D(matrixData);
            ColumnVectorBase vector = new ColumnVector(vectorData);

            // Act
            var result = matrix * vector;

            // Assert
            // [1 2]   [5]   [1*5+2*6]   [17]
            // [3 4] * [6] = [3*5+4*6] = [39]
            Assert.AreEqual(2, result.Size);
            Assert.AreEqual(17, result[0], Tolerance);
            Assert.AreEqual(39, result[1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_MatrixTimesColumn_ZeroVector_ReturnsZeroVector()
        {
            // Arrange
            float[,] matrixData = { { 1, 2 }, { 3, 4 } };
            float[] zeroVector = { 0, 0 };
            var matrix = new Matrix2D(matrixData);
            var vector = new ColumnVector(zeroVector);

            // Act
            var result = matrix.MatrixTimesColumn(vector);

            // Assert
            Assert.AreEqual(0, result[0], Tolerance);
            Assert.AreEqual(0, result[1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_MatrixTimesColumn_SingleElement()
        {
            // Arrange - 1x1 matrix times 1-element vector
            float[,] matrixData = { { 5 } };
            float[] vectorData = { 3 };
            var matrix = new Matrix2D(matrixData);
            var vector = new ColumnVector(vectorData);

            // Act
            var result = matrix.MatrixTimesColumn(vector);

            // Assert
            Assert.AreEqual(1, result.Size);
            Assert.AreEqual(15, result[0], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_MatrixTimesColumn_IncompatibleDimensions_ThrowsException()
        {
            // Arrange - 2x3 matrix with 2-element vector (incompatible)
            float[,] matrixData = { { 1, 2, 3 }, { 4, 5, 6 } };
            float[] vectorData = { 7, 8 };
            var matrix = new Matrix2D(matrixData);
            var vector = new ColumnVector(vectorData);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => matrix.MatrixTimesColumn(vector));
        }

        #endregion

        #region Hadamard Product Tests

        [TestMethod]
        public void Matrix2D_HadamardProduct_ReturnsElementWiseProduct()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 5, 6 }, { 7, 8 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA.HadamardProduct(matrixB);

            // Assert - element-wise multiplication, not matrix multiplication
            Assert.AreEqual(5, result[0, 0], Tolerance);   // 1*5
            Assert.AreEqual(12, result[0, 1], Tolerance);  // 2*6
            Assert.AreEqual(21, result[1, 0], Tolerance);  // 3*7
            Assert.AreEqual(32, result[1, 1], Tolerance);  // 4*8
        }

        [TestMethod]
        public void Matrix2D_HadamardProduct_DifferentFromMatrixMultiply()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 5, 6 }, { 7, 8 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act
            var hadamardResult = matrixA.HadamardProduct(matrixB);
            var matrixResult = matrixA.Multiply(matrixB);

            // Assert - verify they're different
            Assert.AreNotEqual(matrixResult[0, 0], hadamardResult[0, 0]);
            Assert.AreEqual(5, hadamardResult[0, 0], Tolerance);  // 1*5
            Assert.AreEqual(19, matrixResult[0, 0], Tolerance);   // 1*5+2*7
        }

        [TestMethod]
        public void Matrix2D_HadamardProduct_DifferentSizes_ThrowsException()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 1, 2, 3 }, { 4, 5, 6 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => matrixA.HadamardProduct(matrixB));
        }

        #endregion
    }
}
