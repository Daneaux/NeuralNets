using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for Matrix2D basic arithmetic operations
    /// Phase 1: Core Matrix Operations - Addition, Subtraction, Scalar operations
    /// </summary>
    [TestClass]
    public class Matrix2DBasicArithmeticTests
    {
        private const float Tolerance = 1e-5f;

        #region Addition Tests

        [TestMethod]
        public void Matrix2D_Add_Matrix_ReturnsCorrectSum()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 5, 6 }, { 7, 8 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA.Add(matrixB);

            // Assert
            Assert.AreEqual(6, result[0, 0], Tolerance);
            Assert.AreEqual(8, result[0, 1], Tolerance);
            Assert.AreEqual(10, result[1, 0], Tolerance);
            Assert.AreEqual(12, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Add_Scalar_ReturnsCorrectSum()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Add(10);

            // Assert
            Assert.AreEqual(11, result[0, 0], Tolerance);
            Assert.AreEqual(12, result[0, 1], Tolerance);
            Assert.AreEqual(13, result[1, 0], Tolerance);
            Assert.AreEqual(14, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorPlus_Matrix_ReturnsCorrectSum()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 5, 6 }, { 7, 8 } };
            MatrixBase matrixA = new Matrix2D(dataA);
            MatrixBase matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA + matrixB;

            // Assert
            Assert.AreEqual(6, result[0, 0], Tolerance);
            Assert.AreEqual(8, result[0, 1], Tolerance);
            Assert.AreEqual(10, result[1, 0], Tolerance);
            Assert.AreEqual(12, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorPlus_Scalar_ReturnsCorrectSum()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            MatrixBase matrix = new Matrix2D(data);

            // Act
            var result = matrix + 10;

            // Assert
            Assert.AreEqual(11, result[0, 0], Tolerance);
            Assert.AreEqual(12, result[0, 1], Tolerance);
            Assert.AreEqual(13, result[1, 0], Tolerance);
            Assert.AreEqual(14, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Add_NegativeScalar_ReturnsCorrectSum()
        {
            // Arrange
            float[,] data = { { 10, 20 }, { 30, 40 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Add(-5);

            // Assert
            Assert.AreEqual(5, result[0, 0], Tolerance);
            Assert.AreEqual(15, result[0, 1], Tolerance);
            Assert.AreEqual(25, result[1, 0], Tolerance);
            Assert.AreEqual(35, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Add_ZeroScalar_ReturnsSameMatrix()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Add(0);

            // Assert
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(2, result[0, 1], Tolerance);
            Assert.AreEqual(3, result[1, 0], Tolerance);
            Assert.AreEqual(4, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Add_DifferentSizes_ThrowsException()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 1, 2, 3 }, { 4, 5, 6 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => matrixA.Add(matrixB));
        }

        #endregion

        #region Subtraction Tests

        [TestMethod]
        public void Matrix2D_Subtract_Matrix_ReturnsCorrectDifference()
        {
            // Arrange
            float[,] dataA = { { 10, 20 }, { 30, 40 } };
            float[,] dataB = { { 1, 2 }, { 3, 4 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA.Subtract(matrixB);

            // Assert
            Assert.AreEqual(9, result[0, 0], Tolerance);
            Assert.AreEqual(18, result[0, 1], Tolerance);
            Assert.AreEqual(27, result[1, 0], Tolerance);
            Assert.AreEqual(36, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Subtract_Self_ReturnsZeroMatrix()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Subtract(matrix);

            // Assert
            Assert.AreEqual(0, result[0, 0], Tolerance);
            Assert.AreEqual(0, result[0, 1], Tolerance);
            Assert.AreEqual(0, result[1, 0], Tolerance);
            Assert.AreEqual(0, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorMinus_Matrix_ReturnsCorrectDifference()
        {
            // Arrange
            float[,] dataA = { { 10, 20 }, { 30, 40 } };
            float[,] dataB = { { 1, 2 }, { 3, 4 } };
            MatrixBase matrixA = new Matrix2D(dataA);
            MatrixBase matrixB = new Matrix2D(dataB);

            // Act
            var result = matrixA - matrixB;

            // Assert
            Assert.AreEqual(9, result[0, 0], Tolerance);
            Assert.AreEqual(18, result[0, 1], Tolerance);
            Assert.AreEqual(27, result[1, 0], Tolerance);
            Assert.AreEqual(36, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Subtract_DifferentSizes_ThrowsException()
        {
            // Arrange
            float[,] dataA = { { 1, 2 }, { 3, 4 } };
            float[,] dataB = { { 1, 2, 3 }, { 4, 5, 6 } };
            var matrixA = new Matrix2D(dataA);
            var matrixB = new Matrix2D(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => matrixA.Subtract(matrixB));
        }

        #endregion

        #region Scalar Multiplication Tests

        [TestMethod]
        public void Matrix2D_Multiply_Scalar_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Multiply(2);

            // Assert
            Assert.AreEqual(2, result[0, 0], Tolerance);
            Assert.AreEqual(4, result[0, 1], Tolerance);
            Assert.AreEqual(6, result[1, 0], Tolerance);
            Assert.AreEqual(8, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorMultiply_Scalar_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            MatrixBase matrix = new Matrix2D(data);

            // Act
            var result = matrix * 3;

            // Assert
            Assert.AreEqual(3, result[0, 0], Tolerance);
            Assert.AreEqual(6, result[0, 1], Tolerance);
            Assert.AreEqual(9, result[1, 0], Tolerance);
            Assert.AreEqual(12, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_OperatorMultiply_ScalarLeft_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            MatrixBase matrix = new Matrix2D(data);

            // Act
            var result = 4 * matrix;

            // Assert
            Assert.AreEqual(4, result[0, 0], Tolerance);
            Assert.AreEqual(8, result[0, 1], Tolerance);
            Assert.AreEqual(12, result[1, 0], Tolerance);
            Assert.AreEqual(16, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_ZeroScalar_ReturnsZeroMatrix()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Multiply(0);

            // Assert
            Assert.AreEqual(0, result[0, 0], Tolerance);
            Assert.AreEqual(0, result[0, 1], Tolerance);
            Assert.AreEqual(0, result[1, 0], Tolerance);
            Assert.AreEqual(0, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_NegativeScalar_ReturnsCorrectProduct()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Multiply(-2);

            // Assert
            Assert.AreEqual(-2, result[0, 0], Tolerance);
            Assert.AreEqual(-4, result[0, 1], Tolerance);
            Assert.AreEqual(-6, result[1, 0], Tolerance);
            Assert.AreEqual(-8, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_OneScalar_ReturnsSameMatrix()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Multiply(1);

            // Assert
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(2, result[0, 1], Tolerance);
            Assert.AreEqual(3, result[1, 0], Tolerance);
            Assert.AreEqual(4, result[1, 1], Tolerance);
        }

        #endregion

        #region Immutability Tests

        [TestMethod]
        public void Matrix2D_Add_DoesNotModifyOriginal()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);
            var originalValue = matrix[0, 0];

            // Act
            var result = matrix.Add(100);

            // Assert
            Assert.AreEqual(originalValue, matrix[0, 0], Tolerance);
            Assert.AreNotEqual(originalValue, result[0, 0], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Multiply_DoesNotModifyOriginal()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);
            var originalValue = matrix[0, 0];

            // Act
            var result = matrix.Multiply(100);

            // Assert
            Assert.AreEqual(originalValue, matrix[0, 0], Tolerance);
            Assert.AreNotEqual(originalValue, result[0, 0], Tolerance);
        }

        #endregion
    }
}
