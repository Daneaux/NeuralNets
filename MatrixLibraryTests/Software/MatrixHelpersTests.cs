using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for MatrixHelpers utility methods
    /// Phase 5: Helper Methods
    /// </summary>
    [TestClass]
    public class MatrixHelpersTests
    {
        private const float Tolerance = 1e-5f;

        #region UnrollMatricesToColumnVector Tests

        [TestMethod]
        public void MatrixHelpers_UnrollMatricesToColumnVector_SingleMatrix_ReturnsCorrectVector()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);
            var matrices = new List<MatrixBase> { matrix };

            // Act
            var result = MatrixHelpers.UnrollMatricesToColumnVector(matrices);

            // Assert
            Assert.AreEqual(4, result.Size);
            Assert.AreEqual(1, result[0], Tolerance);
            Assert.AreEqual(2, result[1], Tolerance);
            Assert.AreEqual(3, result[2], Tolerance);
            Assert.AreEqual(4, result[3], Tolerance);
        }

        [TestMethod]
        public void MatrixHelpers_UnrollMatricesToColumnVector_MultipleMatrices_ReturnsCorrectVector()
        {
            // Arrange
            float[,] data1 = { { 1, 2 } };
            float[,] data2 = { { 3, 4 } };
            var matrix1 = new Matrix2D(data1);
            var matrix2 = new Matrix2D(data2);
            var matrices = new List<MatrixBase> { matrix1, matrix2 };

            // Act
            var result = MatrixHelpers.UnrollMatricesToColumnVector(matrices);

            // Assert
            Assert.AreEqual(4, result.Size);
            Assert.AreEqual(1, result[0], Tolerance);
            Assert.AreEqual(2, result[1], Tolerance);
            Assert.AreEqual(3, result[2], Tolerance);
            Assert.AreEqual(4, result[3], Tolerance);
        }

        [TestMethod]
        public void MatrixHelpers_UnrollMatricesToColumnVector_3DMatrices_ReturnsCorrectVector()
        {
            // Arrange - simulate CNN feature maps (3 matrices of 2x2)
            var matrices = new List<MatrixBase>
            {
                new Matrix2D(new float[,] { { 1, 2 }, { 3, 4 } }),
                new Matrix2D(new float[,] { { 5, 6 }, { 7, 8 } }),
                new Matrix2D(new float[,] { { 9, 10 }, { 11, 12 } })
            };

            // Act
            var result = MatrixHelpers.UnrollMatricesToColumnVector(matrices);

            // Assert - 3 matrices * 4 elements each = 12 elements
            Assert.AreEqual(12, result.Size);
            Assert.AreEqual(1, result[0], Tolerance);
            Assert.AreEqual(4, result[3], Tolerance);
            Assert.AreEqual(5, result[4], Tolerance);
            Assert.AreEqual(12, result[11], Tolerance);
        }

        [TestMethod]
        public void MatrixHelpers_UnrollMatricesToColumnVector_EmptyList_ReturnsEmptyVector()
        {
            // Arrange
            var matrices = new List<MatrixBase>();
            Assert.ThrowsException<ArgumentException>(() => MatrixHelpers.UnrollMatricesToColumnVector(matrices));
        }

        [TestMethod]
        public void MatrixHelpers_UnrollMatricesToColumnVector_OrderIsRowMajor()
        {
            // Arrange
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 } };
            var matrix = new Matrix2D(data);
            var matrices = new List<MatrixBase> { matrix };

            // Act
            var result = MatrixHelpers.UnrollMatricesToColumnVector(matrices);

            // Assert - row-major order: [1,2,3,4,5,6]
            Assert.AreEqual(6, result.Size);
            Assert.AreEqual(1, result[0], Tolerance);
            Assert.AreEqual(2, result[1], Tolerance);
            Assert.AreEqual(3, result[2], Tolerance);
            Assert.AreEqual(4, result[3], Tolerance);
            Assert.AreEqual(5, result[4], Tolerance);
            Assert.AreEqual(6, result[5], Tolerance);
        }

        #endregion

        #region ConvolutionSizeHelper Tests

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_3x3With3x3_Returns1x1()
        {
            // Arrange
            float[,] data = new float[3, 3];
            float[,] kernelData = new float[3, 3];
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(matrix, kernel);

            // Assert
            Assert.AreEqual(1, rows);
            Assert.AreEqual(1, cols);
        }

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_5x5With3x3_Returns3x3()
        {
            // Arrange
            float[,] data = new float[5, 5];
            float[,] kernelData = new float[3, 3];
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(matrix, kernel);

            // Assert - (5-3)+1 = 3
            Assert.AreEqual(3, rows);
            Assert.AreEqual(3, cols);
        }

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_5x5With3x3Stride2_Returns2x2()
        {
            // Arrange
            float[,] data = new float[5, 5];
            var matrix = new Matrix2D(data);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(matrix, 3, isFull: false, stride: 2);

            // Assert - (5-3)/2 + 1 = 2
            Assert.AreEqual(2, rows);
            Assert.AreEqual(2, cols);
        }

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_5x5With3x3Full_Returns7x7()
        {
            // Arrange
            float[,] data = new float[5, 5];
            var matrix = new Matrix2D(data);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(matrix, 3, isFull: true, stride: 1);

            // Assert - 5 + 3 - 1 = 7
            Assert.AreEqual(7, rows);
            Assert.AreEqual(7, cols);
        }

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_28x28With4x4_Returns25x25()
        {
            // Arrange - MNIST dimensions
            float[,] data = new float[28, 28];
            var matrix = new Matrix2D(data);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(matrix, 4, isFull: false, stride: 1);

            // Assert - (28-4)+1 = 25
            Assert.AreEqual(25, rows);
            Assert.AreEqual(25, cols);
        }

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_WithInputOutputShape()
        {
            // Arrange
            var shape = new InputOutputShape(28, 28, 1, 1);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(shape, 4, isFull: false, stride: 1);

            // Assert
            Assert.AreEqual(25, rows);
            Assert.AreEqual(25, cols);
        }

        [TestMethod]
        public void MatrixHelpers_ConvolutionSizeHelper_SinglePixelKernel()
        {
            // Arrange
            float[,] data = new float[5, 5];
            var matrix = new Matrix2D(data);

            // Act
            var (rows, cols) = MatrixHelpers.ConvolutionSizeHelper(matrix, 1, isFull: false, stride: 1);

            // Assert - 1x1 kernel preserves size
            Assert.AreEqual(5, rows);
            Assert.AreEqual(5, cols);
        }

        #endregion
    }
}
