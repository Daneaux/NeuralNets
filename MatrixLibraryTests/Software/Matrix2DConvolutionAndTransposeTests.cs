using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for Matrix2D convolution and transpose operations
    /// Phase 1: Core Matrix Operations - Convolution, Transpose, Special operations
    /// </summary>
    [TestClass]
    public class Matrix2DConvolutionAndTransposeTests
    {
        private const float Tolerance = 1e-5f;

        #region Convolution Tests

        [TestMethod]
        public void Matrix2D_Convolution_IdentityKernel_ReturnsSameMatrix()
        {
            // Arrange - identity kernel (1 in center, 0 elsewhere)
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            float[,] kernelData = { { 0, 0, 0 }, { 0, 1, 0 }, { 0, 0, 0 } };
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var result = matrix.Convolution(kernel);

            // Assert - valid convolution with identity should preserve inner values
            Assert.AreEqual(5, result[0, 0], Tolerance);  // Center element
        }

        [TestMethod]
        public void Matrix2D_Convolution_EdgeDetectionKernel()
        {
            // Arrange
            float[,] data = { { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } };
            float[,] kernelData = { { 0, -1, 0 }, { -1, 4, -1 }, { 0, -1, 0 } };  // Laplacian
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var result = matrix.Convolution(kernel);

            // Assert - uniform image should give 0 edges
            Assert.AreEqual(0, result[0, 0], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Convolution_ValidConvolution_ReducesSize()
        {
            // Arrange - 5x5 with 3x3 kernel = 3x3 output
            float[,] data = new float[5, 5];
            float[,] kernelData = new float[3, 3];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    data[i, j] = i * 5 + j + 1;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    kernelData[i, j] = 1;  // All ones

            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var result = matrix.Convolution(kernel);

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(3, result.Cols);
        }

        [TestMethod]
        public void Matrix2D_Convolution_1x1Kernel_ReturnsScaledMatrix()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            float[,] kernelData = { { 5 } };  // 1x1 kernel
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var result = matrix.Convolution(kernel);

            // Assert - 1x1 kernel acts like scalar multiplication
            Assert.AreEqual(2, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(5, result[0, 0], Tolerance);
            Assert.AreEqual(10, result[0, 1], Tolerance);
            Assert.AreEqual(15, result[1, 0], Tolerance);
            Assert.AreEqual(20, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Convolution_KernelLargerThanMatrix_ThrowsException()
        {
            // Arrange - 2x2 matrix with 3x3 kernel
            float[,] data = { { 1, 2 }, { 3, 4 } };
            float[,] kernelData = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => matrix.Convolution(kernel));
        }

        #endregion

        #region Full Convolution Tests

        [TestMethod]
        public void Matrix2D_ConvolutionFull_IncreasesSize()
        {
            // Arrange - 3x3 with 3x3 kernel = 5x5 output
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            float[,] kernelData = { { 1, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var result = matrix.ConvolutionFull(kernel);

            // Assert
            Assert.AreEqual(5, result.Rows);
            Assert.AreEqual(5, result.Cols);
        }

        [TestMethod]
        public void Matrix2D_ConvolutionFull_PreservesSum()
        {
            // Arrange - impulse kernel should shift values
            float[,] data = { { 1, 2 }, { 3, 4 } };
            float[,] kernelData = { { 1 } };  // Single impulse
            var matrix = new Matrix2D(data);
            var kernel = new Matrix2D(kernelData);

            // Act
            var result = matrix.ConvolutionFull(kernel);

            // Assert - sum should be preserved
            float inputSum = 1 + 2 + 3 + 4;
            float outputSum = 0;
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Cols; j++)
                    outputSum += result[i, j];
            Assert.AreEqual(inputSum, outputSum, Tolerance);
        }

        #endregion

        #region Transpose Tests

        [TestMethod]
        public void Matrix2D_GetTransposedMatrix_SwapsDimensions()
        {
            // Arrange
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.GetTransposedMatrix();

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(4, result[0, 1], Tolerance);
            Assert.AreEqual(2, result[1, 0], Tolerance);
            Assert.AreEqual(5, result[1, 1], Tolerance);
            Assert.AreEqual(3, result[2, 0], Tolerance);
            Assert.AreEqual(6, result[2, 1], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Transpose_SquareMatrix_SwapsValues()
        {
            // Arrange
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Transpose();

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(3, result.Cols);
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(4, result[0, 1], Tolerance);
            Assert.AreEqual(7, result[0, 2], Tolerance);
            Assert.AreEqual(2, result[1, 0], Tolerance);
            Assert.AreEqual(5, result[1, 1], Tolerance);
            Assert.AreEqual(8, result[1, 2], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Transpose_SingleRow_BecomesSingleColumn()
        {
            // Arrange
            float[,] data = { { 1, 2, 3, 4, 5 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Transpose();

            // Assert
            Assert.AreEqual(5, result.Rows);
            Assert.AreEqual(1, result.Cols);
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(2, result[1, 0], Tolerance);
            Assert.AreEqual(3, result[2, 0], Tolerance);
            Assert.AreEqual(4, result[3, 0], Tolerance);
            Assert.AreEqual(5, result[4, 0], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Transpose_SingleColumn_BecomesSingleRow()
        {
            // Arrange
            float[,] data = { { 1 }, { 2 }, { 3 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Transpose();

            // Assert
            Assert.AreEqual(1, result.Rows);
            Assert.AreEqual(3, result.Cols);
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(2, result[0, 1], Tolerance);
            Assert.AreEqual(3, result[0, 2], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Transpose_TwiceReturnsOriginal()
        {
            // Arrange
            float[,] data = { { 1, 2, 3 }, { 4, 5, 6 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Transpose().Transpose();

            // Assert
            Assert.AreEqual(matrix.Rows, result.Rows);
            Assert.AreEqual(matrix.Cols, result.Cols);
            for (int i = 0; i < matrix.Rows; i++)
                for (int j = 0; j < matrix.Cols; j++)
                    Assert.AreEqual(matrix[i, j], result[i, j], Tolerance);
        }

        #endregion

        #region Log and Sum Tests

        [TestMethod]
        public void Matrix2D_Log_ReturnsNaturalLog()
        {
            // Arrange
            float[,] data = { { 1, (float)Math.E }, { (float)Math.E * (float)Math.E, 10 } };
            var matrix = new Matrix2D(data);

            // Act
            var result = matrix.Log();

            // Assert
            Assert.AreEqual(0, result[0, 0], Tolerance);  // ln(1) = 0
            Assert.AreEqual(1, result[0, 1], Tolerance);  // ln(e) = 1
            Assert.AreEqual(2, result[1, 0], Tolerance);  // ln(e²) = 2
        }

        [TestMethod]
        public void Matrix2D_Sum_ReturnsCorrectTotal()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 } };
            var matrix = new Matrix2D(data);

            // Act
            float sum = matrix.Sum();

            // Assert
            Assert.AreEqual(10, sum, Tolerance);
        }

        [TestMethod]
        public void Matrix2D_Sum_EmptyMatrix_ReturnsZero()
        {
            // Arrange
            float[,] data = new float[0, 0];
            var matrix = new Matrix2D(data);

            // Act
            float sum = matrix.Sum();

            // Assert
            Assert.AreEqual(0, sum, Tolerance);
        }

        #endregion

        #region SetDiagonal Tests

        [TestMethod]
        public void Matrix2D_SetDiagonal_SetsCorrectValues()
        {
            // Arrange
            float[,] data = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
            var matrix = new Matrix2D(data);

            // Act
            matrix.SetDiagonal(5);

            // Assert
            Assert.AreEqual(5, matrix[0, 0], Tolerance);
            Assert.AreEqual(5, matrix[1, 1], Tolerance);
            Assert.AreEqual(5, matrix[2, 2], Tolerance);
            Assert.AreEqual(0, matrix[0, 1], Tolerance);  // Off-diagonal unchanged
            Assert.AreEqual(0, matrix[1, 0], Tolerance);
        }

        [TestMethod]
        public void Matrix2D_SetDiagonal_NonSquareMatrix_ThrowsException()
        {
            // Arrange
            float[,] data = { { 1, 2 }, { 3, 4 }, { 5, 6 } };  // 3x2
            var matrix = new Matrix2D(data);

            // Act & Assert
            Assert.ThrowsException<InvalidOperationException>(() => matrix.SetDiagonal(5));
        }

        #endregion
    }
}
