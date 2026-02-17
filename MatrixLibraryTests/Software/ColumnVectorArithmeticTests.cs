using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for ColumnVector arithmetic operations
    /// Phase 2: ColumnVector Operations
    /// </summary>
    [TestClass]
    public class ColumnVectorArithmeticTests
    {
        private const float Tolerance = 1e-5f;

        #region Addition Tests

        [TestMethod]
        public void ColumnVector_Add_Vector_ReturnsCorrectSum()
        {
            // Arrange
            float[] dataA = { 1, 2, 3, 4, 5 };
            float[] dataB = { 10, 20, 30, 40, 50 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act
            var result = vecA.Add(vecB);

            // Assert
            Assert.AreEqual(11, result[0], Tolerance);
            Assert.AreEqual(22, result[1], Tolerance);
            Assert.AreEqual(33, result[2], Tolerance);
            Assert.AreEqual(44, result[3], Tolerance);
            Assert.AreEqual(55, result[4], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Add_Scalar_ReturnsCorrectSum()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Add(100);

            // Assert
            Assert.AreEqual(101, result[0], Tolerance);
            Assert.AreEqual(102, result[1], Tolerance);
            Assert.AreEqual(103, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorPlus_Vector_ReturnsCorrectSum()
        {
            // Arrange
            float[] dataA = { 5, 10, 15 };
            float[] dataB = { 1, 2, 3 };
            ColumnVectorBase vecA = new ColumnVector(dataA);
            ColumnVectorBase vecB = new ColumnVector(dataB);

            // Act
            var result = vecA + vecB;

            // Assert
            Assert.AreEqual(6, result[0], Tolerance);
            Assert.AreEqual(12, result[1], Tolerance);
            Assert.AreEqual(18, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorPlus_Scalar_ReturnsCorrectSum()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            ColumnVectorBase vector = new ColumnVector(data);

            // Act
            var result = vector + 10;

            // Assert
            Assert.AreEqual(11, result[0], Tolerance);
            Assert.AreEqual(12, result[1], Tolerance);
            Assert.AreEqual(13, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorPlus_ScalarLeft_ReturnsCorrectSum()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            ColumnVectorBase vector = new ColumnVector(data);

            // Act
            var result = 10 + vector;

            // Assert
            Assert.AreEqual(11, result[0], Tolerance);
            Assert.AreEqual(12, result[1], Tolerance);
            Assert.AreEqual(13, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Add_NegativeScalar_ReturnsCorrectSum()
        {
            // Arrange
            float[] data = { 10, 20, 30 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Add(-5);

            // Assert
            Assert.AreEqual(5, result[0], Tolerance);
            Assert.AreEqual(15, result[1], Tolerance);
            Assert.AreEqual(25, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Add_ZeroScalar_ReturnsSameVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Add(0);

            // Assert
            Assert.AreEqual(1, result[0], Tolerance);
            Assert.AreEqual(2, result[1], Tolerance);
            Assert.AreEqual(3, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Add_DifferentSizes_ThrowsException()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] dataB = { 1, 2 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => vecA.Add(vecB));
        }

        #endregion

        #region Subtraction Tests

        [TestMethod]
        public void ColumnVector_Subtract_Vector_ReturnsCorrectDifference()
        {
            // Arrange
            float[] dataA = { 10, 20, 30 };
            float[] dataB = { 1, 2, 3 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act
            var result = vecA.Subtract(vecB);

            // Assert
            Assert.AreEqual(9, result[0], Tolerance);
            Assert.AreEqual(18, result[1], Tolerance);
            Assert.AreEqual(27, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Subtract_Scalar_ReturnsCorrectDifference()
        {
            // Arrange
            float[] data = { 10, 20, 30 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Subtract(5);

            // Assert
            Assert.AreEqual(5, result[0], Tolerance);
            Assert.AreEqual(15, result[1], Tolerance);
            Assert.AreEqual(25, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorMinus_Vector_ReturnsCorrectDifference()
        {
            // Arrange
            float[] dataA = { 10, 20, 30 };
            float[] dataB = { 1, 2, 3 };
            ColumnVectorBase vecA = new ColumnVector(dataA);
            ColumnVectorBase vecB = new ColumnVector(dataB);

            // Act
            var result = vecA - vecB;

            // Assert
            Assert.AreEqual(9, result[0], Tolerance);
            Assert.AreEqual(18, result[1], Tolerance);
            Assert.AreEqual(27, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorMinus_ScalarLeft_ReturnsReversedDifference()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            ColumnVectorBase vector = new ColumnVector(data);

            // Act - 10 - vector = [10-1, 10-2, 10-3]
            var result = 10 - vector;

            // Assert
            Assert.AreEqual(9, result[0], Tolerance);
            Assert.AreEqual(8, result[1], Tolerance);
            Assert.AreEqual(7, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorMinus_ScalarRight_ReturnsCorrectDifference()
        {
            // Arrange
            float[] data = { 10, 20, 30 };
            ColumnVectorBase vector = new ColumnVector(data);

            // Act - vector - 5
            var result = vector - 5;

            // Assert
            Assert.AreEqual(5, result[0], Tolerance);
            Assert.AreEqual(15, result[1], Tolerance);
            Assert.AreEqual(25, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Subtract_Self_ReturnsZeroVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Subtract(vector);

            // Assert
            Assert.AreEqual(0, result[0], Tolerance);
            Assert.AreEqual(0, result[1], Tolerance);
            Assert.AreEqual(0, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Subtract_DifferentSizes_ThrowsException()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] dataB = { 1, 2 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => vecA.Subtract(vecB));
        }

        #endregion

        #region Element-wise Multiplication (Hadamard) Tests

        [TestMethod]
        public void ColumnVector_Multiply_Vector_ReturnsElementWiseProduct()
        {
            // Arrange
            float[] dataA = { 2, 3, 4 };
            float[] dataB = { 5, 6, 7 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act
            var result = vecA.Multiply(vecB);

            // Assert
            Assert.AreEqual(10, result[0], Tolerance);  // 2*5
            Assert.AreEqual(18, result[1], Tolerance);  // 3*6
            Assert.AreEqual(28, result[2], Tolerance);  // 4*7
        }

        [TestMethod]
        public void ColumnVector_OperatorMultiply_Vector_ReturnsElementWiseProduct()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] dataB = { 4, 5, 6 };
            ColumnVectorBase vecA = new ColumnVector(dataA);
            ColumnVectorBase vecB = new ColumnVector(dataB);

            // Act
            var result = vecA * vecB;

            // Assert
            Assert.AreEqual(4, result[0], Tolerance);
            Assert.AreEqual(10, result[1], Tolerance);
            Assert.AreEqual(18, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_ZeroVector_ReturnsZeroVector()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] zeroData = { 0, 0, 0 };
            var vecA = new ColumnVector(dataA);
            var zeroVec = new ColumnVector(zeroData);

            // Act
            var result = vecA.Multiply(zeroVec);

            // Assert
            Assert.AreEqual(0, result[0], Tolerance);
            Assert.AreEqual(0, result[1], Tolerance);
            Assert.AreEqual(0, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_DifferentSizes_ThrowsException()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] dataB = { 1, 2 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act & Assert
            Assert.ThrowsException<ArgumentException>(() => vecA.Multiply(vecB));
        }

        #endregion

        #region Scalar Multiplication Tests

        [TestMethod]
        public void ColumnVector_Multiply_Scalar_ReturnsCorrectProduct()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Multiply(3);

            // Assert
            Assert.AreEqual(3, result[0], Tolerance);
            Assert.AreEqual(6, result[1], Tolerance);
            Assert.AreEqual(9, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorMultiply_Scalar_ReturnsCorrectProduct()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            ColumnVectorBase vector = new ColumnVector(data);

            // Act
            var result = vector * 4;

            // Assert
            Assert.AreEqual(4, result[0], Tolerance);
            Assert.AreEqual(8, result[1], Tolerance);
            Assert.AreEqual(12, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OperatorMultiply_ScalarLeft_ReturnsCorrectProduct()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            ColumnVectorBase vector = new ColumnVector(data);

            // Act
            var result = 5 * vector;

            // Assert
            Assert.AreEqual(5, result[0], Tolerance);
            Assert.AreEqual(10, result[1], Tolerance);
            Assert.AreEqual(15, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_ZeroScalar_ReturnsZeroVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Multiply(0);

            // Assert
            Assert.AreEqual(0, result[0], Tolerance);
            Assert.AreEqual(0, result[1], Tolerance);
            Assert.AreEqual(0, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_OneScalar_ReturnsSameVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Multiply(1);

            // Assert
            Assert.AreEqual(1, result[0], Tolerance);
            Assert.AreEqual(2, result[1], Tolerance);
            Assert.AreEqual(3, result[2], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_NegativeScalar_ReturnsNegatedVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Multiply(-2);

            // Assert
            Assert.AreEqual(-2, result[0], Tolerance);
            Assert.AreEqual(-4, result[1], Tolerance);
            Assert.AreEqual(-6, result[2], Tolerance);
        }

        #endregion

        #region Immutability Tests

        [TestMethod]
        public void ColumnVector_Add_DoesNotModifyOriginal()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);
            var originalValue = vector[0];

            // Act
            var result = vector.Add(100);

            // Assert
            Assert.AreEqual(originalValue, vector[0], Tolerance);
            Assert.AreNotEqual(originalValue, result[0], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_DoesNotModifyOriginal()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);
            var originalValue = vector[0];

            // Act
            var result = vector.Multiply(100);

            // Assert
            Assert.AreEqual(originalValue, vector[0], Tolerance);
            Assert.AreNotEqual(originalValue, result[0], Tolerance);
        }

        #endregion
    }
}
