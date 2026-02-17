using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for ColumnVector special operations (Outer Product, Sum, Log, Max)
    /// Phase 2: ColumnVector Special Operations
    /// </summary>
    [TestClass]
    public class ColumnVectorSpecialOperationsTests
    {
        private const float Tolerance = 1e-5f;

        #region Outer Product Tests

        [TestMethod]
        public void ColumnVector_OuterProduct_ReturnsCorrectMatrix()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] dataB = { 4, 5 };
            var vecA = new ColumnVector(dataA);
            var vecB = new ColumnVector(dataB);

            // Act
            var result = vecA.OuterProduct(vecB);

            // Assert
            // [1]   [4 5]   [1*4  1*5]   [4  5 ]
            // [2] *       = [2*4  2*5] = [8  10]
            // [3]         [3*4  3*5]   [12 15]
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(4, result[0, 0], Tolerance);
            Assert.AreEqual(5, result[0, 1], Tolerance);
            Assert.AreEqual(8, result[1, 0], Tolerance);
            Assert.AreEqual(10, result[1, 1], Tolerance);
            Assert.AreEqual(12, result[2, 0], Tolerance);
            Assert.AreEqual(15, result[2, 1], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OuterProduct_WithSelf_ReturnsSymmetricValues()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.OuterProduct(vector);

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(3, result.Cols);
            Assert.AreEqual(1, result[0, 0], Tolerance);  // 1*1
            Assert.AreEqual(2, result[0, 1], Tolerance);  // 1*2
            Assert.AreEqual(3, result[0, 2], Tolerance);  // 1*3
            Assert.AreEqual(2, result[1, 0], Tolerance);  // 2*1
            Assert.AreEqual(4, result[1, 1], Tolerance);  // 2*2
            Assert.AreEqual(6, result[1, 2], Tolerance);  // 2*3
            Assert.AreEqual(3, result[2, 0], Tolerance);  // 3*1
            Assert.AreEqual(6, result[2, 1], Tolerance);  // 3*2
            Assert.AreEqual(9, result[2, 2], Tolerance);  // 3*3
        }

        [TestMethod]
        public void ColumnVector_OuterProduct_ZeroVector_ReturnsZeroMatrix()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] zeroData = { 0, 0, 0 };
            var vecA = new ColumnVector(dataA);
            var zeroVec = new ColumnVector(zeroData);

            // Act
            var result = vecA.OuterProduct(zeroVec);

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(3, result.Cols);
            for (int i = 0; i < result.Rows; i++)
                for (int j = 0; j < result.Cols; j++)
                    Assert.AreEqual(0, result[i, j], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_OuterProduct_UnitVector_ReturnsIdentityColumns()
        {
            // Arrange
            float[] dataA = { 1, 2, 3 };
            float[] unitData = { 1 };  // 1-element vector
            var vecA = new ColumnVector(dataA);
            var unitVec = new ColumnVector(unitData);

            // Act
            var result = vecA.OuterProduct(unitVec);

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(1, result.Cols);
            Assert.AreEqual(1, result[0, 0], Tolerance);
            Assert.AreEqual(2, result[1, 0], Tolerance);
            Assert.AreEqual(3, result[2, 0], Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Multiply_RowVector_ReturnsOuterProduct()
        {
            // Arrange
            float[] columnData = { 1, 2, 3 };
            float[] rowData = { 4, 5, 6, 7 };
            var colVector = new ColumnVector(columnData);
            var rowVector = new RowVector(rowData);

            // Act
            var result = colVector.Multiply(rowVector);

            // Assert
            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(4, result.Cols);
            // [1]           [4 5 6 7]   [4  5  6  7 ]
            // [2] * [4 5 6 7] = [8  10 12 14]
            // [3]           [12 15 18 21]
            Assert.AreEqual(4, result[0, 0], Tolerance);
            Assert.AreEqual(5, result[0, 1], Tolerance);
            Assert.AreEqual(8, result[1, 0], Tolerance);
            Assert.AreEqual(21, result[2, 3], Tolerance);
        }

        #endregion

        #region Sum Tests

        [TestMethod]
        public void ColumnVector_Sum_ReturnsCorrectTotal()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };
            var vector = new ColumnVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(15, sum, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Sum_SingleElement()
        {
            // Arrange
            float[] data = { 42 };
            var vector = new ColumnVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(42, sum, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Sum_ZeroVector_ReturnsZero()
        {
            // Arrange
            float[] data = { 0, 0, 0 };
            var vector = new ColumnVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(0, sum, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Sum_NegativeValues()
        {
            // Arrange
            float[] data = { -1, -2, -3 };
            var vector = new ColumnVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(-6, sum, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Sum_MixedSigns()
        {
            // Arrange
            float[] data = { -5, 10, -3, 8 };
            var vector = new ColumnVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(10, sum, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_Sum_Floats()
        {
            // Arrange
            float[] data = { 1.5f, 2.5f, 3.0f };
            var vector = new ColumnVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(7.0f, sum, Tolerance);
        }

        #endregion

        #region Log Tests

        [TestMethod]
        public void ColumnVector_Log_ReturnsNaturalLog()
        {
            // Arrange
            float[] data = { 1, (float)Math.E, (float)Math.E * (float)Math.E };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Log();

            // Assert
            Assert.AreEqual(0, result[0], Tolerance);  // ln(1) = 0
            Assert.AreEqual(1, result[1], Tolerance);  // ln(e) = 1
            Assert.AreEqual(2, result[2], Tolerance);  // ln(e²) = 2
        }

        [TestMethod]
        public void ColumnVector_Log_SingleElement()
        {
            // Arrange
            float[] data = { (float)Math.E };
            var vector = new ColumnVector(data);

            // Act
            var result = vector.Log();

            // Assert
            Assert.AreEqual(1, result[0], Tolerance);
        }

        #endregion

        #region GetMax Tests

        [TestMethod]
        public void ColumnVector_GetMax_ReturnsMaximumValue()
        {
            // Arrange
            float[] data = { 3, 1, 4, 1, 5, 9, 2, 6 };
            var vector = new ColumnVector(data);

            // Act
            float max = vector.GetMax();

            // Assert
            Assert.AreEqual(9, max, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_GetMax_SingleElement()
        {
            // Arrange
            float[] data = { 42 };
            var vector = new ColumnVector(data);

            // Act
            float max = vector.GetMax();

            // Assert
            Assert.AreEqual(42, max, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_GetMax_NegativeValues()
        {
            // Arrange
            float[] data = { -5, -2, -10, -1 };
            var vector = new ColumnVector(data);

            // Act
            float max = vector.GetMax();

            // Assert
            Assert.AreEqual(-1, max, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_GetMax_MixedSigns()
        {
            // Arrange
            float[] data = { -5, 3, -1, 0, 2 };
            var vector = new ColumnVector(data);

            // Act
            float max = vector.GetMax();

            // Assert
            Assert.AreEqual(3, max, Tolerance);
        }

        [TestMethod]
        public void ColumnVector_GetMax_DuplicateMaximum()
        {
            // Arrange
            float[] data = { 5, 3, 5, 2, 5 };
            var vector = new ColumnVector(data);

            // Act
            float max = vector.GetMax();

            // Assert
            Assert.AreEqual(5, max, Tolerance);
        }

        #endregion

        #region Initialization Tests

        [TestMethod]
        public void ColumnVector_SetRandom_CreatesValuesInRange()
        {
            // Arrange
            var vector = new ColumnVector(100);

            // Act
            vector.SetRandom(seed: 42, min: -5, max: 5);

            // Assert
            for (int i = 0; i < vector.Size; i++)
            {
                Assert.IsTrue(vector[i] >= -5 && vector[i] <= 5,
                    $"Value at index {i} ({vector[i]}) is outside range [-5, 5]");
            }
        }

        [TestMethod]
        public void ColumnVector_SetRandom_SameSeed_SameValues()
        {
            // Arrange
            var vector1 = new ColumnVector(10);
            var vector2 = new ColumnVector(10);

            // Act
            vector1.SetRandom(seed: 123, min: 0, max: 100);
            vector2.SetRandom(seed: 123, min: 0, max: 100);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(vector1[i], vector2[i], Tolerance);
            }
        }

        [TestMethod]
        public void ColumnVector_SetRandom_DifferentSeeds_DifferentValues()
        {
            // Arrange
            var vector1 = new ColumnVector(10);
            var vector2 = new ColumnVector(10);

            // Act
            vector1.SetRandom(seed: 1, min: 0, max: 100);
            vector2.SetRandom(seed: 2, min: 0, max: 100);

            // Assert - very likely to be different with different seeds
            bool anyDifferent = false;
            for (int i = 0; i < 10; i++)
            {
                if (Math.Abs(vector1[i] - vector2[i]) > Tolerance)
                {
                    anyDifferent = true;
                    break;
                }
            }
            Assert.IsTrue(anyDifferent, "Vectors with different seeds should have different values");
        }

        [TestMethod]
        public void ColumnVector_Size_ReturnsCorrectLength()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };
            var vector = new ColumnVector(data);

            // Act
            int size = vector.Size;

            // Assert
            Assert.AreEqual(5, size);
        }

        [TestMethod]
        public void ColumnVector_Column_ReturnsCorrectArray()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new ColumnVector(data);

            // Act
            float[] column = vector.Column;

            // Assert
            Assert.AreEqual(3, column.Length);
            Assert.AreEqual(1, column[0], Tolerance);
            Assert.AreEqual(2, column[1], Tolerance);
            Assert.AreEqual(3, column[2], Tolerance);
        }

        #endregion
    }
}
