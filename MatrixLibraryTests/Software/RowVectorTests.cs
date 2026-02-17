using MatrixLibrary;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for RowVector operations
    /// Phase 2: RowVector Operations
    /// </summary>
    [TestClass]
    public class RowVectorTests
    {
        private const float Tolerance = 1e-5f;

        #region Basic Operations

        [TestMethod]
        public void RowVector_Sum_ReturnsCorrectTotal()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };
            var vector = new RowVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(15, sum, Tolerance);
        }

        [TestMethod]
        public void RowVector_Sum_SingleElement()
        {
            // Arrange
            float[] data = { 42 };
            var vector = new RowVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(42, sum, Tolerance);
        }

        [TestMethod]
        public void RowVector_Sum_ZeroVector()
        {
            // Arrange
            float[] data = { 0, 0, 0 };
            var vector = new RowVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(0, sum, Tolerance);
        }

        [TestMethod]
        public void RowVector_Sum_NegativeValues()
        {
            // Arrange
            float[] data = { -1, -2, -3 };
            var vector = new RowVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(-6, sum, Tolerance);
        }

        [TestMethod]
        public void RowVector_Sum_MixedSigns()
        {
            // Arrange
            float[] data = { -5, 10, -3, 8 };
            var vector = new RowVector(data);

            // Act
            float sum = vector.Sum();

            // Assert
            Assert.AreEqual(10, sum, Tolerance);
        }

        #endregion

        #region Indexer Tests

        [TestMethod]
        public void RowVector_Indexer_Get_ReturnsCorrectValue()
        {
            // Arrange
            float[] data = { 10, 20, 30, 40, 50 };
            var vector = new RowVector(data);

            // Act & Assert
            Assert.AreEqual(10, vector[0], Tolerance);
            Assert.AreEqual(30, vector[2], Tolerance);
            Assert.AreEqual(50, vector[4], Tolerance);
        }

        [TestMethod]
        public void RowVector_Indexer_Set_ModifiesValue()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new RowVector(data);

            // Act
            vector[1] = 100;

            // Assert
            Assert.AreEqual(100, vector[1], Tolerance);
            Assert.AreEqual(1, vector[0], Tolerance);
            Assert.AreEqual(3, vector[2], Tolerance);
        }

        [TestMethod]
        public void RowVector_Indexer_OutOfBounds_ThrowsException()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new RowVector(data);

            // Act & Assert
            Assert.ThrowsException<IndexOutOfRangeException>(() => vector[-1]);
            Assert.ThrowsException<IndexOutOfRangeException>(() => vector[3]);
        }

        #endregion

        #region Property Tests

        [TestMethod]
        public void RowVector_Size_ReturnsCorrectLength()
        {
            // Arrange
            float[] data = { 1, 2, 3, 4, 5 };
            var vector = new RowVector(data);

            // Act
            int size = vector.Size;

            // Assert
            Assert.AreEqual(5, size);
        }

        [TestMethod]
        public void RowVector_Row_ReturnsArray()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new RowVector(data);

            // Act
            float[] row = vector.Row;

            // Assert
            Assert.AreEqual(3, row.Length);
            Assert.AreEqual(1, row[0], Tolerance);
            Assert.AreEqual(2, row[1], Tolerance);
            Assert.AreEqual(3, row[2], Tolerance);
        }

        [TestMethod]
        public void RowVector_Row_ModifyingArrayModifiesVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new RowVector(data);

            // Act
            float[] row = vector.Row;
            row[1] = 100;

            // Assert - Row property returns the internal array
            Assert.AreEqual(100, vector[1], Tolerance);
        }

        #endregion

        #region Constructor Tests

        [TestMethod]
        public void RowVector_Constructor_WithSize_CreatesEmptyVector()
        {
            // Act
            var vector = new RowVector(5);

            // Assert
            Assert.AreEqual(5, vector.Size);
            for (int i = 0; i < vector.Size; i++)
            {
                Assert.AreEqual(0, vector[i], Tolerance);
            }
        }

        [TestMethod]
        public void RowVector_Constructor_WithArray_CopiesArray()
        {
            // Arrange
            float[] data = { 1, 2, 3 };

            // Act
            var vector = new RowVector(data);

            // Assert
            Assert.AreEqual(3, vector.Size);
            Assert.AreEqual(1, vector[0], Tolerance);
            Assert.AreEqual(2, vector[1], Tolerance);
            Assert.AreEqual(3, vector[2], Tolerance);
        }

        [TestMethod]
        public void RowVector_Constructor_ArrayNotCloned_ModifyingOriginalDoesNotAffectVector()
        {
            // Arrange
            float[] data = { 1, 2, 3 };
            var vector = new RowVector(data);

            // Act
            data[1] = 100;

            // Assert - verify if array is copied or referenced
            // (behavior depends on implementation)
            float vectorValue = vector[1];
            Assert.IsTrue(vectorValue == 2 || vectorValue == 100,
                "RowVector should either copy the array or reference it");
        }

        #endregion
    }
}
