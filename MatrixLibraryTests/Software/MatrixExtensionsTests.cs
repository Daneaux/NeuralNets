using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests
{
    /// <summary>
    /// Comprehensive tests for MatrixExtensions (Xavier initialization)
    /// Phase 5: Extension Methods
    /// </summary>
    [TestClass]
    public class MatrixExtensionsTests
    {
        private const float Tolerance = 1e-4f;

        #region XavierInitialize Matrix Tests

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_Matrix_CreatesValuesInRange()
        {
            // Arrange
            var matrix = new Matrix2D(10, 10);
            int fanIn = 10;
            int fanOut = 10;

            // Act
            matrix.XavierInitialize(fanIn, fanOut, seed: 42);

            // Assert - Xavier range: ±√(6/(fanIn+fanOut)) = ±√(6/20) ≈ ±0.5477
            float expectedBound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    Assert.IsTrue(matrix[i, j] >= -expectedBound && matrix[i, j] <= expectedBound,
                        $"Value at [{i},{j}] ({matrix[i, j]}) is outside range [±{expectedBound}]");
                }
            }
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_Matrix_SameSeed_SameValues()
        {
            // Arrange
            var matrix1 = new Matrix2D(5, 5);
            var matrix2 = new Matrix2D(5, 5);
            int fanIn = 5;
            int fanOut = 5;

            // Act
            matrix1.XavierInitialize(fanIn, fanOut, seed: 123);
            matrix2.XavierInitialize(fanIn, fanOut, seed: 123);

            // Assert
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    Assert.AreEqual(matrix1[i, j], matrix2[i, j], Tolerance);
                }
            }
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_Matrix_DifferentSeeds_DifferentValues()
        {
            // Arrange
            var matrix1 = new Matrix2D(5, 5);
            var matrix2 = new Matrix2D(5, 5);
            int fanIn = 5;
            int fanOut = 5;

            // Act
            matrix1.XavierInitialize(fanIn, fanOut, seed: 1);
            matrix2.XavierInitialize(fanIn, fanOut, seed: 2);

            // Assert
            bool anyDifferent = false;
            for (int i = 0; i < 5; i++)
            {
                for (int j = 0; j < 5; j++)
                {
                    if (Math.Abs(matrix1[i, j] - matrix2[i, j]) > Tolerance)
                    {
                        anyDifferent = true;
                        break;
                    }
                }
                if (anyDifferent) break;
            }
            Assert.IsTrue(anyDifferent, "Matrices with different seeds should have different values");
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_Matrix_LargeFanInFanOut_SmallValues()
        {
            // Arrange - typical CNN layer: 512 inputs, 512 outputs
            var matrix = new Matrix2D(100, 100);
            int fanIn = 512;
            int fanOut = 512;

            // Act
            matrix.XavierInitialize(fanIn, fanOut, seed: 42);

            // Assert - bound: ±√(6/1024) ≈ ±0.0765
            float expectedBound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            Assert.IsTrue(expectedBound < 0.1, "Bound should be small for large fan_in/fan_out");

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    Assert.IsTrue(Math.Abs(matrix[i, j]) <= expectedBound,
                        $"Value at [{i},{j}] ({matrix[i, j]}) exceeds bound ±{expectedBound}");
                }
            }
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_Matrix_SmallFanInFanOut_LargeValues()
        {
            // Arrange - small layer: 2 inputs, 2 outputs
            var matrix = new Matrix2D(2, 2);
            int fanIn = 2;
            int fanOut = 2;

            // Act
            matrix.XavierInitialize(fanIn, fanOut, seed: 42);

            // Assert - bound: ±√(6/4) ≈ ±1.2247
            float expectedBound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            Assert.IsTrue(expectedBound > 1.0, "Bound should be large for small fan_in/fan_out");

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    Assert.IsTrue(Math.Abs(matrix[i, j]) <= expectedBound,
                        $"Value at [{i},{j}] ({matrix[i, j]}) exceeds bound ±{expectedBound}");
                }
            }
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_Matrix_ZeroFanIn_HandlesGracefully()
        {
            // Arrange
            var matrix = new Matrix2D(3, 3);
            int fanIn = 0;
            int fanOut = 10;

            // Act - should not throw
            matrix.XavierInitialize(fanIn, fanOut, seed: 42);

            // Assert - values should be finite
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Cols; j++)
                {
                    Assert.IsFalse(float.IsNaN(matrix[i, j]), "Value should not be NaN");
                    Assert.IsFalse(float.IsInfinity(matrix[i, j]), "Value should not be Infinity");
                }
            }
        }

        #endregion

        #region XavierInitialize ColumnVector Tests

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_ColumnVector_CreatesValuesInRange()
        {
            // Arrange
            var vector = new ColumnVector(10);
            int fanIn = 10;
            int fanOut = 1;  // Typical for bias vector

            // Act
            vector.XavierInitialize(fanIn, fanOut, seed: 42);

            // Assert
            float expectedBound = (float)Math.Sqrt(6.0 / (fanIn + fanOut));
            for (int i = 0; i < vector.Size; i++)
            {
                Assert.IsTrue(vector[i] >= -expectedBound && vector[i] <= expectedBound,
                    $"Value at index {i} ({vector[i]}) is outside range [±{expectedBound}]");
            }
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_ColumnVector_SameSeed_SameValues()
        {
            // Arrange
            var vector1 = new ColumnVector(10);
            var vector2 = new ColumnVector(10);
            int fanIn = 10;
            int fanOut = 10;

            // Act
            vector1.XavierInitialize(fanIn, fanOut, seed: 123);
            vector2.XavierInitialize(fanIn, fanOut, seed: 123);

            // Assert
            for (int i = 0; i < 10; i++)
            {
                Assert.AreEqual(vector1[i], vector2[i], Tolerance);
            }
        }

        [TestMethod]
        public void MatrixExtensions_XavierInitialize_ColumnVector_CanBePositiveOrNegative()
        {
            // Arrange
            var vector = new ColumnVector(100);  // Large enough to likely have both signs
            int fanIn = 10;
            int fanOut = 10;

            // Act
            vector.XavierInitialize(fanIn, fanOut, seed: 42);

            // Assert
            bool hasPositive = false;
            bool hasNegative = false;
            for (int i = 0; i < vector.Size; i++)
            {
                if (vector[i] > 0) hasPositive = true;
                if (vector[i] < 0) hasNegative = true;
            }
            Assert.IsTrue(hasPositive || hasNegative, "Vector should have non-zero values");
        }

        #endregion
    }
}
