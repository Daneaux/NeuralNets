using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests.Avx
{
    /// <summary>
    /// Comprehensive tests for FlattenedMatricesAsVector
    /// Tests AVX-optimized matrix operations on flattened matrices
    /// </summary>
    [TestClass]
    public class FlattenedMatricesAsVectorTests
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

        private float[] ComputeExpectedMatrixTimesColumn(float[,] matrixData, List<MatrixBase> flattenedMatrices)
        {
            // Compute expected result manually
            int outputSize = matrixData.GetLength(0); // LHS rows = output size
            int matRows = flattenedMatrices[0].Rows;
            int matCols = flattenedMatrices[0].Cols;
            int matSize = matRows * matCols;
            
            float[] expected = new float[outputSize * flattenedMatrices.Count];
            
            for (int m = 0; m < flattenedMatrices.Count; m++)
            {
                float[] matData = new float[matSize];
                int idx = 0;
                for (int i = 0; i < matRows; i++)
                    for (int j = 0; j < matCols; j++)
                        matData[idx++] = flattenedMatrices[m][i, j];
                
                for (int row = 0; row < outputSize; row++)
                {
                    float sum = 0;
                    for (int col = 0; col < matSize; col++)
                        sum += matrixData[row, col % matrixData.GetLength(1)] * matData[col];
                    expected[m * outputSize + row] = sum;
                }
            }
            
            return expected;
        }

        #endregion

        #region Constructor and Property Tests

        [TestMethod]
        public void FlattenedMatricesAsVector_Constructor_SetsProperties()
        {
            // Arrange
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[,] { { 1, 2 }, { 3, 4 } }),
                new AvxMatrix(new float[,] { { 5, 6 }, { 7, 8 } })
            };

            // Act
            var flattened = new FlattenedMatricesAsVector(matrices);

            // Assert
            Assert.AreEqual(2, flattened.Matrices.Count);
            Assert.AreEqual(8, flattened.Size); // 2 matrices * 2*2 elements = 8
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_Constructor_EmptyList_ZeroSize()
        {
            // Arrange
            var matrices = new List<MatrixBase>();

            // Act
            var flattened = new FlattenedMatricesAsVector(matrices);

            // Assert
            Assert.AreEqual(0, flattened.Matrices.Count);
            Assert.AreEqual(0, flattened.Size);
        }

        #endregion

        #region MatrixTimesColumn Tests

        [TestMethod]
        public void FlattenedMatricesAsVector_MatrixTimesColumn_SingleMatrix()
        {
            // Arrange - simple case with 1 matrix of 2x2 = 4 elements
            // LHS must have 4 columns to match flattened size
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[,] { { 1, 2 }, { 3, 4 } })
            };
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 2 rows x 4 cols (2 outputs, 4 inputs to match 1 matrix * 4 elements)
            float[,] lhsData = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 } };
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 2 outputs (lhs.Rows)
            Assert.AreEqual(2, result.Size);
            // First row of LHS dot [1,2,3,4] = 1*1 + 0*2 + 0*3 + 0*4 = 1
            Assert.AreEqual(1, result[0], Tolerance);
            // Second row of LHS dot [1,2,3,4] = 0*1 + 1*2 + 0*3 + 0*4 = 2
            Assert.AreEqual(2, result[1], Tolerance);
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_MatrixTimesColumn_MultipleMatrices()
        {
            // Arrange - 3 matrices, each 2x2 = 4 elements
            // Total flattened size = 3 * 4 = 12
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[,] { { 1, 2 }, { 3, 4 } }),
                new AvxMatrix(new float[,] { { 5, 6 }, { 7, 8 } }),
                new AvxMatrix(new float[,] { { 9, 10 }, { 11, 12 } })
            };
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 2 rows x 12 cols (2 outputs, 12 inputs to match 3 matrices * 4 elements)
            float[,] lhsData = new float[2, 12];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 12; j++)
                    lhsData[i, j] = (i + 1) * (j + 1) * 0.1f;
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 2 outputs (lhs.Rows)
            Assert.AreEqual(2, result.Size);
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_MatrixTimesColumn_LargeMatrices()
        {
            // Arrange - 4 matrices of 8x8 each (64 elements each, 256 total)
            var random = new Random(42);
            var matrices = new List<MatrixBase>();
            for (int m = 0; m < 4; m++)
            {
                float[,] data = new float[8, 8];
                for (int i = 0; i < 8; i++)
                    for (int j = 0; j < 8; j++)
                        data[i, j] = (float)random.NextDouble() * 5;
                matrices.Add(new AvxMatrix(data));
            }
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 16x256 (16 outputs, 256 inputs to match 4 matrices * 64 elements)
            float[,] lhsData = new float[16, 256];
            for (int i = 0; i < 16; i++)
                for (int j = 0; j < 256; j++)
                    lhsData[i, j] = (float)random.NextDouble() * 0.1f;
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 16 outputs (lhs.Rows)
            Assert.AreEqual(16, result.Size);
            
            // Verify no NaN or Infinity
            for (int i = 0; i < result.Size; i++)
            {
                Assert.IsFalse(float.IsNaN(result[i]), $"NaN at index {i}");
                Assert.IsFalse(float.IsInfinity(result[i]), $"Infinity at index {i}");
            }
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_OperatorMultiply_MatchesMethod()
        {
            // Arrange - 1 matrix of 2x2 = 4 elements
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[,] { { 1, 2 }, { 3, 4 } })
            };
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 2x4 to match flattened size
            float[,] lhsData = { { 1, 0, 0, 0 }, { 0, 1, 0, 0 } };
            var lhs = new AvxMatrix(lhsData);

            // Act
            var methodResult = flattened.MatrixTimesColumn(lhs);
            var operatorResult = lhs * flattened;

            // Assert
            AssertVectorsEqual(methodResult, (AvxColumnVector)operatorResult, "Operator should match method");
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_MatrixTimesColumn_CompatibleDimensions()
        {
            // Arrange - test that dimensions are properly validated
            // 2 matrices of 4x4 = 16 elements each, total = 32 elements
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[4, 4]),
                new AvxMatrix(new float[4, 4])
            };
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS must have 32 columns to match total flattened size
            float[,] lhsData = new float[8, 32]; // 8 outputs, 32 inputs
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 8 outputs (lhs.Rows)
            Assert.AreEqual(8, result.Size);
        }

        #endregion

        #region Edge Cases

        [TestMethod]
        public void FlattenedMatricesAsVector_MatrixTimesColumn_SingleElementMatrices()
        {
            // Arrange - 3 matrices of 1x1 = 1 element each, total = 3 elements
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[,] { { 5 } }),
                new AvxMatrix(new float[,] { { 10 } }),
                new AvxMatrix(new float[,] { { 15 } })
            };
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 1x3 to match total flattened size of 3
            float[,] lhsData = { { 2, 3, 4 } };
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 1 output (lhs.Rows)
            Assert.AreEqual(1, result.Size);
            // 5*2 + 10*3 + 15*4 = 10 + 30 + 60 = 100
            Assert.AreEqual(100, result[0], Tolerance);
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_MatrixTimesColumn_ZeroMatrices()
        {
            // Arrange - 2 matrices of 2x2 = 4 elements each, total = 8 elements
            var matrices = new List<MatrixBase>
            {
                new AvxMatrix(new float[,] { { 0, 0 }, { 0, 0 } }),
                new AvxMatrix(new float[,] { { 0, 0 }, { 0, 0 } })
            };
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 2x8 to match total flattened size of 8
            float[,] lhsData = new float[2, 8];
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 8; j++)
                    lhsData[i, j] = (i * 8 + j + 1);
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 2 outputs (lhs.Rows), all zeros
            Assert.AreEqual(2, result.Size);
            for (int i = 0; i < result.Size; i++)
                Assert.AreEqual(0, result[i], Tolerance);
        }

        [TestMethod]
        public void FlattenedMatricesAsVector_LargeNumberOfSmallMatrices()
        {
            // Arrange - 100 matrices of 2x2 = 4 elements each, total = 400 elements
            var random = new Random(42);
            var matrices = new List<MatrixBase>();
            for (int m = 0; m < 100; m++)
            {
                float[,] data = new float[2, 2];
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                        data[i, j] = (float)random.NextDouble() * 2 - 1; // -1 to 1
                matrices.Add(new AvxMatrix(data));
            }
            var flattened = new FlattenedMatricesAsVector(matrices);
            
            // LHS: 10x400 (10 outputs, 400 inputs to match 100 matrices * 4 elements)
            float[,] lhsData = new float[10, 400];
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 400; j++)
                    lhsData[i, j] = (float)random.NextDouble() * 0.5f;
            var lhs = new AvxMatrix(lhsData);

            // Act
            var result = flattened.MatrixTimesColumn(lhs);

            // Assert - 10 outputs (lhs.Rows)
            Assert.AreEqual(10, result.Size);
            
            // Verify all values are finite
            for (int i = 0; i < result.Size; i++)
            {
                Assert.IsFalse(float.IsNaN(result[i]), $"NaN at index {i}");
                Assert.IsFalse(float.IsInfinity(result[i]), $"Infinity at index {i}");
            }
        }

        #endregion
    }
}
