using MatrixLibrary;

namespace NeuralNetsTests.Math
{
    [TestClass]
    public class MatrixTests
    {
        [TestMethod]
        public void Constructor_ShouldInitializeCorrectRowsAndColumns()
        {
            var matrix = new Matrix2D(3, 2);
            Assert.AreEqual(3, matrix.Rows);
            Assert.AreEqual(2, matrix.Cols);
        }

        [TestMethod]
        public void Multiply_ShouldReturnCorrectMatrixProduct()
        {
            var matrixA = new Matrix2D(new float[,] { { 1, 2 }, { 3, 4 } });
            var matrixB = new Matrix2D(new float[,] { { 2, 0 }, { 1, 2 } });
            var result = matrixA.Multiply(matrixB);

            Assert.AreEqual(2, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(4, result[0, 0]);
            Assert.AreEqual(4, result[0, 1]);
            Assert.AreEqual(10, result[1, 0]);
            Assert.AreEqual(8, result[1, 1]);
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void Multiply_ShouldThrowExceptionForInvalidDimensions()
        {
            var matrixA = new Matrix2D(2, 3);
            var matrixB = new Matrix2D(2, 2);
            var result = matrixA.Multiply(matrixB);
        }

        [TestMethod]
        public void HadamardProduct_ShouldReturnCorrectElementwiseProduct()
        {
            var matrixA = new Matrix2D(new float[,] { { 1, 2 }, { 3, 4 } });
            var matrixB = new Matrix2D(new float[,] { { 2, 0 }, { 1, 2 } });
            var result = matrixA.HadamardProduct(matrixB);

            Assert.AreEqual(2, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(2, result[0, 0]);
            Assert.AreEqual(0, result[0, 1]);
            Assert.AreEqual(3, result[1, 0]);
            Assert.AreEqual(8, result[1, 1]);
        }

        [TestMethod]
        public void Subtract_ShouldReturnCorrectMatrixDifference()
        {
            var matrixA = new Matrix2D(new float[,] { { 5, 7 }, { 9, 11 } });
            var matrixB = new Matrix2D(new float[,] { { 2, 3 }, { 4, 5 } });
            var result = matrixA - matrixB;

            Assert.AreEqual(3, result[0, 0]);
            Assert.AreEqual(4, result[0, 1]);
            Assert.AreEqual(5, result[1, 0]);
            Assert.AreEqual(6, result[1, 1]);
        }

        [TestMethod]
        public void GetTransposedMatrix_ShouldReturnCorrectTranspose()
        {
            var matrix = new Matrix2D(new float[,] { { 1, 2, 3 }, { 4, 5, 6 } });
            var transposed = matrix.GetTransposedMatrix();

            Assert.AreEqual(3, transposed.Rows);
            Assert.AreEqual(2, transposed.Cols);
            Assert.AreEqual(1, transposed[0, 0]);
            Assert.AreEqual(4, transposed[0, 1]);
            Assert.AreEqual(2, transposed[1, 0]);
            Assert.AreEqual(5, transposed[1, 1]);
            Assert.AreEqual(3, transposed[2, 0]);
            Assert.AreEqual(6, transposed[2, 1]);
        }
    }

    [TestClass]
    public class ColumnVectorTests
    {
        [TestMethod]
        public void ScalarMultiply_ShouldReturnCorrectResult()
        {
            var vector = new ColumnVector(new float[] { 1, 2, 3 });
            var result = vector * 2;

            Assert.AreEqual(2.0, result[0]);
            Assert.AreEqual(4.0, result[1]);
            Assert.AreEqual(6.0, result[2]);
        }

        [TestMethod]
        public void ScalarAddition_ShouldReturnCorrectResult()
        {
            var vector = new ColumnVector(new float[] { 1, 2, 3 });
            var result = vector + 2;

            Assert.AreEqual(3.0, result[0]);
            Assert.AreEqual(4.0, result[1]);
            Assert.AreEqual(5.0, result[2]);
        }

        [TestMethod]
        public void Subtract_ShouldReturnCorrectColumnVectorDifference()
        {
            var vectorA = new ColumnVector(new float[] { 5, 7, 9 });
            var vectorB = new ColumnVector(new float[] { 2, 3, 4 });
            var result = vectorA - vectorB;

            Assert.AreEqual(3.0, result[0]);
            Assert.AreEqual(4.0, result[1]);
            Assert.AreEqual(5.0, result[2]);
        }
    }

    [TestClass]
    public class RowVectorTests
    {
        [TestMethod]
        public void Constructor_ShouldInitializeCorrectRowVector()
        {
            float[] vector = { 1, 2, 3 };
            var rowVector = new RowVector(vector);

            Assert.AreEqual(1, rowVector.Rows);
            Assert.AreEqual(3, rowVector.Cols);
            Assert.AreEqual(1.0, rowVector[0]);
            Assert.AreEqual(2.0, rowVector[1]);
            Assert.AreEqual(3.0, rowVector[2]);
        }

        [TestMethod]
        public void RowVectorTimesMatrix_ShouldReturnCorrectResult()
        {
            var rowVector = new RowVector(new float[] { 1, 2 });
            var matrix = new Matrix2D(new float[,] { { 3, 4 }, { 5, 6 } });
            var result = rowVector * matrix;

            Assert.AreEqual(1, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(13, result[0]);  // 1*3 + 2*5
            Assert.AreEqual(16, result[1]);  // 1*4 + 2*6
        }
    }
}

