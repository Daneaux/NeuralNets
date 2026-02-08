using MatrixLibrary;

namespace NeuralNetsTests.Math
{
    [TestClass]
    public class MatrixTests2
    {
        [TestMethod]
        public void Matrix_Constructor_SetsDimensionsCorrectly()
        {
            int rows = 3;
            int cols = 4;
            Matrix2D matrix = new Matrix2D(rows, cols);

            Assert.AreEqual(rows, matrix.Rows);
            Assert.AreEqual(cols, matrix.Cols);
        }

        [TestMethod]
        public void Matrix_SetRandom_SetsValuesWithinRange()
        {
            int rows = 3;
            int cols = 3;
            Matrix2D matrix = new Matrix2D(rows, cols);
            int seed = 42;
            float min = 1;
            float max = 5;

            matrix.SetRandom(seed, min, max);

            foreach (float value in matrix.Mat)
            {
                Assert.IsTrue(value >= min && value <= max);
            }
        }

        [TestMethod]
        public void Matrix_Multiply_MultipliesMatricesCorrectly()
        {
            float[,] aArray = { { 1, 2 }, { 3, 4 } };
            float[,] bArray = { { 2, 0 }, { 1, 2 } };
            Matrix2D a = new Matrix2D(aArray);
            Matrix2D b = new Matrix2D(bArray);
            Matrix2D result = a.Multiply(b);

            float[,] expectedArray = { { 4, 4 }, { 10, 8 } };
            Matrix2D expected = new Matrix2D(expectedArray);

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], result[i, j]);
                }
            }
        }

        [TestMethod]
        public void Matrix_Multiply_DifferentDimensionsThrowsException()
        {
            float[,] aArray = { { 1, 2 }, { 3, 4 } };
            float[,] bArray = { { 2, 0, 1 }, { 1, 2, 3 }, { 1, 2, 3 } };
            Matrix2D a = new Matrix2D(aArray);
            Matrix2D b = new Matrix2D(bArray);

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => a.Multiply(b));
        }

        [TestMethod]
        public void Matrix_Subtract_SubtractsMatricesCorrectly()
        {
            float[,] aArray = { { 5, 6 }, { 7, 8 } };
            float[,] bArray = { { 1, 2 }, { 3, 4 } };
            Matrix2D a = new Matrix2D(aArray);
            Matrix2D b = new Matrix2D(bArray);
            Matrix2D result = a.Subtract(b);

            float[,] expectedArray = { { 4, 4 }, { 4, 4 } };
            Matrix2D expected = new Matrix2D(expectedArray);

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], result[i, j]);
                }
            }
        }

        [TestMethod]
        public void Matrix_HadamardProduct_ComputesCorrectly()
        {
            float[,] aArray = { { 1, 2 }, { 3, 4 } };
            float[,] bArray = { { 2, 0 }, { 1, 2 } };
            Matrix2D a = new Matrix2D(aArray);
            Matrix2D b = new Matrix2D(bArray);
            Matrix2D result = a.HadamardProduct(b);

            float[,] expectedArray = { { 2, 0 }, { 3, 8 } };
            Matrix2D expected = new Matrix2D(expectedArray);

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], result[i, j]);
                }
            }
        }

        [TestMethod]
        public void Matrix_GetTransposedMatrix_ComputesCorrectly()
        {
            float[,] originalArray = { { 1, 2, 3 }, { 4, 5, 6 } };
            Matrix2D original = new Matrix2D(originalArray);
            Matrix2D transposed = original.GetTransposedMatrix();

            float[,] expectedArray = { { 1, 4 }, { 2, 5 }, { 3, 6 } };
            Matrix2D expected = new Matrix2D(expectedArray);

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], transposed[i, j]);
                }
            }
        }
    }


}
