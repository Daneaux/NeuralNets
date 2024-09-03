using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNets;
using System;

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
            Matrix matrix = new Matrix(rows, cols);

            Assert.AreEqual(rows, matrix.Rows);
            Assert.AreEqual(cols, matrix.Cols);
        }

        [TestMethod]
        public void Matrix_ConstructorFromVector_SetsValuesCorrectly()
        {
            float[] vector = { 1, 2, 3 };
            Matrix matrix = new Matrix(vector);

            Assert.AreEqual(vector.Length, matrix.Rows);
            Assert.AreEqual(1, matrix.Cols);
            for (int i = 0; i < vector.Length; i++)
            {
                Assert.AreEqual(vector[i], matrix[i, 0]);
            }
        }

        [TestMethod]
        public void Matrix_SetRandom_SetsValuesWithinRange()
        {
            int rows = 3;
            int cols = 3;
            Matrix matrix = new Matrix(rows, cols);
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
            Matrix a = new Matrix(aArray);
            Matrix b = new Matrix(bArray);
            Matrix result = a * b;

            float[,] expectedArray = { { 4, 4 }, { 10, 8 } };
            Matrix expected = new Matrix(expectedArray);

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], result[i, j]);
                }
            }
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentOutOfRangeException))]
        public void Matrix_Multiply_DifferentDimensionsThrowsException()
        {
            float[,] aArray = { { 1, 2 }, { 3, 4 } };
            float[,] bArray = { { 2, 0, 1 }, { 1, 2, 3 }, { 1, 2, 3 } };
            Matrix a = new Matrix(aArray);
            Matrix b = new Matrix(bArray);

            Matrix result = a * b; // This should throw an exception
        }

        [TestMethod]
        public void Matrix_Subtract_SubtractsMatricesCorrectly()
        {
            float[,] aArray = { { 5, 6 }, { 7, 8 } };
            float[,] bArray = { { 1, 2 }, { 3, 4 } };
            Matrix a = new Matrix(aArray);
            Matrix b = new Matrix(bArray);
            Matrix result = a - b;

            float[,] expectedArray = { { 4, 4 }, { 4, 4 } };
            Matrix expected = new Matrix(expectedArray);

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
            Matrix a = new Matrix(aArray);
            Matrix b = new Matrix(bArray);
            Matrix result = a.HadamardProduct(b);

            float[,] expectedArray = { { 2, 0 }, { 3, 8 } };
            Matrix expected = new Matrix(expectedArray);

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
            Matrix original = new Matrix(originalArray);
            Matrix transposed = original.GetTransposedMatrix();

            float[,] expectedArray = { { 1, 4 }, { 2, 5 }, { 3, 6 } };
            Matrix expected = new Matrix(expectedArray);

            for (int i = 0; i < expected.Rows; i++)
            {
                for (int j = 0; j < expected.Cols; j++)
                {
                    Assert.AreEqual(expected[i, j], transposed[i, j]);
                }
            }
        }
    }

    [TestClass]
    public class RowVectorTests2
    {
        [TestMethod]
        public void RowVector_Constructor_SetsValuesCorrectly()
        {
            float[] vector = { 1, 2, 3 };
            RowVector rowVector = new RowVector(vector);

            Assert.AreEqual(1, rowVector.Rows);
            Assert.AreEqual(vector.Length, rowVector.Cols);
            for (int i = 0; i < vector.Length; i++)
            {
                Assert.AreEqual(vector[i], rowVector[0, i]);
            }
        }
    }

    [TestClass]
    public class ColumnVectorTests2
    {
        [TestMethod]
        public void ColumnVector_Constructor_SetsValuesCorrectly()
        {
            float[] vector = { 1, 2, 3 };
            ColumnVector columnVector = new ColumnVector(vector);

            Assert.AreEqual(vector.Length, columnVector.Rows);
            Assert.AreEqual(1, columnVector.Cols);
            for (int i = 0; i < vector.Length; i++)
            {
                Assert.AreEqual(vector[i], columnVector[i, 0]);
            }
        }
    }
}
