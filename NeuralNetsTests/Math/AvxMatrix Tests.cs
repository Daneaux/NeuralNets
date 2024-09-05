using BenchmarkDotNet.Attributes;
using IntrinsicMatrix;
using NeuralNets;

namespace NeuralNetsTests.Math
{
    [TestClass]
    public class AvxMatrixTests
    {
        [TestMethod]
        public void TestAdd()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 3;
            int cols = 17;
            float[,] matrixA = new float[rows, cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = (float)r * (float)c;
                }
            }

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = (float)r * (float)c + (float)c;
                }
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(matrixB);

            AvxMatrix m3 = m1.AddMatrix(m2);

            Assert.AreEqual(m1.Rows, m2.Rows);
            Assert.AreEqual(m2.Rows, m3.Rows);
            Assert.AreEqual(m1.Cols, m2.Cols);
            Assert.AreEqual(m2.Cols, m3.Cols);

            for (int r = 0; r < m3.Rows; r++)
            {
                for (int c = 0; c < m3.Cols; c++)
                {
                    float expected = m1[r, c] + m2[r, c];
                    Assert.AreEqual(expected, m3[r,c]);
                }
            }
        }

        [TestMethod]
        public void TestTimesColumn()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 3;
            int cols = 18;
            float[,] matrixA = new float[rows, cols];
            float[] columnA = new float[cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = (float)r * (float)c + (float)(c+r);
                }
            }

            for (int c = 0; c < cols; c++)
            {
                columnA[c] = (float)12 * (float)c + (float)c;
            }            

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxColumnVector m2 = new AvxColumnVector(columnA);

            AvxColumnVector m3 = m1.MatrixTimesColumn(m2);

            Assert.AreEqual(m1.Cols, m2.Size);
            Assert.AreEqual(m1.Rows, m3.Size);

            for (int r = 0; r < rows; r++)
            {
                // do the dot product row 'r' dot the column vector
                float dot = 0;
                for(int c=0; c < cols; c++)
                {
                    dot += matrixA[r, c] * columnA[c];
                }
                Assert.AreEqual(dot, m3[r]);
            }            
        }

        [TestMethod]
        public void TestMatrixMultiplySquare()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 57;
            int cols = 57;
            float[,] matrixA = new float[rows, cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = r + c;
                }
            }

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = r + c;
                }
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(matrixB);
            AvxMatrix m3 = m1 * m2;

            Matrix mm1 = new Matrix(matrixA);
            Matrix mm2 = new Matrix(matrixB);
            Matrix truth = mm1 * mm2;

            Assert.AreEqual(m1.Rows, m2.Rows);
            Assert.AreEqual(m2.Rows, m3.Rows);
            Assert.AreEqual(m1.Cols, m2.Cols);
            Assert.AreEqual(m2.Cols, m3.Cols);

            for (int r = 0; r < m3.Rows; r++)
            {
                for (int c = 0; c < m3.Cols; c++)
                {
                    Assert.AreEqual(truth[r, c], m3[r, c]);
                }
            }
        }

        [TestMethod]
        public void TestMatrixMultiply_Small_rectangular()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 3;
            int cols = 2;
            float[,] matrixA = new float[rows, cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = r + c;
                }
            }

            rows = 2;
            cols = 5;

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = r + c;
                }
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(matrixB);
            AvxMatrix m3 = m1 * m2;

            Matrix mm1 = new Matrix(matrixA);
            Matrix mm2 = new Matrix(matrixB);
            Matrix truth = mm1 * mm2;

            // Assert.AreEqual(m1.Cols, m3.Rows);

            for (int r = 0; r < m3.Rows; r++)
            {
                for (int c = 0; c < m3.Cols; c++)
                {
                    Assert.AreEqual(truth[r, c], m3[r, c]);
                }
            }

        }

        [TestMethod]
        public void TestMatrixMultiply_Large()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 21;
            int cols = 17;
            float[,] matrixA = new float[rows, cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = r + c;
                }
            }


            rows = 17;
            cols = 27;

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = r + c;
                }
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(matrixB);
            AvxMatrix m3 = m1 * m2;

            Matrix mm1 = new Matrix(matrixA);
            Matrix mm2 = new Matrix(matrixB);
            Matrix truth = mm1 * mm2;

           // Assert.AreEqual(m1.Cols, m3.Rows);

            for (int r = 0; r < m3.Rows; r++)
            {
                for (int c = 0; c < m3.Cols; c++)
                {
                    Assert.AreEqual(truth[r, c], m3[r, c]);
                }
            }

        }


        [TestMethod]
        public void TestMatrixMultiply_Transpose_Large()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 3;
            int cols = 3;
            float[,] matrixA = new float[rows, cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = r + c *2;
                }
            }

            rows = 3;
            cols = 3;

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = r + c * 3;
                }
            }

            Matrix B = new Matrix(matrixB);
            B = B.GetTransposedMatrix();

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(B.Mat);  // transposed version of B
            AvxMatrix m3 = m1.MatrixTimesMatrix_TransposedRHS(m2);

            Matrix mm1 = new Matrix(matrixA);
            Matrix mm2 = new Matrix(matrixB);
            Matrix truth = mm1 * mm2;

            // Assert.AreEqual(m1.Cols, m3.Rows);

            for (int r = 0; r < m3.Rows; r++)
            {
                for (int c = 0; c < m3.Cols; c++)
                {
                    Assert.AreEqual(truth[r, c], m3[r, c]);
                }
            }

        }

        [TestMethod]
        public void NaiveAdd()
        {

        }
    }
}