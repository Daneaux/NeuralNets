using MatrixLibrary;
using MatrixLibrary.Avx;

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
        [DataRow(4, 4)]
        [DataRow(16, 17)]
        [DataRow(23, 41)]
        public void TestTimesColumn(int rows, int cols)
        {
            // Define and initialize a 17x3 matrix with some test values
            float[,] matrixA = new float[rows, cols];
            float[] columnA = new float[cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = (float)r * (float)c + (float)(c + r);
                }
            }

            for (int c = 0; c < cols; c++)
            {
                columnA[c] = (float)12 * (float)c + (float)c;
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxColumnVector v2 = new AvxColumnVector(columnA);

            AvxColumnVector m3 = m1 * v2;

            Assert.AreEqual(m1.Cols, v2.Size);
            Assert.AreEqual(m1.Rows, m3.Size);

            for (int r = 0; r < rows; r++)
            {
                // do the dot product row 'r' dot the column vector
                float dot = 0;
                for (int c = 0; c < cols; c++)
                {
                    dot += matrixA[r, c] * columnA[c];
                }
                Assert.AreEqual(dot, m3[r]);
            }
        }

        [TestMethod]
        [DataRow(4, 4, 2, 1)]
        [DataRow(4, 4, 2, 2)]
        [DataRow(16, 17, 2, 4)]
        [DataRow(23, 41, 2, 5)]
        [DataRow(4, 4, 5, 17)]
        [DataRow(4, 4, 13, 17)]
        [DataRow(16, 17, 5, 17)]
        [DataRow(23, 41, 5, 17)]
        public void TestTimesFlattenedMatrices(int rows, int cols, int lhsRows, int rhsMatCount)
        {
            Random rnd = new Random();

            // 1. create an array of matrices. total size == columns on the lhs matrix.
            // 2. multiply lhs times "array of matrices" using the flattened multiply method
            int lhsCols = rhsMatCount * rows * cols;
            float[,] rhsMatrix = new float[rows, cols];
            float[] rhsLinear = new float[lhsCols];
            float[,] lhs = new float[lhsRows, lhsCols];
            // Fill the rhs matrix with some test values, this 2d float array gets reused in every avxMatrix in the list
            int linear = 0;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    rhsLinear[linear++] = rhsMatrix[r, c] = rnd.Next(1, 100);

            // copy the value over and over into the flattened array for ease of testing
            for (int i = 1; i < rhsMatCount; i++)
                for (int j = 0; j < rhsMatrix.Length; j++)
                    rhsLinear[linear++] = rhsLinear[j];

            for (int r = 0; r < lhsRows; r++)
                for (int c = 0; c < lhsCols; c++)
                    lhs[r, c] = rnd.Next(1, 100);

            AvxMatrix lhsMatrix = new AvxMatrix(lhs);
            List<AvxMatrix> matricesToFlatten = new List<AvxMatrix>();
            for (int i = 0; i < rhsMatCount; i++)
                matricesToFlatten.Add(new AvxMatrix(rhsMatrix));

            FlattenedMatricesAsVector rhsVector = new FlattenedMatricesAsVector(matricesToFlatten);

            var vecResult = rhsVector.MatrixTimesColumn(lhsMatrix);

            Assert.AreEqual(vecResult.Size, lhsRows);
            Assert.AreEqual(lhsMatrix.Cols, rhsLinear.Length);
            Assert.AreEqual(lhsCols, lhsMatrix.Cols);
            Assert.AreEqual(lhsCols, rhsMatrix.Length * rhsMatCount);
            for (int r = 0; r < lhsRows; r++)
            {
                float myDot = DotHelper(lhsMatrix, r, rhsLinear);
                Assert.AreEqual(myDot, vecResult[r], 100);  // turns out that avxmult != software mult for the same numbers
            }
        }

        private float DotHelper(AvxMatrix lhsMatrix, int lhsRow, float[] rhs)
        {
            float dot = 0;
            for (int i = 0; i < rhs.Length; i++)
                dot += lhsMatrix[lhsRow, i] * rhs[i];

            return dot;
        }


        [TestMethod]
        public void TestMatrixMultiplySquare57()
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

            Matrix2D mm1 = new Matrix2D(matrixA);
            Matrix2D mm2 = new Matrix2D(matrixB);
            Matrix2D truth = mm1 * mm2;

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

            Matrix2D mm1 = new Matrix2D(matrixA);
            Matrix2D mm2 = new Matrix2D(matrixB);
            Matrix2D truth = mm1 * mm2;

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

            Matrix2D mm1 = new Matrix2D(matrixA);
            Matrix2D mm2 = new Matrix2D(matrixB);
            Matrix2D truth = mm1 * mm2;

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
        public void TestMatrixMultiply_Larger()
        {
            // Define and initialize a 17x3 matrix with some test values
            int rows = 33;
            int cols = 37;
            float[,] matrixA = new float[rows, cols];

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = r + c *2;
                }
            }

            rows = 37;
            cols = 43;

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = r + c * 3;
                }
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(matrixB);
            AvxMatrix m3 = m1.MatrixTimesMatrix(m2);

            Matrix2D mm1 = new Matrix2D(matrixA);
            Matrix2D mm2 = new Matrix2D(matrixB);
            Matrix2D truth = mm1 * mm2;

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
        [DataRow(6)]
        [DataRow(15)]
        [DataRow(16)]
        [DataRow(17)]
        [DataRow(32)]
        [DataRow(33)]
        public void TestMatrixMultiply_Square16(int size)
        {
            int rows = size;
            int cols = size;
            float[,] matrixA = new float[rows, cols];
            Random rnd = new Random(4134134);

            // Fill the matrix with some test values
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixA[r, c] = (float)(100.0 * rnd.NextDouble());
                }
            }

            float[,] matrixB = new float[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    matrixB[r, c] = (float)(100.0 * rnd.NextDouble());
                }
            }

            AvxMatrix m1 = new AvxMatrix(matrixA);
            AvxMatrix m2 = new AvxMatrix(matrixB); 
            AvxMatrix m3 = m1.MatrixTimesMatrix(m2);

            Matrix2D mm1 = new Matrix2D(matrixA);
            Matrix2D mm2 = new Matrix2D(matrixB);
            Matrix2D truth = mm1 * mm2;

            // Assert.AreEqual(m1.Cols, m3.Rows);

            for (int r = 0; r < m3.Rows; r++)
            {
                for (int c = 0; c < m3.Cols; c++)
                {
                    Assert.AreEqual(truth[r, c], m3[r, c], 0.1);
                }
            }
        }

        [TestMethod]
        public void TestOuterProduct()
        {
            int numLeft = 20;
            int numRight = 19;

            float[] lhs = new float[numLeft];
            for (int c = 0; c < numLeft; c++)
            {
                lhs[c] = (float)(c + 12f + c*2.5f);
            }

            float[] rhs = new float[numRight];
            for(int r = 0; r < numRight; r++)
            {
                rhs[r] = (float)(r * r);
            }

            AvxColumnVector lhsAVx = new AvxColumnVector(lhs);
            AvxColumnVector rhsAvx = new AvxColumnVector(rhs);

            AvxMatrix result = lhsAVx.OuterProduct(rhsAvx);

            Assert.AreEqual(result.Rows, lhs.Length);
            Assert.AreEqual(result.Cols, rhs.Length);

            for(int r = 0; r < result.Rows; r++)
            {
                for(int c = 0; c < result.Cols; c++)
                {
                    Assert.AreEqual(result[r,c], lhs[r] * rhs[c]); 
                }
            }
        }

        [TestMethod]
        public void TestTranspose4x4()
        {
            float[,] floats4x4 = new float[,]
                { {1, 2, 3, 4},
                  {5, 6, 7, 8},
                  {9, 10, 11, 12},
                  {11, 12, 13, 14}
                };

            AvxMatrix Mat4x4 = new AvxMatrix(floats4x4);
            AvxMatrix MT = Mat4x4.GetTransposedMatrix();

            for (int i = 0; i < Mat4x4.Rows; i++)
            {
                for (int j = 0; j < Mat4x4.Cols; j++)
                {
                    Assert.AreEqual(Mat4x4[i, j], MT[j, i]);
                }
            }
        }

        [TestMethod]
        [DataRow(15, 15)]
        [DataRow(3, 3)]
        [DataRow(16, 16)]
        [DataRow(21, 5)]
        [DataRow(32, 16)]
        [DataRow(32, 64)]
        [DataRow(17, 77)]
        public void TestTransposeLarge(int r, int c)
        {
            Random rnd = new Random(4134134);
            float[,] floatsMat = new float[r, c];
            for(int i= 0; i < r; i++)
            {
                for(int j= 0; j < c; j++)
                {
                    floatsMat[i, j] = (float)(100.0 * rnd.NextDouble()); //(float)i + (float)j + (float)i;
                }
            }

            AvxMatrix Mat = new AvxMatrix(floatsMat);
            AvxMatrix MT = Mat.GetTransposedMatrix();

            for (int i = 0; i < Mat.Rows; i++)
            {
                for (int j = 0; j < Mat.Cols; j++)
                {
                    float m1 = Mat[i, j];
                    float m2 = MT[j, i];
                    Assert.AreEqual(m1, m2);
                }
            }
        }
    }
}