using MatrixLibrary;

namespace NeuralNets.Tests
{
    [TestClass()]
    public class MatrixTests
    {
        [TestMethod()]
        public void TestSoftMax()
        {
            ColumnVector vec = new ColumnVector(new float[] { 1, 2, 3 });
            ColumnVector softMax = vec.SoftmaxHelper();
            Assert.AreEqual(softMax[0], 0.09003057, 0.001);
            Assert.AreEqual(softMax[1], 0.24472847, 0.001);
            Assert.AreEqual(softMax[2], 0.66524096, 0.001);
            Assert.AreEqual(softMax.Sum(), 1, 0.001);

            // Make sure we're numerically stable
            vec = new ColumnVector(new float[] { 10000, 20000, 30000 });
            softMax = vec.SoftmaxHelper();
            Assert.AreEqual(softMax[0], 0, 0.001);
            Assert.AreEqual(softMax[1], 0, 0.001);
            Assert.AreEqual(softMax[2], 1, 0.001);
            Assert.AreEqual(softMax.Sum(), 1, 0.001);
        }

        [TestMethod()]
        public void CreateRandom()
        {
            Matrix2D m = new Matrix2D(3, 4);
            m.SetRandom(777, 0, 10);
        }

        [TestMethod]
        public void HadamardProduct()
        {
            float[,] ddd = new float[3, 3]{
                {1, 2, 9},
                {4, 5, 6},
                {7, 8, 9}};
            Matrix2D m1 = new Matrix2D(ddd);
            Matrix2D m2 = new Matrix2D(ddd);
            Matrix2D m3 = m1.HadamardProduct(m2);
            Assert.AreEqual(m1[1, 1], 5);
            Assert.AreEqual(m2[1, 1], 5);
            for (int r = 0; r < m1.Rows; r++)
                for (int c = 0; c < m1.Cols; c++)
                {
                    Assert.AreEqual(m3[r, c], m1[r, c] * m2[r, c]);
                }
        }

        [TestMethod]
        public void MultiplyOperator()
        {
            float[,] ddd = new float[2, 3]
{
                {1, 2, 9},
                {4, 5, 6}
};
            Matrix2D m1 = new Matrix2D(ddd);

            float[,] eee = new float[3, 2]
            {
                {1, 4},
                {2, 5},
                {3, 6}
            };
            Matrix2D m2 = new Matrix2D(eee);

            Matrix2D m12 = m1 * m2;

            Assert.AreEqual(2, m12.Rows);
            Assert.AreEqual(2, m12.Cols);

            Assert.AreEqual(1 + 4 + (3 * 9), m12.Mat[0, 0]);
            Assert.AreEqual(4 + 10 + (6 * 9), m12.Mat[0, 1]);
            Assert.AreEqual(4 + 10 + 18, m12.Mat[1, 0]);
            Assert.AreEqual(16 + 25 + 36, m12.Mat[1, 1]);
        }

        [TestMethod]
        public void Transpose()
        {
            Matrix2D m = new Matrix2D(3, 4);
            m.SetRandom(777, 0, 3);
            m.Mat[0, 1] = 2323;
            m.Mat[1, 3] = 7878;
            m.Mat[2, 2] = 9999;
            Matrix2D mt = m.GetTransposedMatrix();

            Assert.AreEqual(3, m.Rows);
            Assert.AreEqual(4, mt.Rows);
            Assert.AreEqual(3, mt.Cols);
            Assert.AreEqual(m.Mat[0, 1], mt.Mat[1, 0]);
            Assert.AreEqual(m.Mat[1, 3], mt.Mat[3, 1]);
            Assert.AreEqual(m.Mat[2, 2], mt.Mat[2, 2]);
        }

        [TestMethod]
        public void Vectors()
        {
            // todo test row and column vectors
        }

        [TestMethod]
        public void Convolution()
        {
            // Create a 10x12 source matrix
            float[,] source = new float[10, 12];
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 12; j++)
                {
                    source[i, j] = i * 12 + j + 1;
                }
            }

            // Create a 4x4 filter
            float[,] filter = new float[4, 4]
            {
            { 1, 1.5f, -1, 0 },
            { 0, 1, 1.5f, -1 },
            { -1, 0, 1.4f, 0 },
            { 0, -1.1f, 2f, 1 }
            };

            // Calculate expected output dimensions
            int outputRows = source.GetLength(0) - filter.GetLength(0) + 1;
            int outputCols = source.GetLength(1) - filter.GetLength(1) + 1;

            // Create expected output matrix
            float[,] expectedOutput = new float[outputRows, outputCols];

            // Calculate expected output
            for (int i = 0; i < outputRows; i++)
            {
                for (int j = 0; j < outputCols; j++)
                {
                    float sum = 0;
                    for (int m = 0; m < 4; m++)
                    {
                        for (int n = 0; n < 4; n++)
                        {
                            sum += source[i + m, j + n] * filter[m, n];
                        }
                    }
                    expectedOutput[i, j] = sum;
                }
            }

            // TODO: Replace this line with a call to your convolution operator
            Matrix2D lhs = new Matrix2D(source);
            Matrix2D rhs = new Matrix2D(filter);
            Matrix2D result = lhs.Convolution(rhs);

            // Verify the output
            VerifyOutput(expectedOutput, result.Mat);
       } 

        static void VerifyOutput(float[,] expected, float[,] actual)
        {
            Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));

            for (int i = 0; i < expected.GetLength(0); i++)
            {
                for (int j = 0; j < expected.GetLength(1); j++)
                {
                    Assert.AreEqual(expected[i, j], actual[i, j], 0.00001);
                }
            }
        }

    }


}