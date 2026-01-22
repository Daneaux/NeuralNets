using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNetsTests.Math
{
    [TestClass]
    public class ConvolutionTests
    {
        [TestMethod]
        public void ForwardPass()
        {

        }

        [TestMethod]
        [DataRow(10, 12, 4)]
        [DataRow(10, 12, 5)]
        [DataRow(10, 12, 8)]
        [DataRow(23, 19, 7)]
        [DataRow(43, 67, 13)]
        public void Convolution(int rows, int cols, int kernelSize)
        {
            Random rnd = new Random();

            // Create a rows X cols source matrix
            float[,] source = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    source[i, j] = i * 12 + j + 1;
                }
            }

            // Create a filter
            float[,] kernel = new float[kernelSize, kernelSize];
            for (int i = 0; i < kernelSize; i++)
                for (int j = 0; j < kernelSize; j++)
                    kernel[j, i] = (float)rnd.Next(-10, 10);

            // Calculate expected output dimensions
            int outputRows = source.GetLength(0) - kernel.GetLength(0) + 1;
            int outputCols = source.GetLength(1) - kernel.GetLength(1) + 1;

            // Create expected output matrix
            float[,] expectedOutput = new float[outputRows, outputCols];

            // Calculate expected output
            for (int i = 0; i < outputRows; i++)
            {
                for (int j = 0; j < outputCols; j++)
                {
                    float sum = 0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++)
                        {
                            sum += source[i + m, j + n] * kernel[m, n];
                        }
                    }
                    expectedOutput[i, j] = sum;
                }
            }

            AvxMatrix lhs = new AvxMatrix(source);
            var rhs = new AvxMatrix(kernel);
            AvxMatrix result = lhs.Convolution(rhs);
            VerifyOutput(expectedOutput, result.Mat);
        }

        [TestMethod]
        [DataRow(4, 4, 3)]
        [DataRow(10, 12, 5)]
        [DataRow(10, 12, 8)]
        [DataRow(23, 19, 7)]
        [DataRow(43, 67, 13)]
        public void ConvolutionFull(int rows, int cols, int kernelSize)
        {
            Random rnd = new Random();

            // Create a rows X cols source matrix
            int padding = kernelSize - 1;
            int twoP = 2 * padding;
            float[,] paddedSource = new float[rows + twoP, cols + twoP];
            float[,] unpaddedSource = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float val = i * 12 + j + 1; ;
                    unpaddedSource[i, j] = val;
                    paddedSource[i + padding, j + padding] = val;
                }
            }

            // Create a filter
            float[,] kernel = new float[kernelSize, kernelSize];
            for (int i = 0; i < kernelSize; i++)
                for (int j = 0; j < kernelSize; j++)
                    kernel[j, i] = (float)rnd.Next(-10, 10);

            // Calculate expected output dimensions
            int outputRows = rows + kernelSize - 1; // m - k + 1 + 2k - 2 = m + k - 1
            int outputCols = cols + kernelSize - 1;

            // Create expected output matrix
            float[,] expectedOutput = new float[outputRows, outputCols];

            // Calculate expected output
            for (int i = 0; i < outputRows; i++)
            {
                for (int j = 0; j < outputCols; j++)
                {
                    float sum = 0;
                    for (int m = 0; m < kernelSize; m++)
                    {
                        for (int n = 0; n < kernelSize; n++)
                        {
                            int srcR = i + m;
                            int srcC = j + n;
                            sum += paddedSource[srcR, srcC] * kernel[m, n];
                        }
                    }
                    expectedOutput[i, j] = sum;
                }
            }

            AvxMatrix lhs = new AvxMatrix(paddedSource);
            var rhs = new AvxMatrix(kernel);
            AvxMatrix result = lhs.ConvolutionFull(rhs);
            VerifyOutput(expectedOutput, result.Mat, true);
        }

        static void VerifyOutput(float[,] expected, float[,] actual, bool isPadded = false)
        {
            //Assert.AreEqual(expected.GetLength(0), actual.GetLength(0));
            //Assert.AreEqual(expected.GetLength(1), actual.GetLength(1));
            int rows = expected.GetLength(0);
            int cols = expected.GetLength(1);
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    int actualR = i;
                    int actualC = j;
                    if(isPadded)
                    {
                        int deltaR = (actual.GetLength(0) - rows) / 2;
                        int deltaC = (actual.GetLength(1) - cols) / 2;
                        Debug.Assert((actual.GetLength(0) - rows) % 2 == 0);
                        Debug.Assert((actual.GetLength(1) - cols) % 2 == 0);
                        actualR += deltaR;
                        actualC += deltaC;
                    }
                    Assert.AreEqual(expected[i, j], actual[actualR, actualC], 0.00001);
                }
            }
        }

    }
}
