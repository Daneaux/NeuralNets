using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeuralNets;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets.Tests
{
    [TestClass()]
    public class MatrixTests
    {
        [TestMethod()]
        public void TestSoftMax()
        {
            ColumnVector vec = new ColumnVector(new double[] { 1.0, 2.0, 3.0 });
            ColumnVector softMax = vec.SoftmaxHelper();
            Assert.AreEqual(softMax[0], 0.09003057, 0.001);
            Assert.AreEqual(softMax[1], 0.24472847, 0.001);
            Assert.AreEqual(softMax[2], 0.66524096, 0.001);
            Assert.AreEqual(softMax.Sum(), 1.0, 0.001);

            // Make sure we're numerically stable
            vec = new ColumnVector(new double[] { 10000, 20000, 30000 });
            softMax = vec.SoftmaxHelper();
            Assert.AreEqual(softMax[0], 0.0, 0.001);
            Assert.AreEqual(softMax[1], 0.0, 0.001);
            Assert.AreEqual(softMax[2], 1.0, 0.001);
            Assert.AreEqual(softMax.Sum(), 1.0, 0.001);
        }

        [TestMethod()]
        public void CreateRandom()
        {
            Matrix m = new Matrix(3, 4);
            m.SetRandom(777, 0, 10);
        }

        [TestMethod]
        public void HadamardProduct()
        {
            double[,] ddd = new double[3, 3]{
                {1.0, 2.0, 9.0},
                {4.0, 5.0, 6.0},
                {7, 8, 9}};
            Matrix m1 = new Matrix(ddd);
            Matrix m2 = new Matrix(ddd);
            Matrix m3 = m1.HadamardProduct(m2);
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
            double[,] ddd = new double[2, 3]
{
                {1.0, 2.0, 9.0},
                {4.0, 5.0, 6.0}
};
            Matrix m1 = new Matrix(ddd);

            double[,] eee = new double[3, 2]
            {
                {1, 4},
                {2, 5},
                {3, 6}
            };
            Matrix m2 = new Matrix(eee);

            Matrix m12 = m1 * m2;

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
            Matrix m = new Matrix(3, 4);
            m.SetRandom(777, 0, 3);
            m.Mat[0, 1] = 2323;
            m.Mat[1, 3] = 7878;
            m.Mat[2, 2] = 9999;
            Matrix mt = m.GetTransposedMatrix();

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
    }
}