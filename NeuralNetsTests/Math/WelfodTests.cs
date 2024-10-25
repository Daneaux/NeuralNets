using NeuralNets;
using MatrixLibrary;


namespace NeuralNets.Tests
{
    [TestClass()]
    public class WelfordAlgorithmTests
    {
        private const double Epsilon = 0.0000001;

        [TestMethod()]
        public void TestEmptySet()
        {
            AvxColumnVector vec = new AvxColumnVector(new float[] { });
            var stats = new WelfordAlgorithm(vec);
            Assert.AreEqual(0, stats.Mean);
            Assert.AreEqual(0, stats.GetVariance());
            Assert.AreEqual(0, stats.GetStandardDeviation());
        }

        [TestMethod()]
        public void TestSingleValue()
        {
            AvxColumnVector vec = new AvxColumnVector(new float[] { 5 });
            var stats = new WelfordAlgorithm(vec);
            Assert.AreEqual(5, stats.Mean);
            Assert.AreEqual(0, stats.GetVariance());
            Assert.AreEqual(0, stats.GetStandardDeviation());
        }


        [TestMethod()]
        public void TestMultipleValues()
        {
            float[] values = { 2, 4, 4, 4, 5, 5, 7, 9 };
            AvxColumnVector vec = new AvxColumnVector(values);
            var stats = new WelfordAlgorithm(vec);

            Assert.AreEqual(5, stats.Mean, Epsilon);
            Assert.AreEqual(4, stats.GetVariance(), 0.00001);
            Assert.AreEqual(2, stats.GetStandardDeviation(), 0.00001);
        }


        [TestMethod()]
        [DataRow(100000, 100001, 100002)]
        [DataRow(10000000, 10000001, 10000002)]
        public void TestLargeNumbers(float a, float b, float c)
        {
            float[] values = { a, b, c };
            AvxColumnVector vec = new AvxColumnVector(values);
            var stats = new WelfordAlgorithm(vec);

            float t1 = values[0];
            float t2 = values[1];
            float t3 = values[2];

            Assert.AreEqual(t2, (float)stats.Mean, 0.0000001);
            Assert.AreEqual(0.6666666666666, stats.GetVariance(), Epsilon);
            Assert.AreEqual(0.81649658092773, stats.GetStandardDeviation(), Epsilon);
        }

        [TestMethod()]
        public void TestLargeNumbers2()
        {
            float[] values = new float[15];
            for (int i = 0; i < 15; i++)
                values[i] = 1000000f;
            values[8] = 1000001f;

            AvxColumnVector vec = new AvxColumnVector(values);
            var stats = new WelfordAlgorithm(vec);

            Assert.AreEqual(1000000.0625, (float)stats.Mean, 0.00001);
            Assert.AreEqual(0.062222222222222, stats.GetVariance(), 0.000001);
            Assert.AreEqual(0.24944382578493, stats.GetStandardDeviation(), 0.0000001);
        }

        [TestMethod()]
        [DataRow(1000000f, 100)]
        [DataRow(1000000f, 1000)]
        [DataRow(1000000f, 5000)]
        [DataRow(1000000f, 1000000)]
        public void TestLargeNumbers3(float number, int count)
        {
            float[] values = new float[count];
            for (int i = 0; i < count / 2; i++)
                values[i] = number;
            for (int i = count / 2; i < count; i++)
                values[i] = number + 1;

            AvxColumnVector vec = new AvxColumnVector(values);
            var stats = new WelfordAlgorithm(vec);

            Assert.AreEqual(number + 0.5f, (float)stats.Mean, 0.01);
            Assert.AreEqual(0.25, stats.GetVariance(), 0.01);
            Assert.AreEqual(0.5, stats.GetStandardDeviation(), 0.01);
        }


        [TestMethod()]
        public void TestNegativeNumbers()
        {
            float[] values = { -5, -3, -1, 1, 3, 5 };
            AvxColumnVector vec = new AvxColumnVector(values);
            var stats = new WelfordAlgorithm(vec);

            Assert.AreEqual(0, stats.Mean, Epsilon);
            Assert.AreEqual(11.66666666666666, stats.GetVariance(), 0.00000001);
            Assert.AreEqual(3.4156502553199, stats.GetStandardDeviation(), 0.00000001);
        }

        [TestMethod()]
        public void TestPrecision()
        {
            float[] values = { 1.1f, 1.2f, 1.3f, 1.4f, 1.5f };
            AvxColumnVector vec = new AvxColumnVector(values);
            var stats = new WelfordAlgorithm(vec);

            Assert.AreEqual(1.3, stats.Mean, Epsilon);
            Assert.AreEqual(0.02, stats.GetVariance(), Epsilon);
            Assert.AreEqual(0.14142135623731, stats.GetStandardDeviation(), Epsilon);
        }
    }
}
