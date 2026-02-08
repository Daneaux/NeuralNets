using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace CublasTests
{
    [TestClass]
    public sealed class GpuMatrixTests
    {
        private const float Tolerance = 1e-3f;

        [TestInitialize]
        public void CheckGpuAvailable()
        {
            if (!BackendSelector.IsGPUAvailable())
                Assert.Inconclusive("No CUDA GPU available — skipping GPU tests.");
        }

        private static void AssertMatricesEqual(MatrixBase expected, MatrixBase actual, float tolerance = Tolerance)
        {
            Assert.AreEqual(expected.Rows, actual.Rows, "Row count mismatch");
            Assert.AreEqual(expected.Cols, actual.Cols, "Col count mismatch");
            for (int r = 0; r < expected.Rows; r++)
                for (int c = 0; c < expected.Cols; c++)
                    Assert.AreEqual(expected[r, c], actual[r, c], tolerance,
                        $"Mismatch at [{r},{c}]: expected {expected[r, c]}, got {actual[r, c]}");
        }

        private static (GpuMatrix gpu, AvxMatrix avx) CreateRandomPair(int rows, int cols, int seed = 42)
        {
            var gpu = new GpuMatrix(rows, cols);
            gpu.SetRandom(seed, -10, 10);
            var avx = new AvxMatrix(gpu.Mat);
            return (gpu, avx);
        }

        // ============================================================
        // Matrix Multiply
        // ============================================================

        [TestMethod]
        [DataRow(4, 4, 4)]
        [DataRow(3, 5, 2)]
        [DataRow(16, 16, 16)]
        [DataRow(57, 57, 57)]
        [DataRow(128, 64, 96)]
        [DataRow(1, 1, 1)]
        public void TestMultiply(int rowsA, int colsA, int colsB)
        {
            var (gpuA, avxA) = CreateRandomPair(rowsA, colsA, 42);
            var (gpuB, avxB) = CreateRandomPair(colsA, colsB, 43);

            var gpuResult = gpuA.Multiply(gpuB);
            var avxResult = avxA.Multiply(avxB);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestMultiply_LargeMatrix()
        {
            var (gpuA, avxA) = CreateRandomPair(256, 256, 42);
            var (gpuB, avxB) = CreateRandomPair(256, 256, 43);

            var gpuResult = gpuA.Multiply(gpuB);
            var avxResult = avxA.Multiply(avxB);

            AssertMatricesEqual(avxResult, gpuResult, 1.0f); // larger tolerance for big matrices
        }

        [TestMethod]
        public void TestMultiply_KnownValues()
        {
            float[,] a = { { 1, 2 }, { 3, 4 } };
            float[,] b = { { 5, 6 }, { 7, 8 } };

            var gpu = new GpuMatrix(a);
            var result = gpu.Multiply(new GpuMatrix(b));

            Assert.AreEqual(19f, result[0, 0], Tolerance); // 1*5 + 2*7
            Assert.AreEqual(22f, result[0, 1], Tolerance); // 1*6 + 2*8
            Assert.AreEqual(43f, result[1, 0], Tolerance); // 3*5 + 4*7
            Assert.AreEqual(50f, result[1, 1], Tolerance); // 3*6 + 4*8
        }

        [TestMethod]
        [ExpectedException(typeof(ArgumentException))]
        public void TestMultiply_DimensionMismatch()
        {
            var a = new GpuMatrix(3, 4);
            var b = new GpuMatrix(5, 2); // 4 != 5
            a.Multiply(b);
        }

        // ============================================================
        // Matrix-Vector Multiply
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(16, 16)]
        [DataRow(3, 5)]
        [DataRow(57, 41)]
        [DataRow(128, 256)]
        public void TestMatrixTimesColumn(int rows, int cols)
        {
            var (gpuM, avxM) = CreateRandomPair(rows, cols, 42);
            var vec = new float[cols];
            var rnd = new Random(43);
            for (int i = 0; i < cols; i++) vec[i] = (float)(rnd.NextDouble() * 20 - 10);

            var gpuVec = new GpuColumnVector(vec);
            var avxVec = new AvxColumnVector(vec);

            var gpuResult = gpuM.MatrixTimesColumn(gpuVec);
            var avxResult = avxM.MatrixTimesColumn(avxVec);

            Assert.AreEqual(avxResult.Size, gpuResult.Size);
            for (int i = 0; i < rows; i++)
                Assert.AreEqual(avxResult[i], gpuResult[i], Tolerance,
                    $"Mismatch at [{i}]: expected {avxResult[i]}, got {gpuResult[i]}");
        }

        [TestMethod]
        public void TestMatrixTimesColumn_KnownValues()
        {
            float[,] m = { { 1, 2, 3 }, { 4, 5, 6 } };
            float[] v = { 1, 2, 3 };

            var gpu = new GpuMatrix(m);
            var result = gpu.MatrixTimesColumn(new GpuColumnVector(v));

            Assert.AreEqual(14f, result[0], Tolerance); // 1+4+9
            Assert.AreEqual(32f, result[1], Tolerance); // 4+10+18
        }

        // ============================================================
        // Add (Matrix)
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(16, 32)]
        [DataRow(57, 41)]
        public void TestAdd_Matrix(int rows, int cols)
        {
            var (gpuA, avxA) = CreateRandomPair(rows, cols, 42);
            var (gpuB, avxB) = CreateRandomPair(rows, cols, 43);

            var gpuResult = gpuA.Add(gpuB);
            var avxResult = avxA.Add(avxB);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestAdd_Matrix_KnownValues()
        {
            float[,] a = { { 1, 2 }, { 3, 4 } };
            float[,] b = { { 5, 6 }, { 7, 8 } };

            var result = new GpuMatrix(a).Add(new GpuMatrix(b));

            Assert.AreEqual(6f, result[0, 0], Tolerance);
            Assert.AreEqual(8f, result[0, 1], Tolerance);
            Assert.AreEqual(10f, result[1, 0], Tolerance);
            Assert.AreEqual(12f, result[1, 1], Tolerance);
        }

        // ============================================================
        // Subtract
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(16, 32)]
        [DataRow(57, 41)]
        public void TestSubtract(int rows, int cols)
        {
            var (gpuA, avxA) = CreateRandomPair(rows, cols, 42);
            var (gpuB, avxB) = CreateRandomPair(rows, cols, 43);

            var gpuResult = gpuA.Subtract(gpuB);
            var avxResult = avxA.Subtract(avxB);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestSubtract_KnownValues()
        {
            float[,] a = { { 5, 6 }, { 7, 8 } };
            float[,] b = { { 1, 2 }, { 3, 4 } };

            var result = new GpuMatrix(a).Subtract(new GpuMatrix(b));

            Assert.AreEqual(4f, result[0, 0], Tolerance);
            Assert.AreEqual(4f, result[0, 1], Tolerance);
            Assert.AreEqual(4f, result[1, 0], Tolerance);
            Assert.AreEqual(4f, result[1, 1], Tolerance);
        }

        // ============================================================
        // Multiply (Scalar)
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(16, 32)]
        [DataRow(57, 41)]
        public void TestMultiply_Scalar(int rows, int cols)
        {
            var (gpuA, avxA) = CreateRandomPair(rows, cols, 42);
            float scalar = 3.5f;

            var gpuResult = gpuA.Multiply(scalar);
            var avxResult = avxA.Multiply(scalar);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestMultiply_Scalar_KnownValues()
        {
            float[,] a = { { 1, 2 }, { 3, 4 } };
            var result = new GpuMatrix(a).Multiply(2.0f);

            Assert.AreEqual(2f, result[0, 0], Tolerance);
            Assert.AreEqual(4f, result[0, 1], Tolerance);
            Assert.AreEqual(6f, result[1, 0], Tolerance);
            Assert.AreEqual(8f, result[1, 1], Tolerance);
        }

        // ============================================================
        // Add (Scalar) — fallback
        // ============================================================

        [TestMethod]
        public void TestAdd_Scalar()
        {
            var (gpuA, avxA) = CreateRandomPair(16, 16, 42);
            float scalar = 5.5f;

            var gpuResult = gpuA.Add(scalar);
            var avxResult = avxA.Add(scalar);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        // ============================================================
        // Transpose
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(3, 5)]
        [DataRow(16, 32)]
        [DataRow(57, 41)]
        public void TestTranspose(int rows, int cols)
        {
            var (gpuA, avxA) = CreateRandomPair(rows, cols, 42);

            var gpuResult = gpuA.GetTransposedMatrix();
            var avxResult = avxA.GetTransposedMatrix();

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestTranspose_KnownValues()
        {
            float[,] a = { { 1, 2, 3 }, { 4, 5, 6 } };
            var result = new GpuMatrix(a).GetTransposedMatrix();

            Assert.AreEqual(3, result.Rows);
            Assert.AreEqual(2, result.Cols);
            Assert.AreEqual(1f, result[0, 0], Tolerance);
            Assert.AreEqual(4f, result[0, 1], Tolerance);
            Assert.AreEqual(2f, result[1, 0], Tolerance);
            Assert.AreEqual(5f, result[1, 1], Tolerance);
            Assert.AreEqual(3f, result[2, 0], Tolerance);
            Assert.AreEqual(6f, result[2, 1], Tolerance);
        }

        [TestMethod]
        public void TestTranspose_DoubleTranspose_IsIdentity()
        {
            var (gpuA, _) = CreateRandomPair(7, 11, 42);
            var doubleT = gpuA.GetTransposedMatrix().GetTransposedMatrix();
            AssertMatricesEqual(gpuA, doubleT);
        }

        // ============================================================
        // HadamardProduct — fallback
        // ============================================================

        [TestMethod]
        [DataRow(4, 4)]
        [DataRow(16, 32)]
        public void TestHadamardProduct(int rows, int cols)
        {
            var (gpuA, avxA) = CreateRandomPair(rows, cols, 42);
            var (gpuB, avxB) = CreateRandomPair(rows, cols, 43);

            var gpuResult = gpuA.HadamardProduct(gpuB);
            var avxResult = avxA.HadamardProduct(avxB);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        // ============================================================
        // Log — fallback
        // ============================================================

        [TestMethod]
        public void TestLog()
        {
            float[,] a = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
            var gpu = new GpuMatrix(a);
            var sw = new Matrix2D(a);

            var gpuResult = gpu.Log();
            var swResult = sw.Log();

            AssertMatricesEqual(swResult, gpuResult);
        }

        // ============================================================
        // Sum — fallback
        // ============================================================

        [TestMethod]
        public void TestSum()
        {
            var (gpuA, avxA) = CreateRandomPair(16, 16, 42);
            float gpuSum = gpuA.Sum();
            float avxSum = avxA.Sum();
            Assert.AreEqual(avxSum, gpuSum, Tolerance);
        }

        // ============================================================
        // SetDiagonal
        // ============================================================

        [TestMethod]
        public void TestSetDiagonal()
        {
            var gpu = new GpuMatrix(4, 4);
            gpu.SetDiagonal(1.0f);

            for (int i = 0; i < 4; i++)
            {
                Assert.AreEqual(1.0f, gpu[i, i], Tolerance);
                for (int j = 0; j < 4; j++)
                    if (i != j)
                        Assert.AreEqual(0.0f, gpu[i, j], Tolerance);
            }
        }

        // ============================================================
        // Convolution — fallback
        // ============================================================

        [TestMethod]
        [DataRow(10, 12, 4)]
        [DataRow(16, 16, 8)]
        public void TestConvolution(int rows, int cols, int kernelSize)
        {
            var (gpuA, avxA) = CreateRandomPair(rows, cols, 42);
            var (gpuK, avxK) = CreateRandomPair(kernelSize, kernelSize, 43);

            var gpuResult = gpuA.Convolution(gpuK);
            var avxResult = avxA.Convolution(avxK);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        // ============================================================
        // Operator overloads
        // ============================================================

        [TestMethod]
        public void TestOperator_Plus()
        {
            float[,] a = { { 1, 2 }, { 3, 4 } };
            float[,] b = { { 5, 6 }, { 7, 8 } };

            MatrixBase result = new GpuMatrix(a) + new GpuMatrix(b);

            Assert.AreEqual(6f, result[0, 0], Tolerance);
            Assert.AreEqual(12f, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void TestOperator_Minus()
        {
            float[,] a = { { 5, 6 }, { 7, 8 } };
            float[,] b = { { 1, 2 }, { 3, 4 } };

            MatrixBase result = new GpuMatrix(a) - new GpuMatrix(b);

            Assert.AreEqual(4f, result[0, 0], Tolerance);
            Assert.AreEqual(4f, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void TestOperator_ScalarMultiply()
        {
            float[,] a = { { 1, 2 }, { 3, 4 } };
            MatrixBase result = 3.0f * new GpuMatrix(a);

            Assert.AreEqual(3f, result[0, 0], Tolerance);
            Assert.AreEqual(12f, result[1, 1], Tolerance);
        }

        [TestMethod]
        public void TestOperator_MatrixMultiply()
        {
            float[,] a = { { 1, 2 }, { 3, 4 } };
            float[,] b = { { 5, 6 }, { 7, 8 } };

            MatrixBase result = new GpuMatrix(a) * new GpuMatrix(b);

            Assert.AreEqual(19f, result[0, 0], Tolerance);
            Assert.AreEqual(50f, result[1, 1], Tolerance);
        }

        // ============================================================
        // Device memory caching
        // ============================================================

        [TestMethod]
        public void TestDeviceCaching_MultipleOps()
        {
            // Verify that chained operations work correctly
            var (gpuA, avxA) = CreateRandomPair(32, 32, 42);
            var (gpuB, avxB) = CreateRandomPair(32, 32, 43);

            // Chain: (A + B) * 2.0
            var gpuResult = gpuA.Add(gpuB).Multiply(2.0f);
            var avxResult = avxA.Add(avxB).Multiply(2.0f);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        // ============================================================
        // Chaining: GPU operations produce GpuMatrix results that
        // remain device-resident, avoiding redundant host→device
        // uploads when used as inputs to subsequent operations.
        // ============================================================

        [TestMethod]
        public void TestChaining_AddThenMultiply()
        {
            // (A + B) * C — intermediate (A+B) stays on device
            var (gpuA, avxA) = CreateRandomPair(64, 64, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 64, 43);
            var (gpuC, avxC) = CreateRandomPair(64, 64, 44);

            var gpuResult = gpuA.Add(gpuB).Multiply(gpuC);
            var avxResult = avxA.Add(avxB).Multiply(avxC);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestChaining_MultiplyThenAdd()
        {
            // (A * B) + C
            var (gpuA, avxA) = CreateRandomPair(32, 64, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 32, 43);
            var (gpuC, avxC) = CreateRandomPair(32, 32, 44);

            var gpuResult = gpuA.Multiply(gpuB).Add(gpuC);
            var avxResult = avxA.Multiply(avxB).Add(avxC);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestChaining_SubtractThenScalarMultiply()
        {
            // (A - B) * scalar
            var (gpuA, avxA) = CreateRandomPair(128, 128, 42);
            var (gpuB, avxB) = CreateRandomPair(128, 128, 43);

            var gpuResult = gpuA.Subtract(gpuB).Multiply(0.5f);
            var avxResult = avxA.Subtract(avxB).Multiply(0.5f);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestChaining_TransposeThenMultiply()
        {
            // A^T * B — transpose result feeds into multiply
            var (gpuA, avxA) = CreateRandomPair(64, 32, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 48, 43);

            var gpuResult = gpuA.GetTransposedMatrix().Multiply(gpuB);
            var avxResult = avxA.GetTransposedMatrix().Multiply(avxB);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestChaining_MultipleScalarOps()
        {
            // A * 2.0 + B * 3.0 — both scalar results chain into add
            var (gpuA, avxA) = CreateRandomPair(64, 64, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 64, 43);

            var gpuResult = gpuA.Multiply(2.0f).Add(gpuB.Multiply(3.0f));
            var avxResult = avxA.Multiply(2.0f).Add(avxB.Multiply(3.0f));

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestChaining_LongPipeline()
        {
            // ((A + B) * C - D) * 0.1  — 4 chained ops
            var (gpuA, avxA) = CreateRandomPair(32, 32, 42);
            var (gpuB, avxB) = CreateRandomPair(32, 32, 43);
            var (gpuC, avxC) = CreateRandomPair(32, 32, 44);
            var (gpuD, avxD) = CreateRandomPair(32, 32, 45);

            var gpuResult = gpuA.Add(gpuB).Multiply(gpuC).Subtract(gpuD).Multiply(0.1f);
            var avxResult = avxA.Add(avxB).Multiply(avxC).Subtract(avxD).Multiply(0.1f);

            AssertMatricesEqual(avxResult, gpuResult, 1.0f); // larger tolerance for accumulated error
        }

        [TestMethod]
        public void TestChaining_ReuseInputAcrossMultipleOps()
        {
            // Same matrix used in multiple operations stays on device
            var (gpuA, avxA) = CreateRandomPair(64, 64, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 64, 43);

            // gpuA is used 3 times — device pointer should be reused
            var gpuR1 = gpuA.Add(gpuB);
            var gpuR2 = gpuA.Subtract(gpuB);
            var gpuR3 = gpuA.Multiply(gpuB);

            var avxR1 = avxA.Add(avxB);
            var avxR2 = avxA.Subtract(avxB);
            var avxR3 = avxA.Multiply(avxB);

            AssertMatricesEqual(avxR1, gpuR1);
            AssertMatricesEqual(avxR2, gpuR2);
            AssertMatricesEqual(avxR3, gpuR3);
        }

        [TestMethod]
        public void TestChaining_MatVecAfterMatMul()
        {
            // (A * B) * v — matrix multiply result feeds into mat-vec
            var (gpuA, avxA) = CreateRandomPair(32, 64, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 32, 43);

            float[] vecData = new float[32];
            var rnd = new Random(44);
            for (int i = 0; i < 32; i++) vecData[i] = (float)(rnd.NextDouble() * 20 - 10);

            var gpuVec = new GpuColumnVector(vecData);
            var avxVec = new AvxColumnVector(vecData);

            // Chain: (A * B) then multiply by vector
            var gpuMatResult = gpuA.Multiply(gpuB);
            var gpuVecResult = gpuMatResult.MatrixTimesColumn(gpuVec);

            var avxMatResult = avxA.Multiply(avxB);
            var avxVecResult = avxMatResult.MatrixTimesColumn(avxVec);

            Assert.AreEqual(avxVecResult.Size, gpuVecResult.Size);
            for (int i = 0; i < 32; i++)
                Assert.AreEqual(avxVecResult[i], gpuVecResult[i], 0.01f,
                    $"Mismatch at [{i}]"); // relaxed tolerance: chained 64-wide dot products accumulate FP error
        }

        [TestMethod]
        public void TestChaining_IntermediateDispose_NoCorruption()
        {
            // Dispose intermediate results — final result should still be valid
            var (gpuA, avxA) = CreateRandomPair(32, 32, 42);
            var (gpuB, avxB) = CreateRandomPair(32, 32, 43);

            var intermediate = gpuA.Add(gpuB);
            var gpuResult = intermediate.Multiply(2.0f);
            (intermediate as IDisposable)?.Dispose(); // free intermediate GPU memory

            var avxResult = avxA.Add(avxB).Multiply(2.0f);

            // Final result should still be correct despite intermediate being disposed
            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestChaining_TransposeThenAddThenMultiply()
        {
            // (A^T + B) * C — exercises transpose, add, multiply in sequence
            var (gpuA, avxA) = CreateRandomPair(48, 64, 42);
            var (gpuB, avxB) = CreateRandomPair(64, 48, 43);
            var (gpuC, avxC) = CreateRandomPair(64, 48, 44);

            // A is 48x64, A^T is 64x48, B is 64x48, C is 64x48
            // (A^T + B) is 64x48, then * C would be dimension mismatch
            // Instead: (A^T + B) * scalar
            var gpuResult = gpuA.GetTransposedMatrix().Add(gpuB).Multiply(0.25f);
            var avxResult = avxA.GetTransposedMatrix().Add(avxB).Multiply(0.25f);

            AssertMatricesEqual(avxResult, gpuResult);
        }

        [TestMethod]
        public void TestDispose_NoThrow()
        {
            var gpu = new GpuMatrix(4, 4);
            gpu.SetRandom(42, -10, 10);

            // Force device allocation
            var _ = gpu.Multiply(2.0f);

            // Dispose should not throw
            gpu.Dispose();
            gpu.Dispose(); // double dispose should be safe
        }
    }
}
