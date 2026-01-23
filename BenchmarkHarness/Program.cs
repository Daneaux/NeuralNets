using MatrixLibrary;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using MatrixLibrary.BaseClasses;

public class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--gpu")
        {
            var summary = BenchmarkRunner.Run<GpuBenchmarks>();
        }
        else if (args.Length > 0 && args[0] == "--all")
        {
            BenchmarkRunner.Run<IntrinsicsBenchmarks>();
            if (BackendSelector.IsGPUAvailable())
                BenchmarkRunner.Run<GpuBenchmarks>();
        }
        else
        {
            var summary = BenchmarkRunner.Run<IntrinsicsBenchmarks>();
        }
    }
}


public class IntrinsicsBenchmarks
{
    private AvxMatrix m1;
    private AvxMatrix m2;

    private Matrix2D nm1;
    private Matrix2D nm2;
    private Matrix2D filterMat2d;

    private AvxMatrix SquareKernel4;
    private AvxMatrix SquareKernel8;
    private AvxMatrix SquareKernel7;
    private AvxMatrix SquareKernel14;

    [GlobalSetup]
    public void Setup()
    {
        int matSize = 1024;
        int range = 100;
        int seed = 1253443;
        m1 = new AvxMatrix(matSize, matSize);
        m2 = new AvxMatrix(matSize, matSize);

        m1.SetRandom(seed, -range, range);
        m2.SetRandom(seed, -range, range);

        nm1 = new Matrix2D(matSize, matSize);
        nm2 = new Matrix2D(matSize, matSize);

        nm1.SetRandom(seed, -range, range);
        nm2.SetRandom(seed, -range, range);

        SquareKernel4 = new AvxMatrix(4,4);
        SquareKernel8 = new AvxMatrix(8,8);
        SquareKernel7 = new AvxMatrix(7, 7);
        SquareKernel14 = new AvxMatrix(14, 14);

        SquareKernel4.SetRandom(seed, -range, range);
        SquareKernel8.SetRandom(seed, -range, range);
        SquareKernel7.SetRandom(seed, -range, range);
        SquareKernel14.SetRandom(seed, -range, range);

        filterMat2d = new Matrix2D(4, 4);
        filterMat2d.SetRandom(seed, -range, range);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
    }

    [Benchmark]
    public void NaiveConvolution()
    {
        Matrix2D m3 = nm1.Convolution(filterMat2d);
    }

    [Benchmark]
    public void AvxConvolution4x4()
    {
        AvxMatrix m3 = m1.Convolution(SquareKernel4);
    }

    [Benchmark]
    public void AvxConvolution8x8()
    {
        AvxMatrix m3 = m1.Convolution(SquareKernel8);
    }

    [Benchmark]
    public void AvxConvolution7x7()
    {
        AvxMatrix m3 = m1.Convolution(SquareKernel7);
    }

    [Benchmark]
    public void AvxConvolution14x14()
    {
        AvxMatrix m3 = m1.Convolution(SquareKernel14);
    }

    [Benchmark]
    public void NaiveMatrixMultiply()
    {
        MatrixBase m3 = nm1 * nm2;
    }

    [Benchmark]
    public void IntrinsicMatrixMultiply()
    {
        MatrixBase m3 = m1 * m2;
    }

    [Benchmark]
    public void IntrinsicMatrixMult_Tiled()
    {
        AvxMatrix m3 = m1.MatrixMultiply_Tiled(m2);
    }

    [Benchmark]
    public void NaiveAdd()
    {
        MatrixBase nm3 = nm1 + nm2;
    }

    [Benchmark]
    public void IntrinsicAdd()
    {
        MatrixBase m3 = m1 + m2;
    }

    [Benchmark]
    public void NaiveTranspose()
    {
        Matrix2D mT = nm1.GetTransposedMatrix();
    }

    [Benchmark]
    public void AvxTranspose()
    {
        AvxMatrix MT = m1.GetTransposedMatrix();
    }
}


public class GpuBenchmarks
{
    private GpuMatrix g1;
    private GpuMatrix g2;
    private GpuMatrix gpuKernel4;
    private GpuColumnVector gpuVec;

    private AvxMatrix m1;
    private AvxMatrix m2;

    private Matrix2D nm1;
    private Matrix2D nm2;

    [GlobalSetup]
    public void Setup()
    {
        if (!BackendSelector.IsGPUAvailable())
            throw new InvalidOperationException("GPU not available for benchmarks");

        int matSize = 1024;
        int range = 100;
        int seed = 1253443;

        g1 = new GpuMatrix(matSize, matSize);
        g2 = new GpuMatrix(matSize, matSize);
        g1.SetRandom(seed, -range, range);
        g2.SetRandom(seed, -range, range);

        m1 = new AvxMatrix(g1.Mat);
        m2 = new AvxMatrix(g2.Mat);

        nm1 = new Matrix2D(g1.Mat);
        nm2 = new Matrix2D(g2.Mat);

        gpuKernel4 = new GpuMatrix(4, 4);
        gpuKernel4.SetRandom(seed, -range, range);

        float[] vecData = new float[matSize];
        var rnd = new Random(seed);
        for (int i = 0; i < matSize; i++) vecData[i] = (float)(rnd.NextDouble() * 2 * range - range);
        gpuVec = new GpuColumnVector(vecData);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
        g1?.Dispose();
        g2?.Dispose();
        gpuKernel4?.Dispose();
        gpuVec?.Dispose();
    }

    [Benchmark]
    public void GpuMatrixMultiply()
    {
        MatrixBase m3 = g1.Multiply(g2);
        (m3 as IDisposable)?.Dispose();
    }

    [Benchmark]
    public void NaiveMatrixMultiply()
    {
        MatrixBase m3 = nm1 * nm2;
    }

    [Benchmark]
    public void AvxMatrixMultiply()
    {
        MatrixBase m3 = m1 * m2;
    }

    [Benchmark]
    public void GpuAdd()
    {
        MatrixBase m3 = g1.Add(g2);
        (m3 as IDisposable)?.Dispose();
    }

    [Benchmark]
    public void AvxAdd()
    {
        MatrixBase m3 = m1 + m2;
    }

    [Benchmark]
    public void GpuTranspose()
    {
        MatrixBase mT = g1.GetTransposedMatrix();
        (mT as IDisposable)?.Dispose();
    }

    [Benchmark]
    public void AvxTranspose()
    {
        AvxMatrix mT = m1.GetTransposedMatrix();
    }

    [Benchmark]
    public void GpuScalarMultiply()
    {
        MatrixBase m3 = g1.Multiply(2.5f);
        (m3 as IDisposable)?.Dispose();
    }

    [Benchmark]
    public void AvxScalarMultiply()
    {
        MatrixBase m3 = m1.Multiply(2.5f);
    }

    [Benchmark]
    public void GpuMatrixTimesColumn()
    {
        ColumnVectorBase result = g1.MatrixTimesColumn(gpuVec);
        (result as IDisposable)?.Dispose();
    }

    [Benchmark]
    public void GpuConvolution4x4()
    {
        MatrixBase m3 = g1.Convolution(gpuKernel4);
        (m3 as IDisposable)?.Dispose();
    }

    [Benchmark]
    public void GpuHadamardProduct()
    {
        MatrixBase m3 = g1.HadamardProduct(g2);
        (m3 as IDisposable)?.Dispose();
    }
}
