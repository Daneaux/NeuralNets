using BenchmarkDotNet.Attributes;
using MatrixLibrary;
using BenchmarkDotNet.Running;
using MatrixLibrary.Avx;

public class Program
{
    public static void Main(string[] args)
    {
        var summary = BenchmarkRunner.Run<IntrinsicsBenchmarks>();
    }
}


public class IntrinsicsBenchmarks
{
    private AvxMatrix m1;
    private AvxMatrix m2;

    private Matrix2D nm1;
    private Matrix2D nm2;
    private Matrix2D filterMat2d;

    private SquareKernel SquareKernel4;
    private SquareKernel SquareKernel8;
    private SquareKernel SquareKernel7;
    private SquareKernel SquareKernel14;

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

        SquareKernel4 = new SquareKernel(4);
        SquareKernel8 = new SquareKernel(8);
        SquareKernel7 = new SquareKernel(7);
        SquareKernel14 = new SquareKernel(14);

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
        Matrix2D m3 = nm1 * nm2;
    }

    [Benchmark]
    public void IntrinsicMatrixMultiply()
    {
        AvxMatrix m3 = m1 * m2;
    }

    [Benchmark]
    public void IntrinsicMatrixMult_Tiled()
    {
        AvxMatrix m3 = m1.MatrixMultiply_Tiled(m2);
    }

    [Benchmark]
    public void NaiveAdd()
    {
        Matrix2D nm3 = nm1 + nm2;
    }

    [Benchmark]
    public void IntrinsicAdd()
    {
        AvxMatrix m3 = m1 + m2;
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
