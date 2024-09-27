using BenchmarkDotNet.Attributes;
using MatrixLibrary;
using BenchmarkDotNet.Running;

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
    private AvxMatrix filterAvx;

    private Matrix2D nm1;
    private Matrix2D nm2;
    private Matrix2D filterMat2d;
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


        float[,] filter = new float[4, 4]{
            { 1, 1.5f, -1, 0 },
            { 0, 1, 1.5f, -1 },
            { -1, 0, 1.4f, 0 },
            { 0, -1.1f, 2f, 1 }};

        filterAvx = new AvxMatrix(filter);
        filterMat2d = new Matrix2D(filter);
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
    public void AvxConvolution()
    {
        AvxMatrix m3 = m1.Convolution(filterAvx);
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
