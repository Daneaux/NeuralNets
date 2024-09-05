using BenchmarkDotNet.Attributes;
//using BenchmarkDotNet.Running;
using NeuralNets;
using IntrinsicMatrix;
using BenchmarkDotNet.Running;
using Microsoft.CodeAnalysis.Emit;

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
    private AvxMatrix m2T;

    private Matrix nm1;
    private Matrix nm2;
    [GlobalSetup]
    public void Setup()
    {
        int seed = 1253443;
        m1 = new AvxMatrix(1000, 1000);
        m2 = new AvxMatrix(1000, 1000);
        

        m1.SetRandom(seed, -100, 100);
        m2.SetRandom(seed, -100, 100);

        nm1 = new Matrix(1000, 1000);
        nm2 = new Matrix(1000, 1000);

        nm1.SetRandom(seed, -1000, 1000);
        nm2.SetRandom(seed, -1000, 1000);

        var T = nm2.GetTransposedMatrix();
        m2T = new AvxMatrix(T.Mat);
    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
    }

    [Benchmark]
    public void NaiveMultiply()
    {
        Matrix m3 = nm1 * nm2;
    }

    [Benchmark]
    public void IntrinsicAdd()
    {
        AvxMatrix m3 = m1 + m2;
    }

    [Benchmark]
    public void NaiveAdd()
    {
        Matrix nm3 = nm1 + nm2;
    }

    [Benchmark]
    public void IntrinsicMult()
    {
        AvxMatrix m3 = m1 * m2;
    }


    [Benchmark]
    public void IntrinsicMult_TransposeFirst()
    {
        AvxMatrix m3 = m1.MatrixTimesMatrix_TransposedRHS(m2T);
    }

}
