using BenchmarkDotNet.Attributes;
//using BenchmarkDotNet.Running;
using NeuralNets;
using IntrinsicMatrix;
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
    [GlobalSetup]
    public void Setup()
    {

    }

    [GlobalCleanup]
    public void GlobalCleanup()
    {
    }

    [Benchmark]
    public void NaiveMultiply()
    {
        Matrix m1 = new Matrix(100, 100);
        Matrix m2 = new Matrix(100, 100);

        Matrix m3 = m1 * m2;
    }

    [Benchmark]
    public void IntrinsicAdd()
    {
        AvxMatrix m1 = new AvxMatrix(100, 100);
        AvxMatrix m2 = new AvxMatrix(100, 100);
        AvxMatrix m3 = m1 + m2;
    }

    [Benchmark]
    public void NaiveAdd()
    {
        Matrix m1 = new Matrix(100, 100);
        Matrix m2 = new Matrix(100, 100);
        Matrix m3 = m1 + m2;
    }

}
