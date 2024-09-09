﻿using BenchmarkDotNet.Attributes;
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

    private Matrix2D nm1;
    private Matrix2D nm2;
    [GlobalSetup]
    public void Setup()
    {
        int matSize = 400;
        int seed = 1253443;
        m1 = new AvxMatrix(matSize, matSize);
        m2 = new AvxMatrix(matSize, matSize);

        m1.SetRandom(seed, -100, 100);
        m2.SetRandom(seed, -100, 100);

        nm1 = new Matrix2D(matSize, matSize);
        nm2 = new Matrix2D(matSize, matSize);

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
        Matrix2D m3 = nm1 * nm2;
    }

/*    [Benchmark]
    public void IntrinsicAdd()
    {
        AvxMatrix m3 = m1 + m2;
    }*/

/*    [Benchmark]
    public void NaiveAdd()
    {
        Matrix nm3 = nm1 + nm2;
    }*/

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

    [Benchmark]
    public void IntrinsicMult_Tiled()
    {
        AvxMatrix m3 = m1.MatrixMultiply_Tiled(m2);
    }

}
