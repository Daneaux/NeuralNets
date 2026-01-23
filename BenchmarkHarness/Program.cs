using MatrixLibrary;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Reports;
using MatrixLibrary.BaseClasses;

public class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0] == "--conv")
        {
            BenchmarkRunner.Run<ConvolutionBenchmarks>();
        }
        else if (args.Length > 0 && args[0] == "--all")
        {
            BenchmarkRunner.Run<MultiplyBenchmarks>();
            BenchmarkRunner.Run<AddBenchmarks>();
            BenchmarkRunner.Run<SubtractBenchmarks>();
            BenchmarkRunner.Run<TransposeBenchmarks>();
            BenchmarkRunner.Run<ConvolutionBenchmarks>();
        }
        else
        {
            BenchmarkRunner.Run<MultiplyBenchmarks>();
            BenchmarkRunner.Run<AddBenchmarks>();
            BenchmarkRunner.Run<SubtractBenchmarks>();
            BenchmarkRunner.Run<TransposeBenchmarks>();
        }
    }
}

// ============================================================
// Multiply Benchmarks: Software vs AVX vs GPU at 64, 256, 1024
// ============================================================

[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class MultiplyBenchmarks
{
    [Params(64, 256, 1024)]
    public int Size;

    private Matrix2D sw1, sw2;
    private AvxMatrix avx1, avx2;
    private GpuMatrix gpu1, gpu2;
    private bool gpuAvailable;

    [GlobalSetup]
    public void Setup()
    {
        int range = 100;
        int seed = 1253443;

        sw1 = new Matrix2D(Size, Size);
        sw2 = new Matrix2D(Size, Size);
        sw1.SetRandom(seed, -range, range);
        sw2.SetRandom(seed + 1, -range, range);

        avx1 = new AvxMatrix(sw1.Mat);
        avx2 = new AvxMatrix(sw2.Mat);

        gpuAvailable = BackendSelector.IsGPUAvailable();
        if (gpuAvailable)
        {
            gpu1 = new GpuMatrix(sw1.Mat);
            gpu2 = new GpuMatrix(sw2.Mat);
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        gpu1?.Dispose();
        gpu2?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public MatrixBase Software() => sw1.Multiply(sw2);

    [Benchmark]
    public MatrixBase AVX() => avx1.Multiply(avx2);

    [Benchmark]
    public MatrixBase GPU()
    {
        if (!gpuAvailable) return avx1.Multiply(avx2);
        var r = gpu1.Multiply(gpu2);
        (r as IDisposable)?.Dispose();
        return r;
    }
}

// ============================================================
// Add Benchmarks: Software vs AVX vs GPU at 64, 256, 1024
// ============================================================

[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class AddBenchmarks
{
    [Params(64, 256, 1024)]
    public int Size;

    private Matrix2D sw1, sw2;
    private AvxMatrix avx1, avx2;
    private GpuMatrix gpu1, gpu2;
    private bool gpuAvailable;

    [GlobalSetup]
    public void Setup()
    {
        int range = 100;
        int seed = 1253443;

        sw1 = new Matrix2D(Size, Size);
        sw2 = new Matrix2D(Size, Size);
        sw1.SetRandom(seed, -range, range);
        sw2.SetRandom(seed + 1, -range, range);

        avx1 = new AvxMatrix(sw1.Mat);
        avx2 = new AvxMatrix(sw2.Mat);

        gpuAvailable = BackendSelector.IsGPUAvailable();
        if (gpuAvailable)
        {
            gpu1 = new GpuMatrix(sw1.Mat);
            gpu2 = new GpuMatrix(sw2.Mat);
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        gpu1?.Dispose();
        gpu2?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public MatrixBase Software() => sw1.Add(sw2);

    [Benchmark]
    public MatrixBase AVX() => avx1.Add(avx2);

    [Benchmark]
    public MatrixBase GPU()
    {
        if (!gpuAvailable) return avx1.Add(avx2);
        var r = gpu1.Add(gpu2);
        (r as IDisposable)?.Dispose();
        return r;
    }
}

// ============================================================
// Subtract Benchmarks: Software vs AVX vs GPU at 64, 256, 1024
// ============================================================

[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class SubtractBenchmarks
{
    [Params(64, 256, 1024)]
    public int Size;

    private Matrix2D sw1, sw2;
    private AvxMatrix avx1, avx2;
    private GpuMatrix gpu1, gpu2;
    private bool gpuAvailable;

    [GlobalSetup]
    public void Setup()
    {
        int range = 100;
        int seed = 1253443;

        sw1 = new Matrix2D(Size, Size);
        sw2 = new Matrix2D(Size, Size);
        sw1.SetRandom(seed, -range, range);
        sw2.SetRandom(seed + 1, -range, range);

        avx1 = new AvxMatrix(sw1.Mat);
        avx2 = new AvxMatrix(sw2.Mat);

        gpuAvailable = BackendSelector.IsGPUAvailable();
        if (gpuAvailable)
        {
            gpu1 = new GpuMatrix(sw1.Mat);
            gpu2 = new GpuMatrix(sw2.Mat);
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        gpu1?.Dispose();
        gpu2?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public MatrixBase Software() => sw1.Subtract(sw2);

    [Benchmark]
    public MatrixBase AVX() => avx1.Subtract(avx2);

    [Benchmark]
    public MatrixBase GPU()
    {
        if (!gpuAvailable) return avx1.Subtract(avx2);
        var r = gpu1.Subtract(gpu2);
        (r as IDisposable)?.Dispose();
        return r;
    }
}

// ============================================================
// Transpose Benchmarks: Software vs AVX vs GPU at 64, 256, 1024
// ============================================================

[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByParams)]
public class TransposeBenchmarks
{
    [Params(64, 256, 1024)]
    public int Size;

    private Matrix2D sw1;
    private AvxMatrix avx1;
    private GpuMatrix gpu1;
    private bool gpuAvailable;

    [GlobalSetup]
    public void Setup()
    {
        int range = 100;
        int seed = 1253443;

        sw1 = new Matrix2D(Size, Size);
        sw1.SetRandom(seed, -range, range);

        avx1 = new AvxMatrix(sw1.Mat);

        gpuAvailable = BackendSelector.IsGPUAvailable();
        if (gpuAvailable)
        {
            gpu1 = new GpuMatrix(sw1.Mat);
        }
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        gpu1?.Dispose();
    }

    [Benchmark(Baseline = true)]
    public MatrixBase Software() => sw1.GetTransposedMatrix();

    [Benchmark]
    public MatrixBase AVX() => avx1.GetTransposedMatrix();

    [Benchmark]
    public MatrixBase GPU()
    {
        if (!gpuAvailable) return avx1.GetTransposedMatrix();
        var r = gpu1.GetTransposedMatrix();
        (r as IDisposable)?.Dispose();
        return r;
    }
}

// ============================================================
// Convolution Benchmarks (kept from original)
// ============================================================

public class ConvolutionBenchmarks
{
    private AvxMatrix m1;
    private Matrix2D nm1;
    private Matrix2D filterMat2d;
    private AvxMatrix SquareKernel4;
    private AvxMatrix SquareKernel8;

    [GlobalSetup]
    public void Setup()
    {
        int matSize = 1024;
        int range = 100;
        int seed = 1253443;

        m1 = new AvxMatrix(matSize, matSize);
        m1.SetRandom(seed, -range, range);

        nm1 = new Matrix2D(matSize, matSize);
        nm1.SetRandom(seed, -range, range);

        SquareKernel4 = new AvxMatrix(4, 4);
        SquareKernel8 = new AvxMatrix(8, 8);
        SquareKernel4.SetRandom(seed, -range, range);
        SquareKernel8.SetRandom(seed, -range, range);

        filterMat2d = new Matrix2D(4, 4);
        filterMat2d.SetRandom(seed, -range, range);
    }

    [Benchmark]
    public Matrix2D NaiveConvolution4x4() => nm1.Convolution(filterMat2d);

    [Benchmark]
    public AvxMatrix AvxConvolution4x4() => m1.Convolution(SquareKernel4);

    [Benchmark]
    public AvxMatrix AvxConvolution8x8() => m1.Convolution(SquareKernel8);
}
