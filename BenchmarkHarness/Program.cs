using MatrixLibrary;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Loggers;
using MatrixLibrary.BaseClasses;

public class SummaryOnlyLogger : ILogger
{
    private bool _inSummary;
    public string Id => nameof(SummaryOnlyLogger);
    public int Priority => 0;

    public void Write(LogKind logKind, string text)
    {
        if (text.Contains("// * Summary *"))
            _inSummary = true;
        if (_inSummary)
            Console.Write(text);
    }

    public void WriteLine()
    {
        if (_inSummary) Console.WriteLine();
    }

    public void WriteLine(LogKind logKind, string text)
    {
        if (text.Contains("// * Summary *"))
            _inSummary = true;
        if (_inSummary)
            Console.WriteLine(text);
        if (text.Contains("// * Legends *"))
            _inSummary = false;
    }

    public void Flush() { }
}

public class Program
{
    private static ManualConfig GetConfig()
    {
        var defaults = DefaultConfig.Instance;
        var config = ManualConfig.CreateEmpty();

        // Copy everything from defaults except loggers
        foreach (var job in defaults.GetJobs()) config.AddJob(job);
        foreach (var a in defaults.GetAnalysers()) config.AddAnalyser(a);
        foreach (var v in defaults.GetValidators()) config.AddValidator(v);
        foreach (var e in defaults.GetExporters()) config.AddExporter(e);
        foreach (var cp in defaults.GetColumnProviders()) config.AddColumnProvider(cp);

        // Use our summary-only logger instead of ConsoleLogger
        config.AddLogger(new SummaryOnlyLogger());
        config.WithOptions(ConfigOptions.DisableLogFile);

        return config;
    }

    public static void Main(string[] args)
    {
        var config = GetConfig();
        bool onlyDoChainBench = args.Length > 0 && args[0] == "--chain";

        if (true) //onlyDoChainBench)
        {
            BenchmarkRunner.Run<ChainingBenchmarks>(config);
        }
        else
        {
            BenchmarkRunner.Run<MultiplyBenchmarks>(config);
            BenchmarkRunner.Run<AddBenchmarks>(config);
            BenchmarkRunner.Run<SubtractBenchmarks>(config);
            BenchmarkRunner.Run<TransposeBenchmarks>(config);
            BenchmarkRunner.Run<ConvolutionBenchmarks>(config);
            if (BackendSelector.IsGPUAvailable())
                BenchmarkRunner.Run<ChainingBenchmarks>(config);
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

// ============================================================
// Chaining Benchmarks: chained GPU ops vs non-chained (forced round-trip)
// ============================================================

/// <summary>
/// Benchmarks comparing chained GPU operations (data stays on device between ops)
/// vs non-chained (force host round-trip between each operation).
///
/// Chained: result GpuMatrix from op1 is passed directly to op2. The device pointer
/// is already valid, so EnsureDeviceUpToDate() skips the upload.
///
/// NonChained: after each op, we create a new GpuMatrix from the host Mat of the result.
/// This discards the device pointer and forces a fresh upload on the next op.
/// </summary>
[GroupBenchmarksBy(BenchmarkLogicalGroupRule.ByCategory)]
[BenchmarkCategory("Chaining")]
public class ChainingBenchmarks
{
    private GpuMatrix g1, g2, g3;

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
        g3 = new GpuMatrix(matSize, matSize);
        g1.SetRandom(seed, -range, range);
        g2.SetRandom(seed + 1, -range, range);
        g3.SetRandom(seed + 2, -range, range);

        // Warm up device pointers so first-run upload cost isn't in the benchmark
        var warmup = g1.Add(g2);
        (warmup as IDisposable)?.Dispose();
        warmup = g2.Add(g3);
        (warmup as IDisposable)?.Dispose();
        warmup = g3.Add(g1);
        (warmup as IDisposable)?.Dispose();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        g1?.Dispose();
        g2?.Dispose();
        g3?.Dispose();
    }

    /// <summary>
    /// Forces a host round-trip: creates a new GpuMatrix from host data,
    /// discarding the device pointer and forcing re-upload on next use.
    /// </summary>
    private GpuMatrix ForceHostRoundTrip(MatrixBase result)
    {
        // Access .Mat to ensure host is populated, then create new GpuMatrix
        // from that host data. The new instance has _deviceValid=false,
        // so the next operation will have to upload again.
        var hostData = result.Mat;
        (result as IDisposable)?.Dispose(); // free the old device memory
        return new GpuMatrix(hostData);
    }

    // ============================================================
    // Add chain: A + B + C  (two additions)
    // ============================================================

    [Benchmark]
    [BenchmarkCategory("Add3")]
    public void AddChain_Chained()
    {
        var ab = (GpuMatrix)g1.Add(g2);
        var abc = ab.Add(g3);
        (ab as IDisposable)?.Dispose();
        (abc as IDisposable)?.Dispose();
    }

    [Benchmark(Baseline = true)]
    [BenchmarkCategory("Add3")]
    public void AddChain_NonChained()
    {
        var ab = ForceHostRoundTrip(g1.Add(g2));
        var abc = ab.Add(g3);
        (ab as IDisposable)?.Dispose();
        (abc as IDisposable)?.Dispose();
    }

    // ============================================================
    // Multiply chain: (A * B) * C  (two multiplications)
    // ============================================================

    [Benchmark]
    [BenchmarkCategory("Mul3")]
    public void MulChain_Chained()
    {
        var ab = (GpuMatrix)g1.Multiply(g2);
        var abc = ab.Multiply(g3);
        (ab as IDisposable)?.Dispose();
        (abc as IDisposable)?.Dispose();
    }

    [Benchmark(Baseline = true)]
    [BenchmarkCategory("Mul3")]
    public void MulChain_NonChained()
    {
        var ab = ForceHostRoundTrip(g1.Multiply(g2));
        var abc = ab.Multiply(g3);
        (ab as IDisposable)?.Dispose();
        (abc as IDisposable)?.Dispose();
    }

    // ============================================================
    // Mixed chain: (A + B) * C  (add then multiply)
    // ============================================================

    [Benchmark]
    [BenchmarkCategory("AddThenMul")]
    public void AddThenMul_Chained()
    {
        var ab = (GpuMatrix)g1.Add(g2);
        var result = ab.Multiply(g3);
        (ab as IDisposable)?.Dispose();
        (result as IDisposable)?.Dispose();
    }

    [Benchmark(Baseline = true)]
    [BenchmarkCategory("AddThenMul")]
    public void AddThenMul_NonChained()
    {
        var ab = ForceHostRoundTrip(g1.Add(g2));
        var result = ab.Multiply(g3);
        (ab as IDisposable)?.Dispose();
        (result as IDisposable)?.Dispose();
    }

    // ============================================================
    // Long chain: A + B - C + scalar * A  (four ops)
    // ============================================================

    [Benchmark]
    [BenchmarkCategory("LongChain")]
    public void LongChain_Chained()
    {
        var step1 = (GpuMatrix)g1.Add(g2);
        var step2 = (GpuMatrix)step1.Subtract(g3);
        var step3 = (GpuMatrix)g1.Multiply(2.5f);
        var step4 = step2.Add(step3);
        (step1 as IDisposable)?.Dispose();
        (step2 as IDisposable)?.Dispose();
        (step3 as IDisposable)?.Dispose();
        (step4 as IDisposable)?.Dispose();
    }

    [Benchmark(Baseline = true)]
    [BenchmarkCategory("LongChain")]
    public void LongChain_NonChained()
    {
        var step1 = ForceHostRoundTrip(g1.Add(g2));
        var step2 = ForceHostRoundTrip(step1.Subtract(g3));
        var step3 = ForceHostRoundTrip(g1.Multiply(2.5f));
        var step4 = step2.Add(step3);
        (step1 as IDisposable)?.Dispose();
        (step2 as IDisposable)?.Dispose();
        (step3 as IDisposable)?.Dispose();
        (step4 as IDisposable)?.Dispose();
    }

    // ============================================================
    // Transpose then multiply: A^T * B  (transpose + multiply)
    // ============================================================

    [Benchmark]
    [BenchmarkCategory("TransposeMul")]
    public void TransposeThenMul_Chained()
    {
        var aT = (GpuMatrix)g1.GetTransposedMatrix();
        var result = aT.Multiply(g2);
        (aT as IDisposable)?.Dispose();
        (result as IDisposable)?.Dispose();
    }

    [Benchmark(Baseline = true)]
    [BenchmarkCategory("TransposeMul")]
    public void TransposeThenMul_NonChained()
    {
        var aT = ForceHostRoundTrip(g1.GetTransposedMatrix());
        var result = aT.Multiply(g2);
        (aT as IDisposable)?.Dispose();
        (result as IDisposable)?.Dispose();
    }
}
