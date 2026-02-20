using System.Diagnostics;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace MatrixLibraryTests.Avx
{
    /// <summary>
    /// Parallel.For test using AVX matrix multiplication to measure CPU utilization across 8 cores.
    /// </summary>
    [TestClass]
    [Ignore("This test is designed for manual execution to observe CPU utilization. Run it in a controlled environment and monitor Task Manager or Performance Monitor to see the effect on CPU usage.")]
    public class ParallelAvxMatrixTest
    {
        private const int MatrixSize = 512;
        private const int IterationsPerCore = 100;
        private const int CoreCount = 16;

        [TestMethod]
        public  void AvxParallelForPerfCompare()
        {
            Console.WriteLine("=".PadRight(70, '='));
            Console.WriteLine("Parallel.For AVX Matrix Multiplication CPU Utilization Test");
            Console.WriteLine("=".PadRight(70, '='));
            Console.WriteLine($"Matrix Size: {MatrixSize}x{MatrixSize}");
            Console.WriteLine($"Cores: {CoreCount}");
            Console.WriteLine($"Iterations per core: {IterationsPerCore}");
            Console.WriteLine($"Total matrix multiplications: {CoreCount * IterationsPerCore}");
            Console.WriteLine();


            // Prepare matrices
            Console.WriteLine("Initializing test matrices...");
            var random = new Random(42);
            var matricesA = new List<AvxMatrix>();
            var matricesB = new List<AvxMatrix>();

            
            for (int i = 0; i < CoreCount; i++)
            {
                float[,] matA = new float[MatrixSize, MatrixSize];
                float[,] matB = new float[MatrixSize, MatrixSize];

                for (int r = 0; r < MatrixSize; r++)
                {
                    for (int c = 0; c < MatrixSize; c++)
                    {
                        matA[r, c] = (float)(random.NextDouble() * 2.0 - 1.0);
                        matB[r, c] = (float)(random.NextDouble() * 2.0 - 1.0);
                    }
                }

                matricesA.Add(new AvxMatrix(matA));
                matricesB.Add(new AvxMatrix(matB));
            }

            // Warm up
            Console.WriteLine("Warming up...");
            for (int i = 0; i < 5; i++)
            {
               // var warmup = matricesA[0].Multiply(matricesB[0]);
            }
            

            Console.WriteLine();
            Console.WriteLine("Starting parallel matrix multiplication test...");
            Console.WriteLine("Watch your CPU utilization in Task Manager/Performance Monitor.");
            Console.WriteLine();

            // Track results to prevent optimization
            var results = new AvxMatrix[CoreCount];
            long[] operationCounts = new long[CoreCount];

            var stopwatch = Stopwatch.StartNew();
            var process = Process.GetCurrentProcess();
            var startTime = process.TotalProcessorTime;

            // Run parallel matrix multiplications
            Parallel.For(0, CoreCount, new ParallelOptions { MaxDegreeOfParallelism = CoreCount }, coreId =>
            {
                var localStopwatch = Stopwatch.StartNew();
                var coreMatricesA = matricesA[coreId];
                var coreMatricesB = matricesB[coreId];

                for (int iter = 0; iter < IterationsPerCore; iter++)
                {
                    // Perform AVX matrix multiplication
                    results[coreId] = coreMatricesA.Multiply(coreMatricesB);
                    operationCounts[coreId]++;

                    // Swap A and B to keep data moving through cache
                    var temp = coreMatricesA;
                    coreMatricesA = coreMatricesB;
                    coreMatricesB = results[coreId];
                }

                localStopwatch.Stop();
                Console.WriteLine($"Core {coreId}: Completed {operationCounts[coreId]} multiplications in {localStopwatch.ElapsedMilliseconds}ms");
            });

            stopwatch.Stop();
            var endTime = process.TotalProcessorTime;
            var cpuTimeUsed = endTime - startTime;

            // Calculate statistics
            long totalOperations = operationCounts.Sum();
            double totalSeconds = stopwatch.Elapsed.TotalSeconds;
            double operationsPerSecond = totalOperations / totalSeconds;
            double cpuUtilization = (cpuTimeUsed.TotalSeconds / (totalSeconds * CoreCount)) * 100;

            // Verify results (just to prevent optimization)
            float checksum = 0;
            foreach (var result in results)
            {
                if (result != null)
                {
                    checksum += result[0, 0] + result[MatrixSize - 1, MatrixSize - 1];
                }
            }

            Console.WriteLine();
            Console.WriteLine("=".PadRight(70, '='));
            Console.WriteLine("RESULTS");
            Console.WriteLine("=".PadRight(70, '='));
            Console.WriteLine($"Total elapsed time: {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine($"Total CPU time: {cpuTimeUsed.TotalMilliseconds:F0}ms");
            Console.WriteLine($"Total matrix multiplications: {totalOperations}");
            Console.WriteLine($"Multiplications per second: {operationsPerSecond:F2}");
            Console.WriteLine($"CPU utilization: {cpuUtilization:F1}% (across {CoreCount} cores)");
            Console.WriteLine($"Checksum (to prevent optimization): {checksum}");
            Console.WriteLine();

            // Sequential comparison
            Console.WriteLine("Running sequential comparison...");
            var seqStopwatch = Stopwatch.StartNew();
            for (int i = 0; i < CoreCount * IterationsPerCore; i++)
            {
                var result = matricesA[0].Multiply(matricesB[0]);
            }
            seqStopwatch.Stop();

            double speedup = (double)seqStopwatch.ElapsedMilliseconds / stopwatch.ElapsedMilliseconds;
            Console.WriteLine($"Sequential time: {seqStopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine($"Parallel speedup: {speedup:F2}x");
            Console.WriteLine($"Efficiency: {(speedup / CoreCount) * 100:F1}%");
            Console.WriteLine();


            const int floatsPerVector = 16;
            long totalFloatOperations = (long)MatrixSize * MatrixSize * MatrixSize * 2 * totalOperations; // multiply-add pairs
            long vectorOperations = totalFloatOperations / floatsPerVector;
            double gflops = totalFloatOperations / (stopwatch.Elapsed.TotalSeconds * 1e9);

            Console.WriteLine($"Total floating-point operations: {totalFloatOperations:N0}");
            Console.WriteLine($"Estimated vector operations: {vectorOperations:N0}");
            Console.WriteLine($"Throughput: {gflops:F2} GFLOPS");         

            Console.WriteLine();
            Console.WriteLine("Test complete. Check Task Manager to see if all cores were utilized.");
        }
    }
}
