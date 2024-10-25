using MatrixLibrary.Avx;
using System.Diagnostics;

namespace NeuralNets
{
    public class KernelStacks
    {
        public int KernelCount { get; }
        public int KernelDepth { get; }
        public int KernelSize { get; }
        public List<List<SquareKernel>> Stacks { get; private set; }

        public KernelStacks(int kernelCount, int kernelDepth, int kernelSize)
        {
            KernelSize = kernelSize;
            KernelCount = kernelCount;
            KernelDepth = kernelDepth;
            Reset();
        }

        public KernelStacks(List<List<SquareKernel>> stacks)
        {
            Stacks = stacks;
            KernelCount = stacks.Count;
            KernelDepth = stacks[0].Count;
            KernelSize = stacks[0][0].Cols;
            Debug.Assert(stacks[0][0].Cols == stacks[0][0].Rows); // sanity check
        }

        public void Reset()
        {
            Stacks = new List<List<SquareKernel>>(KernelCount);
            for (int i = 0; i < KernelCount; i++)
            {
                Stacks.Add(new List<SquareKernel>(KernelDepth));
                for (int j = 0; j < KernelDepth; j++)
                {
                    Stacks[i].Add(new SquareKernel(KernelSize));
                }
            }
        }

        public SquareKernel this[int i, int j]
        {
            get => Stacks[i][j];
            set => Stacks[i][j] = value;
        }

        public List<SquareKernel> this[int i]
        {
            get => Stacks[i];
        }

        public void Accumulate( KernelStacks ks)
        {
            for (int i = 0; i < Stacks.Count; i++)
                for (int j = 0; j < Stacks[i].Count; j++)
                {
                    SquareKernel s1 = ks[i, j];
                    SquareKernel s2 = this[i, j];
                    this[i, j] = s1 + s2;
                }
        }

        public void ScaleAndAverage(int sampleCount, float learningRate)
        {
            float factor = learningRate / (float)sampleCount;
            for (int i = 0; i < Stacks.Count; i++)
                for (int j = 0; j < Stacks[i].Count; j++)
                {
                    SquareKernel s1 = this[i, j];
                    this[i, j] = (SquareKernel)(s1 * factor);
                }
        }

        internal void Subtract(KernelStacks ks)
        {
            for (int i = 0; i < Stacks.Count; i++)
                for (int j = 0; j < Stacks[i].Count; j++)
                {
                    SquareKernel rhs = ks[i, j];
                    SquareKernel lhs = this[i, j];
                    this[i, j] = (SquareKernel)(lhs - rhs);
                }
        }
    }
}
