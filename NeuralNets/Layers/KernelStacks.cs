using System.Diagnostics;
using MatrixLibrary;
using MatrixLibrary.BaseClasses;

namespace NeuralNets
{
    public class KernelStacks
    {
        public int KernelCount { get; }
        public int KernelDepth { get; }
        public int KernelSize { get; }
        public List<List<MatrixBase>> Stacks { get; private set; }

        public KernelStacks(int kernelCount, int kernelDepth, int kernelSize)
        {
            KernelSize = kernelSize;
            KernelCount = kernelCount;
            KernelDepth = kernelDepth;
            Reset();
        }

        public KernelStacks(List<List<MatrixBase>> stacks)
        {
            Stacks = stacks;
            KernelCount = stacks.Count;
            KernelDepth = stacks[0].Count;
            KernelSize = stacks[0][0].Cols;
            Debug.Assert(stacks[0][0].Cols == stacks[0][0].Rows); // sanity check
        }

        public void Reset()
        {
            Stacks = new List<List<MatrixBase>>(KernelCount);
            for (int i = 0; i < KernelCount; i++)
            {
                Stacks.Add(new List<MatrixBase>(KernelDepth));
                for (int j = 0; j < KernelDepth; j++)
                {
                    Stacks[i].Add(MatrixFactory.CreateMatrix(KernelSize, KernelSize));
                }
            }
        }

        public MatrixBase this[int i, int j]
        {
            get => Stacks[i][j];
            set => Stacks[i][j] = value;
        }

        public List<MatrixBase> this[int i]
        {
            get => Stacks[i];
        }

        public void Accumulate( KernelStacks ks)
        {
            for (int i = 0; i < Stacks.Count; i++)
                for (int j = 0; j < Stacks[i].Count; j++)
                {
                    MatrixBase s1 = ks[i, j];
                    MatrixBase s2 = this[i, j];
                    this[i, j] = s1 + s2;
                }
        }

        public void ScaleAndAverage(int sampleCount, float learningRate)
        {
            float factor = learningRate / (float)sampleCount;
            for (int i = 0; i < Stacks.Count; i++)
                for (int j = 0; j < Stacks[i].Count; j++)
                {
                    MatrixBase s1 = this[i, j];
                    this[i, j] = (MatrixBase)(s1 * factor);
                }
        }

        internal void Subtract(KernelStacks ks)
        {
            for (int i = 0; i < Stacks.Count; i++)
                for (int j = 0; j < Stacks[i].Count; j++)
                {
                    MatrixBase rhs = ks[i, j];
                    MatrixBase lhs = this[i, j];
                    this[i, j] = (MatrixBase)(lhs - rhs);
                }
        }
    }
}
