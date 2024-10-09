using MatrixLibrary;
using MatrixLibrary.Avx;
using System.Diagnostics;

namespace NeuralNets
{
    public class PoolingLayer : Layer
    {
        public PoolingLayer(
            InputOutputShape inputShape,
            int stride,
            int kernelCount,
            int kernelSquareDimension,
            int kernelDepth
            ) : base(inputShape, kernelCount, null)
        {
            Stride = stride;
            KernelCount = kernelCount;
            KernelDepth = kernelDepth;
            KernelSize = kernelSquareDimension;
            FlatOutputSize = kernelDepth * KernelSize * KernelSize * KernelCount;
            (int destRows, int destColumns) = AvxMatrix.ConvolutionSizeHelper(inputShape, KernelSize, isFull:false, Stride);

            OutputShape = new InputOutputShape(destColumns, destRows, KernelDepth, KernelCount);
        }

        public int Stride { get; }
        public int KernelCount { get; }
        public int KernelDepth { get; }
        public int KernelSize { get; }
        public int FlatOutputSize { get; }
        public override InputOutputShape OutputShape { get; }

        public override Tensor FeedFoward(Tensor input)
        {
            List<AvxMatrix> pooledMatrices = new List<AvxMatrix>();

            foreach(AvxMatrix mat in input.Matrices)
            {
                AvxMatrix pooledMat = DoMaxPool(mat, Stride, KernelSize);
                pooledMatrices.Add(pooledMat);
            }

            ConvolutionTensor convolutionTensor = new ConvolutionTensor(pooledMatrices);
            return convolutionTensor;
        }

        private AvxMatrix DoMaxPool(AvxMatrix mat, int stride, int filterSize)
        {
            Debug.Assert(stride >= 1);

            (int destRows, int destColumns) = AvxMatrix.ConvolutionSizeHelper(mat, filterSize, isFull: false, stride);

            AvxMatrix result = new AvxMatrix(destRows, destColumns);
            for(int r = 0, dr = 0; dr < destRows; r += stride, dr++)
            {
                for(int c = 0, dc = 0; dc < destColumns; c += stride, dc++)
                {
                    float maxSample = DoMaxSample(mat, r, c, filterSize);
                    result[dr, dc] = maxSample;
                }
            }

            return result;
        }

        private float DoMaxSample(AvxMatrix mat, int r, int c, int filterSize)
        {
            float max = float.MinValue;
            for(int i = r; i < r + filterSize; i++)
            {
                for(int j = c; j < c + filterSize; j++)
                {
                    if (mat[i, j] > max)
                        max = mat[i, j];
                }
            }
            return max;
        }

        public override void UpdateWeightsAndBiasesWithScaledGradients(Tensor weightGradient, Tensor biasGradient)
        {
            throw new NotImplementedException();
        }
    }
}
