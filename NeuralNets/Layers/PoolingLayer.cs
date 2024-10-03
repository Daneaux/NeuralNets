using MatrixLibrary;
using MatrixLibrary.Avx;
using System.Diagnostics;

namespace NeuralNets
{
    public class PoolingLayer : Layer
    {
        public PoolingLayer(
            int stride,
            int kernelCount,
            int kernelSquareDimension,
            int kernelDepth
            ) : base(kernelCount, null, kernelDepth, 0)
        {
            Stride = stride;
            KernelCount = kernelCount;
            KernelDepth = kernelDepth;
            FilterSize = kernelSquareDimension;
        }

        public int Stride { get; }
        public int KernelCount { get; }
        public int KernelDepth { get; }
        public int FilterSize { get; }

        public override Tensor FeedFoward(Tensor input)
        {
            List<AvxMatrix> pooledMatrices = new List<AvxMatrix>();

            ConvolutionTensor inT = input as ConvolutionTensor;
            if(inT != null) 
                throw new ArgumentException("Expected a convolutionTensor");

            foreach(AvxMatrix mat in inT.Matrices)
            {
                AvxMatrix pooledMat = DoMaxPool(mat, Stride, FilterSize);
                pooledMatrices.Add(pooledMat);
            }

            ConvolutionTensor convolutionTensor = new ConvolutionTensor(pooledMatrices);
            return convolutionTensor;
        }

        private AvxMatrix DoMaxPool(AvxMatrix mat, int stride, int filterSize)
        {
            Debug.Assert(stride >= 1);

            (int destRows, int destColumns) = AvxMatrix.ConvolutionSizeHelper(mat, filterSize, stride);

            AvxMatrix result = new AvxMatrix(destRows, destColumns);
            for(int r = 0, dr = 0; dr < destRows; r += stride, dr++)
            {
                for(int c = 0, dc = 0; dc < destColumns; c += stride, dc++)
                {
                    float maxSample = DoMaxSample(mat, r, c, filterSize);
                    result[r,c] = maxSample;
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
