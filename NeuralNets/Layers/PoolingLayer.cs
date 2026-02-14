using MatrixLibrary;
using System.Diagnostics;
using MatrixLibrary.BaseClasses;

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
            ) : base(inputShape, kernelCount)
        {
            Stride = stride;
            KernelCount = kernelCount;
            KernelDepth = kernelDepth;
            KernelSize = kernelSquareDimension;
            FlatOutputSize = kernelDepth * KernelSize * KernelSize * KernelCount;
            (int destRows, int destColumns) = MatrixLibrary.MatrixHelpers.ConvolutionSizeHelper(inputShape, KernelSize, isFull:false, Stride);

            OutputShape = new InputOutputShape(destColumns, destRows, KernelDepth, KernelCount);
        }

        public int Stride { get; }
        public int KernelCount { get; }
        public int KernelDepth { get; }
        public int KernelSize { get; }
        public int FlatOutputSize { get; }
        public override InputOutputShape OutputShape { get; }

        private List<GradientRouter> GradientRouters;
        private struct GradientRouter
        {
            internal int originalRows;
            internal int originalColumns;
            internal bool[,] maxPoolHistogram;
        }

        public override Tensor FeedFoward(Tensor input)
        {
            GradientRouters = new List<GradientRouter>();
            List<MatrixBase> pooledMatrices = new List<MatrixBase>();
            foreach(MatrixBase mat in input.Matrices)
            {
                GradientRouter gradientRouter = new GradientRouter();
                MatrixBase pooledMat = DoMaxPool(mat, Stride, KernelSize, out gradientRouter);
                pooledMatrices.Add(pooledMat);
                GradientRouters.Add(gradientRouter);
            }

            ConvolutionTensor convolutionTensor = new ConvolutionTensor(pooledMatrices);
            return convolutionTensor;
        }

        private MatrixBase DoMaxPool(MatrixBase mat, int stride, int filterSize, out GradientRouter gradientRouter)
        {
            Debug.Assert(stride >= 1);

            (int destRows, int destColumns) = MatrixLibrary.MatrixHelpers.ConvolutionSizeHelper(mat, filterSize, isFull: false, stride);
            gradientRouter.maxPoolHistogram = new bool[mat.Rows, mat.Cols];
            gradientRouter.originalRows = mat.Rows;
            gradientRouter.originalColumns = mat.Cols;

            MatrixBase result = MatrixFactory.CreateMatrix(destRows, destColumns);
            for(int r = 0, dr = 0; dr < destRows; r += stride, dr++)
            {
                for(int c = 0, dc = 0; dc < destColumns; c += stride, dc++)
                {
                    float maxSample = DoMaxSample(mat, r, c, filterSize, gradientRouter);
                    result[dr, dc] = maxSample;
                }
            }

            return result;
        }

        private float DoMaxSample(MatrixBase mat, int r, int c, int filterSize, GradientRouter gradientRouter) 
        {
            int maxR =0, maxC=0;
            float max = float.MinValue;
            for(int i = r; i < r + filterSize; i++)
            {
                for(int j = c; j < c + filterSize; j++)
                {
                    if (mat[i, j] > max)
                    {
                        maxR = i;
                        maxC = j;
                        max = mat[i, j];
                    }
                }
            }

            // remember which element was max
            gradientRouter.maxPoolHistogram[maxR, maxC] = true;
            return max;
        }

        public override Tensor BackPropagation(Tensor dE_dX)
        {
            List<MatrixBase> incomingGradients;
            if (dE_dX.ToColumnVector() != null)
            {
                // unflatten the incoming gradients back to a series of matrices (that we orignally flattened to pass on to a dense layer)
                incomingGradients = UnflattenIncomingVector(dE_dX.ToColumnVector());
            }
            else
            {
                incomingGradients = dE_dX.Matrices;
            }

            List<MatrixBase> outgoingGradients = new List<MatrixBase>();
            Debug.Assert(incomingGradients.Count == GradientRouters.Count);
            for(int i = 0;  i < incomingGradients.Count; i++)
            {
                int incomingCount = 0;
                MatrixBase incomingGradient = incomingGradients[i];
                GradientRouter gradientRouter = GradientRouters[i];
                // todo: how to assert assumption that gradient order corresponds to previous output order (and associated gradientrouter)??
                MatrixBase gradient = MatrixFactory.CreateMatrix(gradientRouter.originalRows, gradientRouter.originalColumns);
                // only pass up gradients to the max elemenet in the previously pooled window. All other gradients are zero.
                for (int r = 0; r < gradientRouter.originalRows; r++)
                {
                    for (int c = 0; c < gradientRouter.originalColumns; c++)
                    {
                        if (gradientRouter.maxPoolHistogram[r, c] == true)
                        {
                            (int ir, int ic) = TwoDIndexHelper(incomingCount, incomingGradient.Rows, incomingGradient.Cols);
                            gradient[r, c] = incomingGradient[ir, ic];
                            incomingCount++;
                        }
                    }
                }
                outgoingGradients.Add(gradient);
            }

            return outgoingGradients.ToTensor();
        }

        // there has to be a better way than copying. todo.
        private List<MatrixBase> UnflattenIncomingVector(ColumnVectorBase? ColumnVectorBase)
        {
            List<MatrixBase> mats = new List<MatrixBase>();
            int rows = GradientRouters[0].originalRows;
            int cols = GradientRouters[0].originalColumns;
            int numMatrices = ColumnVectorBase.Size / (rows * cols);
            Debug.Assert(ColumnVectorBase.Size % (rows * cols) == 0 );
            for (int i = 0, j = 0; i < numMatrices; i++)
            {
                mats.Add(MatrixFactory.CreateMatrix(rows, cols));
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        mats[i][r, c] = ColumnVectorBase[j++];
            }

            return mats;
        }

        private (int r, int c) TwoDIndexHelper(int flatIndex, int rows, int cols)
        {
            int r = flatIndex / cols;
            int c = flatIndex % cols;
            return (r, c);
        }
    }
}
