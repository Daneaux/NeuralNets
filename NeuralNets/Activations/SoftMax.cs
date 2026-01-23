using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using System.Diagnostics;

namespace NeuralNets
{
    public class SoftMax : IActivationFunction
    {
        public Tensor LastActivation {  get; private set; }

        public ColumnVectorBase Activate(ColumnVectorBase input)
        {
            float max = input.GetMax();
            float scaleFactor = SumExpEMinusMax(max, input);
            float[] softMaxVec = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
                softMaxVec[i] = (float)(Math.Exp(input[i] - max) / scaleFactor);

            ColumnVectorBase la = MatrixFactory.CreateColumnVector(softMaxVec);
            LastActivation = la.ToTensor();
            return la;
        }

        public MatrixBase Activate(MatrixBase input)
        {
            throw new NotImplementedException();
        }

        public List<MatrixBase> Activate(List<MatrixBase> input)
        {
            throw new NotImplementedException();
        }

        // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
        // https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
        public ColumnVectorBase Derivative(ColumnVectorBase lastActivation)
        {
            Debug.Assert(lastActivation != null);
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");
        }
        private static float SumExpEMinusMax(float max, ColumnVectorBase vec)
        {
            float scale = 0;
            for (int i = 0; i < vec.Size; i++)
            {
                scale += (float)Math.Exp(vec[i] - max);
            }
            return scale;
        }
    }
}