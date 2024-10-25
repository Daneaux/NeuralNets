using MatrixLibrary;
using System.Diagnostics;

namespace NeuralNets
{
    public class SoftMax : IActivationFunction
    {
        public Tensor LastActivation {  get; private set; }

        public AvxColumnVector Activate(AvxColumnVector input)
        {
            float max = input.GetMax();
            float scaleFactor = SumExpEMinusMax(max, input);
            float[] softMaxVec = new float[input.Size];
            for (int i = 0; i < input.Size; i++)
                softMaxVec[i] = (float)(Math.Exp(input[i] - max) / scaleFactor);

            AvxColumnVector la = new AvxColumnVector(softMaxVec);
            LastActivation = la.ToTensor();
            return la;
        }

        public AvxMatrix Activate(AvxMatrix input)
        {
            throw new NotImplementedException();
        }

        public List<AvxMatrix> Activate(List<AvxMatrix> input)
        {
            throw new NotImplementedException();
        }

        // https://www.mldawn.com/the-derivative-of-softmaxz-function-w-r-t-z/
        // https://stats.stackexchange.com/questions/453539/softmax-derivative-implementation
        public AvxColumnVector Derivative(AvxColumnVector lastActivation)
        {
            Debug.Assert(lastActivation != null);
            throw new InvalidOperationException("Don't call derivative on softmax, just use the softmax*crossentropy derivative which is a-y'");
        }
        private static float SumExpEMinusMax(float max, AvxColumnVector vec)
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