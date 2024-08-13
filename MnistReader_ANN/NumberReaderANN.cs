using NeuralNets;
using System.Diagnostics;

namespace NumReaderNetwork
{
    public class NumberReaderANN
    {
        public double TrainingRate { get; }
        public int InputDim { get; private set; }
        public int HiddenLayerDim { get; private set; }
        public int OutputLayerDim { get; private set; }
        //public InputLayer InputLayer { get; private set; }
        public List<WeightedLayer> WeightedLayers { get;  set; }
        public ILossFunction LossFunction { get; private set; }

        public NumberReaderANN(int inputDim)//, int hiddenLayerDim, int outputLayerDim)
        {
            this.TrainingRate = 0.01;
            this.InputDim = inputDim;
            this.OutputLayerDim = 10;
            this.HiddenLayerDim = 16;
            this.LossFunction = new SquaredLoss();
        }

        public void InitializeNetwork()
        {
           // InputLayer = new InputLayer(InputDim);
            WeightedLayers.Add(new WeightedLayer(HiddenLayerDim, new ReLUActivaction(), InputDim));
            WeightedLayers.Add(new WeightedLayer(OutputLayerDim, new SigmoidActivation(), HiddenLayerDim));
        }

        public ColumnVector FeedForward(ColumnVector inputVec)
        {
            Debug.Assert(inputVec.Size == this.InputDim);
            Debug.Assert(WeightedLayers.Count == 2);
            ColumnVector currentActivation = inputVec;
            //InputLayer = new InputLayer(currentActivation); // doesn't see necessary at all?
            for (int i = 0; i < 2; i++)
            {
                WeightedLayer currentLayer = WeightedLayers[i];
                ColumnVector z1 = currentLayer.Weights * currentActivation;
                ColumnVector z12 = z1 - currentLayer.Biases;
                ColumnVector o1 = currentLayer.ActivationFunction.Activate(z12);
                currentActivation = o1;
            }
            return currentActivation;
        }

        public void BackProp(TrainingPair trainingPair)
        {
            // First do a feed forward
            ColumnVector predictedOut = this.FeedForward(trainingPair.Input);

            
            // Second: find W0 - Wn in the hidden layer, just before the output layer
            // We want the derivative of the Error function in terms of the weights (w0 ... wn)
            // d(E)/dw1 = d(E)/o2 * d(o2)/z2 * d(z2)/w = (a-b) * layer.derivative * o1
            // <matrix form> ==> 
            //          (pred - actual)_vec * sigmoid_derivate_vec * layer-1.output_vec
            // d(E)/db = d(E)/o2 * d(o2)/z2 * d(z2)/b
            // 

            WeightedLayer hiddenLayer = WeightedLayers[0];
            WeightedLayer outputLayer = WeightedLayers[1];

            // partial product, before we start the per-w differentials.
            ColumnVector w2_sigma =
                this.LossFunction.Derivative(predictedOut, trainingPair.Output) *                               // pred - actual
                outputLayer.ActivationFunction.Derivative(outputLayer.ActivationFunction.LastActivation);       // sigmoid derivative

            // Remember that the weights in the weight matrix are ROWS ...
            // so the dot product of row1 and output vector or activation vector minus the bias is = Z (the input to the activation function)

            // so the gradient w' matrix needs to be rows of gradient weights (or weight deltas) that we get from all the partial derivative shenanigans
            int numCols = hiddenLayer.ActivationFunction.LastActivation.Size;
            ColumnVector[] w2_delta = new ColumnVector[numCols];
            for (int i = 0; i < numCols; i++)
            {
                double o2 = hiddenLayer.ActivationFunction.LastActivation[i];
                w2_delta[i] = this.TrainingRate * w2_sigma * o2;                                // d(z2)/w
            }

            // Now turn that array of columns into a matrix and transpose
            Matrix scaledGradientWeights = new Matrix(w2_delta, w2_delta[0].Size);
            scaledGradientWeights = scaledGradientWeights.GetTransposedMatrix();

            ColumnVector b2_delta = this.TrainingRate * w2_sigma * 1.0;  // d(z2)/b  .. i think?


            // WARNING: UPDATE THESE WEIGHTS AFTER BACK PROP IS DONE!
            // Now: Update W2 weight matrix with w2_delta (and same for b)
            outputLayer.UpdateWeights(scaledGradientWeights);
            outputLayer.UpdateBiases(b2_delta);

            // Now, let's do the weights left of the hidden layer!
            // d(e)/dw1 = d(e)/o1 * d(o1)/z1 * d(z1)/w
            // but d(e)/o1 influences (or influences by?) all the incoming edges to this node. So have to sum all their partials wrt w 
            // So:  d(e)/o1 = sum of all: d(error O2 node1)/o1 + d(error o2 node2)/o1
            // and: d(error o2 node1)/o1 = d(error o2)/z2 * d(z2)/o1
            // because d(z2)/o1 ==> z2 = w1 * o11 + w2 * o12 - b.  So Z2/o11 --> w1
        }
    }
}
