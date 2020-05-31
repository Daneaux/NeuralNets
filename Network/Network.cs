using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    class Network
    {
        public int Seed { get; }
        public List<Layer> Layers { get; }
        public ILossFunction LossFunction { get; }

        public Network(List<Layer> layers, ILossFunction lossFunction)
        {
            Seed = 123;  // TODO: temp
            Layers = layers;
            LossFunction = lossFunction;
            InitializeWeightMatricies();
        }

        public ColumnVector FeedForward(double[] inputVector)
        {
            ColumnVector activationVector = new ColumnVector(inputVector);
            for(int i = 0; i < this.weightMatrices.Count; i++)
            {
                Matrix weightMatrix = this.weightMatrices[i];
                ColumnVector weightedSum = weightMatrix * activationVector;
                activationVector = DoActivation(weightedSum, this.Layers[i]);
            }

            return activationVector;
        }

        private List<Matrix> weightMatrices = new List<Matrix>();
        private void InitializeWeightMatricies()
        {
            // Number of matrices is num layers - 1.  So if we have 1 input, 1 hidden, and 1 output, numMatrices = 2

            // first weight matrix, if inputs is 4 for example, and next layer is 20
            // then every input has 20 weighted edges to layer1 (layer0 is theinput layer)
            // so the top row in this matrix is 4 wide, it represents all the weights (edges)
            // incoming to the FIRST node in the SECOND layer, ie: the first output!
            // Hence the number of columns is equal to the number of inputs.
            // So, the number of Rows corresponds to the number of nodes in the next layer,
            // in other words, the outputs (before activation)
            //// ---layer0_1 = new Matrix(layers[1].NumNodes, layers[0].NumNodes);

            // ok layer 1 to 2
            // input layer is layer 1, output layer is layer 2
            // so, the Matrix rows = l2  and matrix columns = l1
            // example:  layer 1 has 20 nodes, and layer 2 has 35 nodes
            // so each output node has 20 incoming edges.  
            // O[1] = weighted sume of inputs 1 --- 20 = input1 * w11 + input2 * w21 + input3 * w31 ... inputN * w1;
            //// ----layer1_2 = new Matrix(layers[2].NumNodes, layers[1].NumNodes);

            int numLayers = this.Layers.Count;
            int matrixCount = numLayers - 1;
            for (int i = 0; i < matrixCount; i++)
            {
                Matrix mat = new Matrix(Layers[i + 1].NumNodes, Layers[i].NumNodes);
                mat.SetRandom(Seed, 0.0, 1.0); // TODO: add he et al initialization, Xavier Initialization (each based on different activation functions)
                this.weightMatrices.Add(mat);
            }
        }

        private ColumnVector DoActivation(ColumnVector inputColumnVector, Layer layer)
        {
            Debug.Assert(inputColumnVector.Size == layer.NumNodes);

            double[] vector = new double[inputColumnVector.Size];
            for(int i = 0; i < inputColumnVector.Size; i++)
            {
                vector[i] = layer.ActivationFunction.Activate(inputColumnVector[i]);
            }
            return new ColumnVector(vector);
        }
    }
}
