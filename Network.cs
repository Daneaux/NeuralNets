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
        public List<Layer> Layers { get; private set; }
        public Network(List<Layer> layers, ILossFunction lossFunction)
        {
            Layers = layers;

            // first weight matrix, if inputs is 4 for example, and next layer is 20
            // then every input has 20 weighted edges to layer1 (layer0 is theinput layer)
            // so the top row in this matrix is 4 wide, it represents all the weights (edges)
            // incoming to the FIRST node in the SECOND layer, ie: the first output!
            // Hence the number of columns is equal to the number of inputs.
            // So, the number of Rows corresponds to the number of nodes in the next layer,
            // in other words, the outputs (before activation)
            layer0_1 = new Matrix(layers[1].NumNodes, layers[0].NumNodes);

            // ok layer 1 to 2
            // input layer is layer 1, output layer is layer 2
            // so, the Matrix rows = l2  and matrix columns = l1
            // example:  layer 1 has 20 nodes, and layer 2 has 35 nodes
            // so each output node has 20 incoming edges.  
            // O[1] = weighted sume of inputs 1 --- 20 = input1 * w11 + input2 * w21 + input3 * w31 ... inputN * w1;
            layer1_2 = new Matrix(layers[2].NumNodes, layers[1].NumNodes);

        }

        // private temp for 3 layers total.  input, middle, output
        private Matrix layer0_1;
        private Matrix layer1_2;

        public double[] RunForward(double[] inputVector)
        {
            // input vector * [w1, w2, w3, ... wn] = [activationInput1, ai2, ai3, ... aiN]
            // activation vector = [input vector] * [activation function]
            Matrix L0InputMat = new Matrix(inputVector);

            // now get the weight matrix between layer 0 and 1, multiply it against the input vector
            Matrix L1InputMat = layer0_1.Multiply(L0InputMat);

            // now activate! Assume all nodes have the same activation fn
            DoActivation(L1InputMat);

            // Now let's get the linear combo as inputs to layer2
            Matrix L2InputMat = layer1_2.Multiply(L1InputMat);


            return null;
        }

        private void DoActivation(Matrix m)
        {
            Debug.Assert(m.Cols == 1);
            for(int i=0; i < m.Rows; i++)
            {
                m.Mat[i, 1] = Math.Max(0, m.Mat[i, 0]);
            }
        }
    }
}
