using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class Layer
    {
        public int NumNodes { get; private set; }

        protected Layer(int nodeCount)
        {
            this.NumNodes = nodeCount;
        }
    }
    
    public class WeightedLayer : Layer
    {
        public IActivationFunction ActivationFunction { get; protected set; }
        public WeightedLayer(int nodeCount, IActivationFunction activationFunction, int incomingDataPoints) : base(nodeCount)
        {
            int seed = 6874589;
            this.ActivationFunction = activationFunction;
            this.Weights = new Matrix(nodeCount, incomingDataPoints);
            this.Weights.SetRandom(seed, - Math.Sqrt(nodeCount), Math.Sqrt(nodeCount)); // Xavier initilization
            this.Biases = new ColumnVector(nodeCount);
            this.Biases.SetRandom(seed, -0.01, 0.01);            
        }

        public WeightedLayer(
            int nodeCount, 
            IActivationFunction activationFunction, 
            int incomingDataPoints,
            Matrix initialWeights,
            ColumnVector initialBiases) : base(nodeCount)
        {
            Debug.Assert(activationFunction != null);
            this.Weights = initialWeights;
            this.Biases = initialBiases;
            Debug.Assert(this.Weights.Cols == this.Biases.Size);
            Debug.Assert(this.Weights.Rows == incomingDataPoints);
            this.ActivationFunction = activationFunction;

        }

        // Weight matrix is for one sample, and the number of rows corresponds to the number of hidden layer nodes, for example 16.
        // And the number of columns is the number of data points in a samples, for examle 768 b&w pixel values for the MNIST number set
        public Matrix Weights { get;  set; }
        public ColumnVector Biases { get;  set; }
        public ColumnVector ActvationOutput { get; private set; }
        public ColumnVector InputVector { get; private set; }

        public void UpdateWeights(Matrix w2_delta)
        {
            Weights = Weights - w2_delta;
        }

        public void UpdateBiases(ColumnVector b_delta) 
        { 
            Biases = Biases - b_delta;
        }
        public void SetInputVector(ColumnVector z1) { this.InputVector = z1; }
        public void SetActivationOutput(ColumnVector o1) { this.ActvationOutput = o1; }  
        
        public ColumnVector GetActivationFunctionDerivative()
        {
            return this.ActivationFunction.Derivative();
        }
    }

    public class InputLayer : Layer 
    { 
        public InputLayer(int nodeCount) : base(nodeCount)
        {
        }

        public InputLayer(ColumnVector valueVector) : base(valueVector.Size)
        {
            ValueVector = valueVector;
        }

        public ColumnVector ValueVector { get; internal set; }
    }

}
