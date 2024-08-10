using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    internal class TrainingPair
    {
        public TrainingPair(ColumnVector input, ColumnVector output)
        {
            this.Input = input;
            this.Output = output;
        }

        public ColumnVector Input { get; }
        public ColumnVector Output { get; }
    }

    internal class TrainingData
    {
        public TrainingData(int inputDim, int outputDim, TrainingPair[] trainingPairs)
        {
            this.InputDimensions = inputDim;
            this.OutputDimensions = outputDim;
            this.TrainingPairs = trainingPairs;

        }

        public int InputDimensions { get; }
        public int OutputDimensions { get; }
        public TrainingPair[] TrainingPairs { get; }
    }
}
