using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class TrainingPair
    {
        public TrainingPair(ColumnVector input, ColumnVector output)
        {
            this.Input = input;
            this.Output = output;
        }

        public ColumnVector Input { get; }
        public ColumnVector Output { get; }
    }

    public class TrainingData
    {
        public TrainingData(int inputDim, int outputDim, List<TrainingPair> trainingPairs)
        {
            this.InputDimensions = inputDim;
            this.OutputDimensions = outputDim;
            this.TrainingPairs = trainingPairs;
        }

        public int InputDimensions { get; }
        public int OutputDimensions { get; }
        public List<TrainingPair> TrainingPairs { get; }
    }
}
