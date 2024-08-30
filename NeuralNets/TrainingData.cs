using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNets
{
    public class TrainingPair
    {
        public ColumnVector Input { get; }
        public ColumnVector Output { get; }

        public TrainingPair(ColumnVector input, ColumnVector output)
        {
            this.Input = input;
            this.Output = output;
        }
    }

    public interface ITrainingSet
    {
        public int InputDimension { get; }
        public int OutputDimension { get; }
        public int NumberOfSamples { get; }
        public int NumberOfLabels { get; }
        public List<TrainingPair> TrainingList { get; }
        public List<TrainingPair> BuildNewRandomizedTrainingList();
    }
}
