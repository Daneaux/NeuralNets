using MatrixLibrary;

namespace NeuralNets
{
    public class TrainingPair
    {
        public AvxColumnVector Input { get; }
        public AvxColumnVector Output { get; }

        public TrainingPair(AvxColumnVector input, AvxColumnVector output)
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
