using MatrixLibrary;

namespace NeuralNets
{
    public class TrainingPair
    {
        public Tensor Input { get; }
        public Tensor Output { get; }

        public TrainingPair(Tensor input, Tensor output)
        {
            this.Input = input;
            this.Output = output;
        }
        public TrainingPair(AvxColumnVector input, AvxColumnVector output)
        {
            this.Input = input.ToTensor();
            this.Output = output.ToTensor();
        }
    }

    public interface ITrainingSet
    {
        public int Width { get; }
        public int Height { get; }
        public int Depth { get; }
        public InputOutputShape OutputShape { get; }
        public int NumClasses { get; }
        public int NumberOfSamples { get; }
        public int NumberOfLabels { get; }
        public List<TrainingPair> TrainingList { get; }
        public List<TrainingPair> BuildNewRandomizedTrainingList();
    }
}
