using MatrixLibrary;
using NeuralNets;

namespace NeuralNetsTests.ANNTests
{
    public class SimpleTrainingSet : ITrainingSet
    {
        public float Increment { get; }

        public SimpleTrainingSet()
        {
        }

        public int InputDimension => 2;

        public int OutputDimension => 2;

        public int NumberOfSamples => 1;

        public int NumberOfLabels => 0;

        public List<TrainingPair> TrainingList { get; private set; }

        public List<TrainingPair> BuildNewRandomizedTrainingList()
        {
            TrainingPair tp = new TrainingPair(
                    new ColumnVector(new float[] { 0.4f, 0.9f }),
                    new ColumnVector(new float[] { 0.8357f })
                    );
            this.TrainingList = new List<TrainingPair>() { tp };
            return this.TrainingList;
        }
    }

}
