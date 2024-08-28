using NeuralNets;
using NumReaderNetwork;

namespace MnistReader_ANN
{
    internal class MNISTTrainingSet : ITrainingSet
    {
        public MNISTTrainingSet() 
        {
            this.OutputDimension = 10;

            int width=0;
            int height=0;
            int numLabels = 0;
            int numSamples = 0;

            MnistReader.GetMNISTTrainingMetaData(out numSamples, out numLabels, out width, out height);

            this.NumberOfLabels = numLabels;
            this.NumberOfSamples = numSamples;
            this.InputDimension = width * height;
        }

        public int InputDimension { get; private set; }
        public int OutputDimension { get; private set; }
        public int NumberOfSamples { get; private set; }
        public int NumberOfLabels {  get; private set; }

        public IEnumerable<TrainingPair> GetTrainingPair(bool doRandom)
        {            
            // todo: do random... how?
            foreach (Image image in MnistReader.ReadTrainingData())
            {
                yield return TrainingPairFromImage(image);
            }
        }
        private TrainingPair TrainingPairFromImage(Image image)
        {
            ColumnVector inputVector = ImageDataToColumnVector(image);
            ColumnVector outputVector = OneHotEncodeLabelData(image);
            TrainingPair trainingPair = new TrainingPair(inputVector, outputVector);
            return trainingPair;
        }

        private ColumnVector OneHotEncodeLabelData(Image image)
        {
            // convert the label data (0,1,2, ...) into a columnvector. if the label is 7 (ie: byte == 7), then set the 7th double to 1.0
            double[] labelData = new double[this.OutputDimension];
            labelData[(int)image.Label] = 1.0;
            return new ColumnVector(labelData);
        }

        private static ColumnVector ImageDataToColumnVector(Image image)
        {
            // convert the image data into a columnvector
            double[] imageData = new double[image.Size];
            int i = 0;
            foreach (byte b in image.Data)
            {
                imageData[i++] = (double)b;
            }

            return new ColumnVector(imageData);
        }
    }
}
