using NeuralNets;
using NeuralNets;
using System.Diagnostics;

namespace MnistReader_ANN
{
    public class MNISTTrainingSet : ITrainingSet
    {
        public MNISTTrainingSet()
        {
            this.OutputDimension = 10;

            int width = 0;
            int height = 0;
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
        public int NumberOfLabels { get; private set; }

        public List<TrainingPair> TrainingList { get; private set; }
        private List<Image> ImageList { get; set; } = null;
        public List<TrainingPair> BuildNewRandomizedTrainingList()
        {
            Random rnd = new Random();
            if (this.ImageList == null)
            {
                this.ImageList = MnistReader.ReadTrainingData().ToList();
                Debug.Assert(ImageList.Count == 60000);
            }
            List<TrainingPair> trainingPairs = new List<TrainingPair>((int)ImageList.Count);
            foreach (Image image in ImageList)
            {
                trainingPairs.Add(TrainingPairFromImage(image));
            }            

            this.TrainingList = trainingPairs.OrderBy(x => rnd.Next()).ToList();
            return this.TrainingList;
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
            // convert the label data (0,1,2, ...) into a columnvector. if the label is 7 (ie: byte == 7), then set the 7th float to 1.0
            float[] labelData = new float[this.OutputDimension];
            labelData[(int)image.Label] = 1;
            return new ColumnVector(labelData);
        }

        private static ColumnVector ImageDataToColumnVector(Image image)
        {
            // convert the image data into a columnvector
            float[] imageData = new float[image.Size];
            int i = 0;
            foreach (byte b in image.Data)
            {
                imageData[i++] = (float)b;
            }

            return new ColumnVector(imageData);
        }
    }
}
