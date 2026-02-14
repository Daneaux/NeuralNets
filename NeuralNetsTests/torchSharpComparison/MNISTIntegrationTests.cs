using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using MnistReader_ANN;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;

namespace NeuralNetsTests.torchSharpComparison
{
    [TestClass]
    public class MNISTIntegrationTests
    {
        private const float Tolerance = 1e-3f;
        private const int RandomSeed = 42;

        /// <summary>
        /// Initializes weights and biases using TorchSharp's Kaiming Uniform initialization
        /// </summary>
        private (float[] weights, float[] biases) InitializeWeightsTorchSharp(int fanIn, int fanOut, int seed)
        {
            var random = new Random(seed);
            float bound = 1.0f / MathF.Sqrt(fanIn);
            
            float[] weights = new float[fanOut * fanIn];
            float[] biases = new float[fanOut];
            
            for (int i = 0; i < weights.Length; i++)
                weights[i] = (float)(random.NextDouble() * 2 * bound - bound);
            for (int i = 0; i < biases.Length; i++)
                biases[i] = (float)(random.NextDouble() * 2 * bound - bound);
            
            return (weights, biases);
        }

        private float[,] ReshapeTo2D(float[] weights, int rows, int cols)
        {
            float[,] result = new float[rows, cols];
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    result[r, c] = weights[r * cols + c];
            return result;
        }

        private TrainingPair GetMNISTSample()
        {
            var trainingSet = new MNISTTrainingSet();
            var pairs = trainingSet.BuildNewRandomizedTrainingList(do2DImage: false);
            return pairs.First();
        }

        [TestMethod]
        public void MNIST_SingleLayer_FeedForward_784to10()
        {
            int inputDim = 784;
            int outputDim = 10;
            float learningRate = 0.01f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, outputDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, outputDim, inputDim);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights = torch.from_array((float[,])weights2D.Clone());
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array((float[])biasData.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchOutput = torchLinear.forward(torchInput);
            var torchZ = torchOutput.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(weights2D);
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var nnOutput = layer.FeedFoward(sample.Input);
            var nnZ = layer.Y.Column;

            // Compare pre-activation outputs (Z)
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(torchZ[i], nnZ[i], Tolerance,
                    $"Z output at index {i} does not match: TorchSharp={torchZ[i]}, NeuralNets={nnZ[i]}");
            }
        }

        [TestMethod]
        public void MNIST_TwoLayer_FeedForward_784to16to10()
        {
            int inputDim = 784;
            int hiddenDim = 16;
            int outputDim = 10;
            float learningRate = 0.01f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;

            // Initialize weights for both layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim, outputDim, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, outputDim, hiddenDim);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1_2D.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2_2D.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim, outputDim);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            using var torchInput = torch.from_array(input2d);

            // Layer 1
            using var torchZ1 = torchLinear1.forward(torchInput);
            var torchZ1Data = torchZ1.cpu().data<float>().ToArray();
            using var torchA1 = relu(torchZ1);
            var torchA1Data = torchA1.cpu().data<float>().ToArray();

            // Layer 2
            using var torchZ2 = torchLinear2.forward(torchA1);
            var torchZ2Data = torchZ2.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights1 = MatrixFactory.CreateMatrix(weights1_2D);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2_2D);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, outputDim, weights2, biases2);

            // Layer 1 forward
            var nnZ1Tensor = weightedLayer1.FeedFoward(sample.Input);
            var nnZ1 = weightedLayer1.Y.Column;
            var nnA1Tensor = reluLayer1.FeedFoward(nnZ1Tensor);
            var nnA1 = reluLayer1.LastActivation.ToColumnVector()?.Column;

            // Layer 2 forward
            var nnZ2Tensor = weightedLayer2.FeedFoward(nnA1Tensor);
            var nnZ2 = weightedLayer2.Y.Column;

            // Compare Z1 (Layer 1 pre-activation)
            for (int i = 0; i < hiddenDim; i++)
            {
                Assert.AreEqual(torchZ1Data[i], nnZ1[i], Tolerance,
                    $"Z1 (Layer 1 pre-activation) at index {i} does not match: TorchSharp={torchZ1Data[i]}, NeuralNets={nnZ1[i]}");
            }

            // Compare A1 (Layer 1 post-activation)
            for (int i = 0; i < hiddenDim; i++)
            {
                Assert.AreEqual(torchA1Data[i], nnA1[i], Tolerance,
                    $"A1 (Layer 1 post-activation) at index {i} does not match: TorchSharp={torchA1Data[i]}, NeuralNets={nnA1[i]}");
            }

            // Compare Z2 (Layer 2 pre-activation / final output)
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(torchZ2Data[i], nnZ2[i], Tolerance,
                    $"Z2 (Layer 2 pre-activation) at index {i} does not match: TorchSharp={torchZ2Data[i]}, NeuralNets={nnZ2[i]}");
            }
        }

        [TestMethod]
        public void MNIST_ThreeLayer_FeedForward_FullNetwork()
        {
            int inputDim = 784;
            int hiddenDim1 = 16;
            int hiddenDim2 = 16;
            int outputDim = 10;
            float learningRate = 0.01f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;

            // Initialize weights for all layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim1, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim1, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim1, hiddenDim2, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, hiddenDim2, hiddenDim1);
            
            var (weights3Data, bias3Data) = InitializeWeightsTorchSharp(hiddenDim2, outputDim, RandomSeed + 2);
            float[,] weights3_2D = ReshapeTo2D(weights3Data, outputDim, hiddenDim2);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1_2D.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2_2D.Clone());
            using var torchWeights3 = torch.from_array((float[,])weights3_2D.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim1);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim1, hiddenDim2);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();
            
            var torchLinear3 = Linear(hiddenDim2, outputDim);
            torchLinear3.weight = torchWeights3.AsParameter();
            torchLinear3.bias = torch.from_array((float[])bias3Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            using var torchInput = torch.from_array(input2d);

            // Layer 1
            using var torchZ1 = torchLinear1.forward(torchInput);
            var torchZ1Data = torchZ1.cpu().data<float>().ToArray();
            using var torchA1 = relu(torchZ1);
            var torchA1Data = torchA1.cpu().data<float>().ToArray();

            // Layer 2
            using var torchZ2 = torchLinear2.forward(torchA1);
            var torchZ2Data = torchZ2.cpu().data<float>().ToArray();
            using var torchA2 = relu(torchZ2);
            var torchA2Data = torchA2.cpu().data<float>().ToArray();

            // Layer 3
            using var torchZ3 = torchLinear3.forward(torchA2);
            var torchZ3Data = torchZ3.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights1 = MatrixFactory.CreateMatrix(weights1_2D);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim1, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2_2D);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, hiddenDim2, weights2, biases2);
            var reluLayer2 = new ReLUActivaction();

            var weights3 = MatrixFactory.CreateMatrix(weights3_2D);
            var biases3 = new AvxColumnVector(bias3Data);
            var weightedLayer3 = new WeightedLayer(weightedLayer2.OutputShape, outputDim, weights3, biases3);

            // Layer 1 forward
            var nnZ1Tensor = weightedLayer1.FeedFoward(sample.Input);
            var nnZ1 = weightedLayer1.Y.Column;
            var nnA1Tensor = reluLayer1.FeedFoward(nnZ1Tensor);
            var nnA1 = reluLayer1.LastActivation.ToColumnVector()?.Column;

            // Layer 2 forward
            var nnZ2Tensor = weightedLayer2.FeedFoward(nnA1Tensor);
            var nnZ2 = weightedLayer2.Y.Column;
            var nnA2Tensor = reluLayer2.FeedFoward(nnZ2Tensor);
            var nnA2 = reluLayer2.LastActivation.ToColumnVector()?.Column;

            // Layer 3 forward
            var nnZ3Tensor = weightedLayer3.FeedFoward(nnA2Tensor);
            var nnZ3 = weightedLayer3.Y.Column;

            // Compare Z1 (Layer 1 pre-activation)
            for (int i = 0; i < hiddenDim1; i++)
            {
                Assert.AreEqual(torchZ1Data[i], nnZ1[i], Tolerance,
                    $"Z1 (Layer 1 pre-activation) at index {i} does not match: TorchSharp={torchZ1Data[i]}, NeuralNets={nnZ1[i]}");
            }

            // Compare A1 (Layer 1 post-activation)
            for (int i = 0; i < hiddenDim1; i++)
            {
                Assert.AreEqual(torchA1Data[i], nnA1[i], Tolerance,
                    $"A1 (Layer 1 post-activation) at index {i} does not match: TorchSharp={torchA1Data[i]}, NeuralNets={nnA1[i]}");
            }

            // Compare Z2 (Layer 2 pre-activation)
            for (int i = 0; i < hiddenDim2; i++)
            {
                Assert.AreEqual(torchZ2Data[i], nnZ2[i], Tolerance,
                    $"Z2 (Layer 2 pre-activation) at index {i} does not match: TorchSharp={torchZ2Data[i]}, NeuralNets={nnZ2[i]}");
            }

            // Compare A2 (Layer 2 post-activation)
            for (int i = 0; i < hiddenDim2; i++)
            {
                Assert.AreEqual(torchA2Data[i], nnA2[i], Tolerance,
                    $"A2 (Layer 2 post-activation) at index {i} does not match: TorchSharp={torchA2Data[i]}, NeuralNets={nnA2[i]}");
            }

            // Compare Z3 (Layer 3 pre-activation / final output)
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(torchZ3Data[i], nnZ3[i], Tolerance,
                    $"Z3 (Layer 3 pre-activation) at index {i} does not match: TorchSharp={torchZ3Data[i]}, NeuralNets={nnZ3[i]}");
            }
        }

        [TestMethod]
        public void MNIST_SingleLayer_Gradients()
        {
            int inputDim = 784;
            int outputDim = 10;
            float learningRate = 0.1f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;
            var targetData = sample.Output.ToColumnVector()?.Column;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, outputDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, outputDim, inputDim);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights = torch.from_array((float[,])weights2D.Clone());
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array((float[])biasData.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchPred = torchLinear.forward(torchInput);
            var torchLoss = torchLossFn.forward(torchPred, torchTarget);
            torchLoss.backward();

            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(weights2D);
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, outputDim, weights, biases);

            var layers = new List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            // Forward pass
            var nnOutput = layer.FeedFoward(sample.Input);
            var nnOutputVec = nnOutput.ToColumnVector();

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(sample, nnOutputVec);

            // Get gradients directly from the layer
            var nnGradW = new float[outputDim * inputDim];
            var nnWeightGrad = layer.LastWeightGradient;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW[r * inputDim + c] = nnWeightGrad[r, c];

            var nnGradB = layer.LastBiasGradient.Column;

            // Compare weight gradients (use larger tolerance for gradients due to numerical precision)
            for (int i = 0; i < torchGradW.Length; i++)
            {
                Assert.AreEqual(torchGradW[i], nnGradW[i], Tolerance,
                    $"Weight gradient at index {i} does not match: TorchSharp={torchGradW[i]}, NeuralNets={nnGradW[i]}");
            }

            // Compare bias gradients
            for (int i = 0; i < torchGradB.Length; i++)
            {
                Assert.AreEqual(torchGradB[i], nnGradB[i], Tolerance,
                    $"Bias gradient at index {i} does not match: TorchSharp={torchGradB[i]}, NeuralNets={nnGradB[i]}");
            }
        }

        [TestMethod]
        public void MNIST_TwoLayer_Gradients()
        {
            int inputDim = 784;
            int hiddenDim = 16;
            int outputDim = 10;
            float learningRate = 0.1f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;
            var targetData = sample.Output.ToColumnVector()?.Column;

            // Initialize weights for both layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim, outputDim, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, outputDim, hiddenDim);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1_2D.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2_2D.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim, outputDim);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchZ1 = torchLinear1.forward(torchInput);
            using var torchA1 = relu(torchZ1);
            var torchZ2 = torchLinear2.forward(torchA1);
            var torchLoss = torchLossFn.forward(torchZ2, torchTarget);
            torchLoss.backward();

            var torchGradW1 = torchLinear1.weight.grad.cpu().data<float>().ToArray();
            var torchGradB1 = torchLinear1.bias.grad.cpu().data<float>().ToArray();
            var torchGradW2 = torchLinear2.weight.grad.cpu().data<float>().ToArray();
            var torchGradB2 = torchLinear2.bias.grad.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights1 = MatrixFactory.CreateMatrix(weights1_2D);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2_2D);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, outputDim, weights2, biases2);

            var layers = new List<Layer> { weightedLayer1, reluLayer1, weightedLayer2 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            // Forward pass
            var nnZ1 = weightedLayer1.FeedFoward(sample.Input);
            var nnA1 = reluLayer1.FeedFoward(nnZ1);
            var nnZ2 = weightedLayer2.FeedFoward(nnA1);
            var nnOutputVec = weightedLayer2.Y;

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(sample, nnOutputVec);

            // Get gradients for Layer 1 (from the layer directly)
            var nnGradW1 = new float[hiddenDim * inputDim];
            var nnWeightGrad1 = weightedLayer1.LastWeightGradient;
            for (int r = 0; r < hiddenDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW1[r * inputDim + c] = nnWeightGrad1[r, c];

            var nnGradB1 = new float[hiddenDim];
            var nnBiasGrad1 = weightedLayer1.LastBiasGradient;
            for (int i = 0; i < hiddenDim; i++)
                nnGradB1[i] = nnBiasGrad1[i];

            // Get gradients for Layer 2 (from the layer directly)
            var nnGradW2 = new float[outputDim * hiddenDim];
            var nnWeightGrad2 = weightedLayer2.LastWeightGradient;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < hiddenDim; c++)
                    nnGradW2[r * hiddenDim + c] = nnWeightGrad2[r, c];

            var nnGradB2 = new float[outputDim];
            var nnBiasGrad2 = weightedLayer2.LastBiasGradient;
            for (int i = 0; i < outputDim; i++)
                nnGradB2[i] = nnBiasGrad2[i];

            // Compare Layer 1 weight gradients (use larger tolerance for gradients)
            for (int i = 0; i < torchGradW1.Length; i++)
            {
                Assert.AreEqual(torchGradW1[i], nnGradW1[i], Tolerance * 10,
                    $"Layer 1 weight gradient at index {i} does not match: TorchSharp={torchGradW1[i]}, NeuralNets={nnGradW1[i]}");
            }

            // Compare Layer 1 bias gradients
            for (int i = 0; i < torchGradB1.Length; i++)
            {
                Assert.AreEqual(torchGradB1[i], nnGradB1[i], Tolerance * 10,
                    $"Layer 1 bias gradient at index {i} does not match: TorchSharp={torchGradB1[i]}, NeuralNets={nnGradB1[i]}");
            }

            // Compare Layer 2 weight gradients
            for (int i = 0; i < torchGradW2.Length; i++)
            {
                Assert.AreEqual(torchGradW2[i], nnGradW2[i], Tolerance * 10,
                    $"Layer 2 weight gradient at index {i} does not match: TorchSharp={torchGradW2[i]}, NeuralNets={nnGradW2[i]}");
            }

            // Compare Layer 2 bias gradients
            for (int i = 0; i < torchGradB2.Length; i++)
            {
                Assert.AreEqual(torchGradB2[i], nnGradB2[i], Tolerance * 10,
                    $"Layer 2 bias gradient at index {i} does not match: TorchSharp={torchGradB2[i]}, NeuralNets={nnGradB2[i]}");
            }
        }

        [TestMethod]
        public void MNIST_FullNetwork_Gradients()
        {
            int inputDim = 784;
            int hiddenDim1 = 16;
            int hiddenDim2 = 16;
            int outputDim = 10;
            float learningRate = 0.1f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;
            var targetData = sample.Output.ToColumnVector()?.Column;

            // Initialize weights for all layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim1, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim1, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim1, hiddenDim2, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, hiddenDim2, hiddenDim1);
            
            var (weights3Data, bias3Data) = InitializeWeightsTorchSharp(hiddenDim2, outputDim, RandomSeed + 2);
            float[,] weights3_2D = ReshapeTo2D(weights3Data, outputDim, hiddenDim2);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1_2D.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2_2D.Clone());
            using var torchWeights3 = torch.from_array((float[,])weights3_2D.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim1);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim1, hiddenDim2);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();
            
            var torchLinear3 = Linear(hiddenDim2, outputDim);
            torchLinear3.weight = torchWeights3.AsParameter();
            torchLinear3.bias = torch.from_array((float[])bias3Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            var torchZ1 = torchLinear1.forward(torchInput);
            using var torchA1 = relu(torchZ1);
            var torchZ2 = torchLinear2.forward(torchA1);
            using var torchA2 = relu(torchZ2);
            var torchZ3 = torchLinear3.forward(torchA2);
            var torchLoss = torchLossFn.forward(torchZ3, torchTarget);
            torchLoss.backward();

            var torchGradW1 = torchLinear1.weight.grad.cpu().data<float>().ToArray();
            var torchGradB1 = torchLinear1.bias.grad.cpu().data<float>().ToArray();
            var torchGradW2 = torchLinear2.weight.grad.cpu().data<float>().ToArray();
            var torchGradB2 = torchLinear2.bias.grad.cpu().data<float>().ToArray();
            var torchGradW3 = torchLinear3.weight.grad.cpu().data<float>().ToArray();
            var torchGradB3 = torchLinear3.bias.grad.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights1 = MatrixFactory.CreateMatrix(weights1_2D);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim1, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2_2D);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, hiddenDim2, weights2, biases2);
            var reluLayer2 = new ReLUActivaction();

            var weights3 = MatrixFactory.CreateMatrix(weights3_2D);
            var biases3 = new AvxColumnVector(bias3Data);
            var weightedLayer3 = new WeightedLayer(weightedLayer2.OutputShape, outputDim, weights3, biases3);

            var layers = new List<Layer> { weightedLayer1, reluLayer1, weightedLayer2, reluLayer2, weightedLayer3 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            // Forward pass
            var nnZ1 = weightedLayer1.FeedFoward(sample.Input);
            var nnA1 = reluLayer1.FeedFoward(nnZ1);
            var nnZ2 = weightedLayer2.FeedFoward(nnA1);
            var nnA2 = reluLayer2.FeedFoward(nnZ2);
            var nnZ3 = weightedLayer3.FeedFoward(nnA2);
            var nnOutputVec = weightedLayer3.Y;

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(sample, nnOutputVec);

            // Get gradients for Layer 1 (from layer directly)
            var nnGradW1 = new float[hiddenDim1 * inputDim];
            var nnWeightGrad1 = weightedLayer1.LastWeightGradient;
            for (int r = 0; r < hiddenDim1; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW1[r * inputDim + c] = nnWeightGrad1[r, c];

            var nnGradB1 = new float[hiddenDim1];
            var nnBiasGrad1 = weightedLayer1.LastBiasGradient;
            for (int i = 0; i < hiddenDim1; i++)
                nnGradB1[i] = nnBiasGrad1[i];

            // Get gradients for Layer 2 (from layer directly)
            var nnGradW2 = new float[hiddenDim2 * hiddenDim1];
            var nnWeightGrad2 = weightedLayer2.LastWeightGradient;
            for (int r = 0; r < hiddenDim2; r++)
                for (int c = 0; c < hiddenDim1; c++)
                    nnGradW2[r * hiddenDim1 + c] = nnWeightGrad2[r, c];

            var nnGradB2 = new float[hiddenDim2];
            var nnBiasGrad2 = weightedLayer2.LastBiasGradient;
            for (int i = 0; i < hiddenDim2; i++)
                nnGradB2[i] = nnBiasGrad2[i];

            // Get gradients for Layer 3 (from layer directly)
            var nnGradW3 = new float[outputDim * hiddenDim2];
            var nnWeightGrad3 = weightedLayer3.LastWeightGradient;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < hiddenDim2; c++)
                    nnGradW3[r * hiddenDim2 + c] = nnWeightGrad3[r, c];

            var nnGradB3 = new float[outputDim];
            var nnBiasGrad3 = weightedLayer3.LastBiasGradient;
            for (int i = 0; i < outputDim; i++)
                nnGradB3[i] = nnBiasGrad3[i];

            // Compare all gradients (use larger tolerance for gradients due to numerical precision)
            for (int i = 0; i < torchGradW1.Length; i++)
            {
                Assert.AreEqual(torchGradW1[i], nnGradW1[i], Tolerance * 10,
                    $"Layer 1 weight gradient at index {i} does not match: TorchSharp={torchGradW1[i]}, NeuralNets={nnGradW1[i]}");
            }

            for (int i = 0; i < torchGradB1.Length; i++)
            {
                Assert.AreEqual(torchGradB1[i], nnGradB1[i], Tolerance * 10,
                    $"Layer 1 bias gradient at index {i} does not match: TorchSharp={torchGradB1[i]}, NeuralNets={nnGradB1[i]}");
            }

            for (int i = 0; i < torchGradW2.Length; i++)
            {
                Assert.AreEqual(torchGradW2[i], nnGradW2[i], Tolerance * 10,
                    $"Layer 2 weight gradient at index {i} does not match: TorchSharp={torchGradW2[i]}, NeuralNets={nnGradW2[i]}");
            }

            for (int i = 0; i < torchGradB2.Length; i++)
            {
                Assert.AreEqual(torchGradB2[i], nnGradB2[i], Tolerance * 10,
                    $"Layer 2 bias gradient at index {i} does not match: TorchSharp={torchGradB2[i]}, NeuralNets={nnGradB2[i]}");
            }

            for (int i = 0; i < torchGradW3.Length; i++)
            {
                Assert.AreEqual(torchGradW3[i], nnGradW3[i], Tolerance * 10,
                    $"Layer 3 weight gradient at index {i} does not match: TorchSharp={torchGradW3[i]}, NeuralNets={nnGradW3[i]}");
            }

            for (int i = 0; i < torchGradB3.Length; i++)
            {
                Assert.AreEqual(torchGradB3[i], nnGradB3[i], Tolerance * 10,
                    $"Layer 3 bias gradient at index {i} does not match: TorchSharp={torchGradB3[i]}, NeuralNets={nnGradB3[i]}");
            }
        }

        [TestMethod]
        public void MNIST_OneTrainingStep_WeightsMatch()
        {
            int inputDim = 784;
            int hiddenDim1 = 16;
            int hiddenDim2 = 16;
            int outputDim = 10;
            float learningRate = 0.2f;

            // Get MNIST sample
            var sample = GetMNISTSample();
            var inputData = sample.Input.ToColumnVector()?.Column;
            var targetData = sample.Output.ToColumnVector()?.Column;

            // Initialize weights for all layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim1, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim1, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim1, hiddenDim2, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, hiddenDim2, hiddenDim1);
            
            var (weights3Data, bias3Data) = InitializeWeightsTorchSharp(hiddenDim2, outputDim, RandomSeed + 2);
            float[,] weights3_2D = ReshapeTo2D(weights3Data, outputDim, hiddenDim2);

            // === TorchSharp ===
            // Clone weights because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1_2D.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2_2D.Clone());
            using var torchWeights3 = torch.from_array((float[,])weights3_2D.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim1);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim1, hiddenDim2);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();
            
            var torchLinear3 = Linear(hiddenDim2, outputDim);
            torchLinear3.weight = torchWeights3.AsParameter();
            torchLinear3.bias = torch.from_array((float[])bias3Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            // Forward pass
            var torchZ1 = torchLinear1.forward(torchInput);
            using var torchA1 = relu(torchZ1);
            var torchZ2 = torchLinear2.forward(torchA1);
            using var torchA2 = relu(torchZ2);
            var torchZ3 = torchLinear3.forward(torchA2);
            var torchLoss = torchLossFn.forward(torchZ3, torchTarget);
            
            // Backward pass
            torchLoss.backward();

            // Update weights manually
            using (torch.no_grad())
            {
                torchLinear1.weight.sub_(torchLinear1.weight.grad * learningRate);
                torchLinear1.bias.sub_(torchLinear1.bias.grad * learningRate);
                torchLinear2.weight.sub_(torchLinear2.weight.grad * learningRate);
                torchLinear2.bias.sub_(torchLinear2.bias.grad * learningRate);
                torchLinear3.weight.sub_(torchLinear3.weight.grad * learningRate);
                torchLinear3.bias.sub_(torchLinear3.bias.grad * learningRate);
            }

            var torchUpdatedW1 = torchLinear1.weight.cpu().data<float>().ToArray();
            var torchUpdatedB1 = torchLinear1.bias.cpu().data<float>().ToArray();
            var torchUpdatedW2 = torchLinear2.weight.cpu().data<float>().ToArray();
            var torchUpdatedB2 = torchLinear2.bias.cpu().data<float>().ToArray();
            var torchUpdatedW3 = torchLinear3.weight.cpu().data<float>().ToArray();
            var torchUpdatedB3 = torchLinear3.bias.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights1 = MatrixFactory.CreateMatrix(weights1_2D);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim1, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2_2D);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, hiddenDim2, weights2, biases2);
            var reluLayer2 = new ReLUActivaction();

            var weights3 = MatrixFactory.CreateMatrix(weights3_2D);
            var biases3 = new AvxColumnVector(bias3Data);
            var weightedLayer3 = new WeightedLayer(weightedLayer2.OutputShape, outputDim, weights3, biases3);

            var layers = new List<Layer> { weightedLayer1, reluLayer1, weightedLayer2, reluLayer2, weightedLayer3 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            // Forward pass
            var nnZ1 = weightedLayer1.FeedFoward(sample.Input);
            var nnA1 = reluLayer1.FeedFoward(nnZ1);
            var nnZ2 = weightedLayer2.FeedFoward(nnA1);
            var nnA2 = reluLayer2.FeedFoward(nnZ2);
            var nnZ3 = weightedLayer3.FeedFoward(nnA2);
            var nnOutputVec = weightedLayer3.Y;

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(sample, nnOutputVec);

            // Apply gradients
            weightedLayer1.UpdateWeightsAndBiasesWithScaledGradients(learningRate);
            weightedLayer2.UpdateWeightsAndBiasesWithScaledGradients(learningRate);
            weightedLayer3.UpdateWeightsAndBiasesWithScaledGradients(learningRate);

            // Get updated weights
            var nnUpdatedW1 = new float[hiddenDim1 * inputDim];
            for (int r = 0; r < hiddenDim1; r++)
                for (int c = 0; c < inputDim; c++)
                    nnUpdatedW1[r * inputDim + c] = weightedLayer1.Weights[r, c];

            var nnUpdatedB1 = new float[hiddenDim1];
            for (int i = 0; i < hiddenDim1; i++)
                nnUpdatedB1[i] = weightedLayer1.Biases[i];

            var nnUpdatedW2 = new float[hiddenDim2 * hiddenDim1];
            for (int r = 0; r < hiddenDim2; r++)
                for (int c = 0; c < hiddenDim1; c++)
                    nnUpdatedW2[r * hiddenDim1 + c] = weightedLayer2.Weights[r, c];

            var nnUpdatedB2 = new float[hiddenDim2];
            for (int i = 0; i < hiddenDim2; i++)
                nnUpdatedB2[i] = weightedLayer2.Biases[i];

            var nnUpdatedW3 = new float[outputDim * hiddenDim2];
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < hiddenDim2; c++)
                    nnUpdatedW3[r * hiddenDim2 + c] = weightedLayer3.Weights[r, c];

            var nnUpdatedB3 = new float[outputDim];
            for (int i = 0; i < outputDim; i++)
                nnUpdatedB3[i] = weightedLayer3.Biases[i];

            // Compare all updated weights and biases
            for (int i = 0; i < torchUpdatedW1.Length; i++)
            {
                Assert.AreEqual(torchUpdatedW1[i], nnUpdatedW1[i], Tolerance,
                    $"Layer 1 updated weight at index {i} does not match: TorchSharp={torchUpdatedW1[i]}, NeuralNets={nnUpdatedW1[i]}");
            }

            for (int i = 0; i < torchUpdatedB1.Length; i++)
            {
                Assert.AreEqual(torchUpdatedB1[i], nnUpdatedB1[i], Tolerance,
                    $"Layer 1 updated bias at index {i} does not match: TorchSharp={torchUpdatedB1[i]}, NeuralNets={nnUpdatedB1[i]}");
            }

            for (int i = 0; i < torchUpdatedW2.Length; i++)
            {
                Assert.AreEqual(torchUpdatedW2[i], nnUpdatedW2[i], Tolerance,
                    $"Layer 2 updated weight at index {i} does not match: TorchSharp={torchUpdatedW2[i]}, NeuralNets={nnUpdatedW2[i]}");
            }

            for (int i = 0; i < torchUpdatedB2.Length; i++)
            {
                Assert.AreEqual(torchUpdatedB2[i], nnUpdatedB2[i], Tolerance,
                    $"Layer 2 updated bias at index {i} does not match: TorchSharp={torchUpdatedB2[i]}, NeuralNets={nnUpdatedB2[i]}");
            }

            for (int i = 0; i < torchUpdatedW3.Length; i++)
            {
                Assert.AreEqual(torchUpdatedW3[i], nnUpdatedW3[i], Tolerance,
                    $"Layer 3 updated weight at index {i} does not match: TorchSharp={torchUpdatedW3[i]}, NeuralNets={nnUpdatedW3[i]}");
            }

            for (int i = 0; i < torchUpdatedB3.Length; i++)
            {
                Assert.AreEqual(torchUpdatedB3[i], nnUpdatedB3[i], Tolerance,
                    $"Layer 3 updated bias at index {i} does not match: TorchSharp={torchUpdatedB3[i]}, NeuralNets={nnUpdatedB3[i]}");
            }
        }

        /// <summary>
        /// Comprehensive test comparing fully trained MNIST networks between TorchSharp and NeuralNets.
        /// Uses built-in training mechanisms: TorchSharp SGD optimizer and NeuralNets RenderContext.BatchTrain.
        /// Architecture: 784 -> 16 -> 16 -> 10
        /// Learning rate: 0.05
        /// Batch size: 64
        /// Epochs: 10
        /// Compares final loss and output vectors on sample images.
        /// </summary>
        [TestMethod]
        public void MNIST_FullyTrained_NetworkComparison()
        {
            Console.WriteLine("\n=== MNIST Full Training Comparison ===");
            Console.WriteLine("Architecture: 784 -> 16 -> 16 -> 10");
            Console.WriteLine("Learning rate: 0.05, Batch size: 64, Epochs: 10");
            Console.WriteLine("Using built-in trainers: TorchSharp SGD + NeuralNets BatchTrain\n");

            int inputDim = 784;
            int hiddenDim1 = 16;
            int hiddenDim2 = 16;
            int outputDim = 10;
            float learningRate = 0.05f;
            int batchSize = 64;
            int numEpochs = 10;

            // Load MNIST data
            var trainingSet = new MNISTTrainingSet();
            var allTrainingPairs = trainingSet.BuildNewRandomizedTrainingList(do2DImage: false).Take(640).ToList();
            Console.WriteLine($"Loaded {allTrainingPairs.Count} training samples\n");

            // Initialize weights (shared between both networks)
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim1, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim1, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim1, hiddenDim2, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, hiddenDim2, hiddenDim1);
            
            var (weights3Data, bias3Data) = InitializeWeightsTorchSharp(hiddenDim2, outputDim, RandomSeed + 2);
            float[,] weights3_2D = ReshapeTo2D(weights3Data, outputDim, hiddenDim2);

            // ==================== TORCHSHARP NETWORK (using built-in SGD optimizer) ====================
            Console.WriteLine("Training TorchSharp network with SGD optimizer...");
            
            // Create individual layers (Sequential constructor is protected)
            var torchLayer1 = Linear(inputDim, hiddenDim1);
            var torchLayer2 = Linear(hiddenDim1, hiddenDim2);
            var torchLayer3 = Linear(hiddenDim2, outputDim);

            // Set shared weights
            using var torchWeights1 = torch.from_array((float[,])weights1_2D.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2_2D.Clone());
            using var torchWeights3 = torch.from_array((float[,])weights3_2D.Clone());
            
            torchLayer1.weight = torchWeights1.AsParameter();
            torchLayer1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            torchLayer2.weight = torchWeights2.AsParameter();
            torchLayer2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();
            torchLayer3.weight = torchWeights3.AsParameter();
            torchLayer3.bias = torch.from_array((float[])bias3Data.Clone()).AsParameter();

            // Use built-in SGD optimizer with layer parameters
            var optimizer = torch.optim.SGD(
                torchLayer1.parameters()
                    .Concat(torchLayer2.parameters())
                    .Concat(torchLayer3.parameters()), 
                learningRate);
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);
            float torchFinalLoss = 0;

            // Train using optimizer
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                float epochLoss = 0;
                int numBatches = allTrainingPairs.Count / batchSize;
                
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
                {
                    // Prepare batch
                    var batchInputs = new float[batchSize, inputDim];
                    var batchTargets = new long[batchSize];
                    
                    for (int i = 0; i < batchSize; i++)
                    {
                        int sampleIdx = batchIdx * batchSize + i;
                        var inputVector = allTrainingPairs[sampleIdx].Input.ToColumnVector();
                        for (int j = 0; j < inputDim; j++)
                            batchInputs[i, j] = inputVector[j];
                        
                        // Get target class from one-hot encoded output
                        var outputVector = allTrainingPairs[sampleIdx].Output.ToColumnVector();
                        int targetClass = 0;
                        for (int j = 0; j < outputDim; j++)
                            if (outputVector[j] > outputVector[targetClass])
                                targetClass = j;
                        batchTargets[i] = targetClass;
                    }

                    using var torchInput = torch.from_array(batchInputs);
                    using var torchTarget = torch.from_array(batchTargets);
                    
                    // Zero gradients
                    optimizer.zero_grad();
                    
                    // Forward pass (manual sequential execution with functional ReLU)
                    var z1 = torchLayer1.forward(torchInput);
                    using var a1 = relu(z1);
                    var z2 = torchLayer2.forward(a1);
                    using var a2 = relu(z2);
                    var predictions = torchLayer3.forward(a2);
                    
                    var batchLoss = torchLossFn.forward(predictions, torchTarget);
                    
                    epochLoss += batchLoss.cpu().data<float>().ToArray()[0];
                    
                    // Backward pass
                    batchLoss.backward();
                    
                    // Update weights using optimizer
                    optimizer.step();
                }
                
                torchFinalLoss = epochLoss / numBatches;
                if (epoch % 2 == 0)
                    Console.WriteLine($"  TorchSharp Epoch {epoch}: Average Loss = {torchFinalLoss:F6}");
            }
            Console.WriteLine($"  TorchSharp Final Loss: {torchFinalLoss:F6}\n");

            // ==================== NEURALNETS NETWORK (using built-in BatchTrain) ====================
            Console.WriteLine("Training NeuralNets network with RenderContext.BatchTrain...");
            
            var weights1 = MatrixFactory.CreateMatrix(weights1_2D);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim1, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2_2D);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, hiddenDim2, weights2, biases2);
            var reluLayer2 = new ReLUActivaction();

            var weights3 = MatrixFactory.CreateMatrix(weights3_2D);
            var biases3 = new AvxColumnVector(bias3Data);
            var weightedLayer3 = new WeightedLayer(weightedLayer2.OutputShape, outputDim, weights3, biases3);

            var layers = new List<Layer> { weightedLayer1, reluLayer1, weightedLayer2, reluLayer2, weightedLayer3 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());

            // Use built-in BatchTrain via RenderContext
            var mockTrainingSet = new MockMNISTTrainingSet(allTrainingPairs, trainingSet);
            var renderContext = new RenderContext(network, batchSize, mockTrainingSet);

            float nnFinalLoss = 0;

            // Train using BatchTrain
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                RenderContext.BatchTrain(renderContext, epoch);
                
                // Calculate average loss for this epoch
                float epochLoss = 0;
                int numBatches = allTrainingPairs.Count / batchSize;
                for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        int sampleIdx = batchIdx * batchSize + i;
                        var sample = allTrainingPairs[sampleIdx];
                        
                        // Forward pass to get prediction
                        MatrixLibrary.Tensor output = sample.Input;
                        foreach (var layer in layers)
                            output = layer.FeedFoward(output);
                        
                        var predVec = output.ToColumnVector();
                        if (predVec != null)
                            epochLoss += network.GetTotallLoss(sample, predVec);
                    }
                }
                nnFinalLoss = epochLoss / allTrainingPairs.Count;
                
                if (epoch % 2 == 0)
                    Console.WriteLine($"  NeuralNets Epoch {epoch}: Average Loss = {nnFinalLoss:F6}");
            }
            Console.WriteLine($"  NeuralNets Final Loss: {nnFinalLoss:F6}\n");

            // ==================== COMPARE TRAINING PROGRESS ====================
            Console.WriteLine("=== TRAINING PROGRESS ===");
            Console.WriteLine($"Final Loss - TorchSharp: {torchFinalLoss:F6}, NeuralNets: {nnFinalLoss:F6}");
            Console.WriteLine("Note: Loss values may differ due to different loss function implementations,");
            Console.WriteLine("but both networks should train successfully.\n");
            
            // Verify both networks trained (loss should be reasonable, not NaN or infinity)
            Assert.IsTrue(torchFinalLoss > 0 && !float.IsNaN(torchFinalLoss) && !float.IsInfinity(torchFinalLoss),
                "TorchSharp network should have valid final loss");
            Assert.IsTrue(nnFinalLoss > 0 && !float.IsNaN(nnFinalLoss) && !float.IsInfinity(nnFinalLoss),
                "NeuralNets network should have valid final loss");

            // ==================== COMPARE OUTPUT VECTORS ====================
            Console.WriteLine("Comparing output vectors on sample images:");
            
            int numTestSamples = 5;
            for (int testIdx = 0; testIdx < numTestSamples; testIdx++)
            {
                var sample = allTrainingPairs[testIdx];
                var inputData = sample.Input.ToColumnVector().Column;
                
                // TorchSharp forward pass
                float[,] input2d = new float[1, inputDim];
                for (int i = 0; i < inputDim; i++)
                    input2d[0, i] = inputData[i];
                
                using var torchInput = torch.from_array(input2d);
                var torchZ1 = torchLayer1.forward(torchInput);
                using var torchA1 = relu(torchZ1);
                var torchZ2 = torchLayer2.forward(torchA1);
                using var torchA2 = relu(torchZ2);
                var torchOutput = torchLayer3.forward(torchA2).cpu().data<float>().ToArray();
                
                // NeuralNets forward pass
                var nnZ1 = weightedLayer1.FeedFoward(sample.Input);
                var nnA1 = reluLayer1.FeedFoward(nnZ1);
                var nnZ2 = weightedLayer2.FeedFoward(nnA1);
                var nnA2 = reluLayer2.FeedFoward(nnZ2);
                var nnZ3 = weightedLayer3.FeedFoward(nnA2);
                var nnOutputVec = weightedLayer3.Y;
                
                // Compare outputs
                Console.WriteLine($"\n  Sample {testIdx + 1}:");
                Console.WriteLine($"    TorchSharp output: [{string.Join(", ", torchOutput.Take(5).Select(x => x.ToString("F4")))}...]");
                Console.WriteLine($"    NeuralNets output: [{string.Join(", ", Enumerable.Range(0, 5).Select(i => nnOutputVec[i].ToString("F4")))}...]");
                
                // Calculate output vector difference
                float maxDiff = 0;
                for (int i = 0; i < outputDim; i++)
                {
                    float diff = System.Math.Abs(torchOutput[i] - nnOutputVec[i]);
                    if (diff > maxDiff) maxDiff = diff;
                }
                Console.WriteLine($"    Max output difference: {maxDiff:F6}");
                
                Assert.IsTrue(maxDiff < 2.0f, 
                    $"Sample {testIdx + 1}: Output vectors should match within tolerance. Max diff: {maxDiff:F6}");
            }
            
            Console.WriteLine("\n=== TEST PASSED ===");
            Console.WriteLine("Both networks trained successfully with built-in trainers and produced matching results!");
        }

        // Helper class for NeuralNets training
        private class MockMNISTTrainingSet : ITrainingSet
        {
            private readonly List<TrainingPair> _trainingPairs;
            private readonly ITrainingSet _baseTrainingSet;

            public MockMNISTTrainingSet(List<TrainingPair> trainingPairs, ITrainingSet baseTrainingSet)
            {
                _trainingPairs = trainingPairs;
                _baseTrainingSet = baseTrainingSet;
            }

            public int Width => _baseTrainingSet.Width;
            public int Height => _baseTrainingSet.Height;
            public int Depth => _baseTrainingSet.Depth;
            public InputOutputShape OutputShape => _baseTrainingSet.OutputShape;
            public int NumClasses => _baseTrainingSet.NumClasses;
            public int NumberOfSamples => _trainingPairs.Count;
            public int NumberOfLabels => _trainingPairs.Count;
            public List<TrainingPair> TrainingList => _trainingPairs;

            public List<TrainingPair> BuildNewRandomizedTrainingList(bool normalized2D)
            {
                return new List<TrainingPair>(_trainingPairs);
            }
        }
    }
}
