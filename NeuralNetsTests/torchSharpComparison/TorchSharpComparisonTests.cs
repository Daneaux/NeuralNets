using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
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
    public class TorchSharpComparisonTests
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

        [TestMethod]
        public void LinearLayer_FeedForward_ZAndAOutputsMatch()
        {
            int inputDim = 4;
            int outputDim = 3;
            float learningRate = 0.01f;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, outputDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, outputDim, inputDim);

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };

            // === TorchSharp ===
            float[,] torchWeights2D = (float[,])weights2D.Clone();
            float[] torchBiasData = (float[])biasData.Clone();
            using var torchWeights = torch.from_array(torchWeights2D);
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array(torchBiasData).AsParameter();

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

            var nnInput = new AvxColumnVector(inputData);
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnZ = layer.Y.Column;

            // Compare pre-activation outputs (Z)
            for (int i = 0; i < outputDim; i++)
            {
                Assert.AreEqual(torchZ[i], nnZ[i], Tolerance,
                    $"Z output at index {i} does not match: TorchSharp={torchZ[i]}, NeuralNets={nnZ[i]}");
            }
        }

        [TestMethod]
        public void LinearWithReLU_FeedForward_ZAndAOutputsMatch()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            float learningRate = 0.01f;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, hiddenDim, inputDim);

            float[] inputData = { 0.5f, -0.3f, 0.2f, -0.1f };

            // === TorchSharp ===
            float[,] torchWeights2D = (float[,])weights2D.Clone();
            float[] torchBiasData = (float[])biasData.Clone();
            using var torchWeights = torch.from_array(torchWeights2D);
            var torchLinear = Linear(inputDim, hiddenDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array(torchBiasData).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            using var torchInput = torch.from_array(input2d);
            
            // Pre-activation (Z)
            using var torchZ = torchLinear.forward(torchInput);
            var torchZData = torchZ.cpu().data<float>().ToArray();
            
            // Post-activation (A) with ReLU
            using var torchA = relu(torchZ);
            var torchAData = torchA.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(weights2D);
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer = new WeightedLayer(inputShape, hiddenDim, weights, biases);
            var reluLayer = new ReLUActivaction();

            var nnInput = new AvxColumnVector(inputData);
            
            // Forward through weighted layer (pre-activation Z)
            var nnZTensor = weightedLayer.FeedFoward(nnInput.ToTensor());
            var nnZ = weightedLayer.Y.Column;
            
            // Forward through ReLU (post-activation A)
            var nnATensor = reluLayer.FeedFoward(nnZTensor);
            var nnA = reluLayer.LastActivation.ToColumnVector()?.Column;

            // Compare pre-activation outputs (Z)
            for (int i = 0; i < hiddenDim; i++)
            {
                Assert.AreEqual(torchZData[i], nnZ[i], Tolerance,
                    $"Z (pre-activation) at index {i} does not match: TorchSharp={torchZData[i]}, NeuralNets={nnZ[i]}");
            }

            // Compare post-activation outputs (A)
            for (int i = 0; i < hiddenDim; i++)
            {
                Assert.AreEqual(torchAData[i], nnA[i], Tolerance,
                    $"A (post-activation) at index {i} does not match: TorchSharp={torchAData[i]}, NeuralNets={nnA[i]}");
            }
        }

        [TestMethod]
        public void TwoLayer_FeedForward_ZAndAOutputsMatch()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            int outputDim = 2;
            float learningRate = 0.01f;

            // Initialize weights for both layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim, outputDim, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, outputDim, hiddenDim);

            float[] inputData = { 0.5f, -0.3f, 0.2f, -0.1f };

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

            var nnInput = new AvxColumnVector(inputData);

            // Layer 1 forward
            var nnZ1Tensor = weightedLayer1.FeedFoward(nnInput.ToTensor());
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
        public void LinearLayer_BackProp_GradientsMatch_SquaredLoss()
        {
            int inputDim = 4;
            int outputDim = 3;
            float learningRate = 0.1f;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, outputDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, outputDim, inputDim);

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] targetData = { 1.0f, 0.0f, 0.0f };

            // === TorchSharp ===
            // Create copies so TorchSharp doesn't modify the original arrays
            float[,] torchWeights2D = (float[,])weights2D.Clone();
            float[] torchBiasData = (float[])biasData.Clone();
            using var torchWeights = torch.from_array(torchWeights2D);
            var torchLinear = Linear(inputDim, outputDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array(torchBiasData).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, outputDim];
            for (int i = 0; i < outputDim; i++) target2d[0, i] = targetData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = MSELoss(reduction: Reduction.Sum);

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
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new MeanSquaredErrorLoss());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnOutputVec = nnOutput.ToColumnVector();

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutputVec);
            layer.UpdateWeightsAndBiasesWithScaledGradients(learningRate);

            // Get gradients
            var nnGradW = new float[outputDim * inputDim];
            var nnWeightGrad = layer.LastWeightGradient;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW[r * inputDim + c] = nnWeightGrad[r, c];

            var nnGradB = layer.LastBiasGradient.Column;

            // Compare weight gradients
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
        public void LinearWithReLU_BackProp_GradientsMatch_SquaredLoss()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            float learningRate = 0.1f;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, hiddenDim, inputDim);

            float[] inputData = { 0.5f, -0.3f, 0.2f, -0.1f };
            float[] targetData = { 1.0f, 0.0f, 0.0f };

            // === TorchSharp ===
            float[,] torchWeights2D = (float[,])weights2D.Clone();
            float[] torchBiasData = (float[])biasData.Clone();
            using var torchWeights = torch.from_array(torchWeights2D);
            var torchLinear = Linear(inputDim, hiddenDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array(torchBiasData).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, hiddenDim];
            for (int i = 0; i < hiddenDim; i++) target2d[0, i] = targetData[i % 3];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = MSELoss(reduction: Reduction.Sum);

            var torchZ = torchLinear.forward(torchInput);
            using var torchA = relu(torchZ);
            var torchLoss = torchLossFn.forward(torchA, torchTarget);
            torchLoss.backward();

            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();

            // === NeuralNets ===
            var weights = MatrixFactory.CreateMatrix(weights2D);
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer = new WeightedLayer(inputShape, hiddenDim, weights, biases);
            var reluLayer = new ReLUActivaction();

            var layers = new List<Layer> { weightedLayer, reluLayer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, hiddenDim, new MeanSquaredErrorLoss());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData.Take(hiddenDim).ToArray());
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnZ = weightedLayer.FeedFoward(nnInput.ToTensor());
            var nnA = reluLayer.FeedFoward(nnZ);
            var nnOutputVec = nnA.ToColumnVector();

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutputVec);

            // Get gradients from layer directly
            var nnGradW = new float[hiddenDim * inputDim];
            var nnWeightGrad = weightedLayer.LastWeightGradient;
            for (int r = 0; r < hiddenDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW[r * inputDim + c] = nnWeightGrad[r, c];

            var nnGradB = new float[hiddenDim];
            var nnBiasGrad = weightedLayer.LastBiasGradient;
            for (int i = 0; i < hiddenDim; i++)
                nnGradB[i] = nnBiasGrad[i];

            // Compare weight gradients
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
        public void TwoLayer_BackProp_GradientsMatch_CatCrossEntropy()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            int outputDim = 2;
            float learningRate = 0.1f;

            // Initialize weights for both layers
            var (weights1Data, bias1Data) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights1_2D = ReshapeTo2D(weights1Data, hiddenDim, inputDim);
            
            var (weights2Data, bias2Data) = InitializeWeightsTorchSharp(hiddenDim, outputDim, RandomSeed + 1);
            float[,] weights2_2D = ReshapeTo2D(weights2Data, outputDim, hiddenDim);

            float[] inputData = { 0.5f, -0.3f, 0.2f, -0.1f };
            long[] targetIndex = { 0 }; // Class index for CrossEntropyLoss

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
            using var torchTarget = torch.from_array(targetIndex);
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

            var nnInput = new AvxColumnVector(inputData);
            // One-hot encode target for NeuralNets (CrossEntropy expects class probabilities)
            float[] targetProbs = { 1.0f, 0.0f };
            var nnTarget = new AvxColumnVector(targetProbs);
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnZ1 = weightedLayer1.FeedFoward(nnInput.ToTensor());
            var nnA1 = reluLayer1.FeedFoward(nnZ1);
            var nnZ2 = weightedLayer2.FeedFoward(nnA1);
            var nnOutputVec = weightedLayer2.Y;

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutputVec);

            // Get gradients for Layer 1 (from layer directly)
            var nnGradW1 = new float[hiddenDim * inputDim];
            var nnWeightGrad1 = weightedLayer1.LastWeightGradient;
            for (int r = 0; r < hiddenDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnGradW1[r * inputDim + c] = nnWeightGrad1[r, c];

            var nnGradB1 = new float[hiddenDim];
            var nnBiasGrad1 = weightedLayer1.LastBiasGradient;
            for (int i = 0; i < hiddenDim; i++)
                nnGradB1[i] = nnBiasGrad1[i];

            // Get gradients for Layer 2 (from layer directly)
            var nnGradW2 = new float[outputDim * hiddenDim];
            var nnWeightGrad2 = weightedLayer2.LastWeightGradient;
            for (int r = 0; r < outputDim; r++)
                for (int c = 0; c < hiddenDim; c++)
                    nnGradW2[r * hiddenDim + c] = nnWeightGrad2[r, c];

            var nnGradB2 = new float[outputDim];
            var nnBiasGrad2 = weightedLayer2.LastBiasGradient;
            for (int i = 0; i < outputDim; i++)
                nnGradB2[i] = nnBiasGrad2[i];

            // Compare Layer 1 weight gradients
            for (int i = 0; i < torchGradW1.Length; i++)
            {
                Assert.AreEqual(torchGradW1[i], nnGradW1[i], Tolerance,
                    $"Layer 1 weight gradient at index {i} does not match: TorchSharp={torchGradW1[i]}, NeuralNets={nnGradW1[i]}");
            }

            // Compare Layer 1 bias gradients
            for (int i = 0; i < torchGradB1.Length; i++)
            {
                Assert.AreEqual(torchGradB1[i], nnGradB1[i], Tolerance,
                    $"Layer 1 bias gradient at index {i} does not match: TorchSharp={torchGradB1[i]}, NeuralNets={nnGradB1[i]}");
            }

            // Compare Layer 2 weight gradients
            for (int i = 0; i < torchGradW2.Length; i++)
            {
                Assert.AreEqual(torchGradW2[i], nnGradW2[i], Tolerance,
                    $"Layer 2 weight gradient at index {i} does not match: TorchSharp={torchGradW2[i]}, NeuralNets={nnGradW2[i]}");
            }

            // Compare Layer 2 bias gradients
            for (int i = 0; i < torchGradB2.Length; i++)
            {
                Assert.AreEqual(torchGradB2[i], nnGradB2[i], Tolerance,
                    $"Layer 2 bias gradient at index {i} does not match: TorchSharp={torchGradB2[i]}, NeuralNets={nnGradB2[i]}");
            }
        }

        [TestMethod]
        public void OneTrainingStep_WeightsMatch_SquaredLoss()
        {
            int inputDim = 4;
            int hiddenDim = 3;
            float learningRate = 0.1f;

            // Initialize weights using TorchSharp Kaiming uniform
            var (weightsData, biasData) = InitializeWeightsTorchSharp(inputDim, hiddenDim, RandomSeed);
            float[,] weights2D = ReshapeTo2D(weightsData, hiddenDim, inputDim);

            Console.WriteLine($"DEBUG: Generated weights2D[0,0] = {weights2D[0, 0]:F6}");

            float[] inputData = { 0.5f, 0.3f, 0.2f, 0.1f };
            float[] targetData = { 1.0f, 0.0f, 0.0f };

            // === TorchSharp ===
            // Create copies so TorchSharp doesn't modify the original arrays
            float[,] torchWeightsData = (float[,])weights2D.Clone();
            float[] torchBiasData = (float[])biasData.Clone();
            using var torchWeights = torch.from_array(torchWeightsData);
            var torchLinear = Linear(inputDim, hiddenDim);
            torchLinear.weight = torchWeights.AsParameter();
            torchLinear.bias = torch.from_array(torchBiasData).AsParameter();

            Console.WriteLine($"DEBUG: weights2D[0,0] after torch setup: {weights2D[0, 0]:F6}");

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            float[,] target2d = new float[1, hiddenDim];
            for (int i = 0; i < hiddenDim; i++) target2d[0, i] = targetData[i % 3];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(target2d);
            using var torchLossFn = MSELoss(reduction: Reduction.Sum);

            var torchPred = torchLinear.forward(torchInput);
            Console.WriteLine($"DEBUG: weights2D[0,0] after forward: {weights2D[0, 0]:F6}");
            var torchLoss = torchLossFn.forward(torchPred, torchTarget);
            torchLoss.backward();
            Console.WriteLine($"DEBUG: weights2D[0,0] after backward: {weights2D[0, 0]:F6}");

            // Get initial weights and gradients for comparison
            var torchInitialWeights = torchLinear.weight.clone().cpu().data<float>().ToArray();
            var torchInitialBias = torchLinear.bias.clone().cpu().data<float>().ToArray();
            var torchGradW = torchLinear.weight.grad.cpu().data<float>().ToArray();
            var torchGradB = torchLinear.bias.grad.cpu().data<float>().ToArray();
            
            Console.WriteLine($"DEBUG: TorchSharp initial weight[0]: {torchInitialWeights[0]:F6}");
            Console.WriteLine($"DEBUG: TorchSharp gradient[0]: {torchGradW[0]:F6}");

            // Manually update weights (simulating optimizer step)
            using (torch.no_grad())
            {
                var gradW = torchLinear.weight.grad;
                var gradB = torchLinear.bias.grad;
                torchLinear.weight.sub_(gradW * learningRate);
                torchLinear.bias.sub_(gradB * learningRate);
            }

            var torchUpdatedWeights = torchLinear.weight.cpu().data<float>().ToArray();
            var torchUpdatedBias = torchLinear.bias.cpu().data<float>().ToArray();

            // === NeuralNets ===
            Console.WriteLine($"DEBUG: weights2D[0,0] before NeuralNets section: {weights2D[0, 0]:F6}");
            var weights = MatrixFactory.CreateMatrix(weights2D);
            Console.WriteLine($"DEBUG: weights[0,0] after CreateMatrix: {weights[0, 0]:F6}");
            var biases = new AvxColumnVector(biasData);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var layer = new WeightedLayer(inputShape, hiddenDim, weights, biases);
            Console.WriteLine($"DEBUG: layer.Weights[0,0] after constructor: {layer.Weights[0, 0]:F6}");

            var layers = new List<Layer> { layer };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, hiddenDim, new MeanSquaredErrorLoss());

            var nnInput = new AvxColumnVector(inputData);
            var nnTarget = new AvxColumnVector(targetData.Take(hiddenDim).ToArray());
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());

            // Forward pass
            var nnOutput = layer.FeedFoward(nnInput.ToTensor());
            var nnOutputVec = nnOutput.ToColumnVector();

            // Backward pass using RenderContext
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, nnOutputVec);
            
            Console.WriteLine($"DEBUG: NeuralNets initial weight[0]: {layer.Weights[0, 0]:F6}");
            Console.WriteLine($"DEBUG: NeuralNets gradient[0]: {layer.LastWeightGradient[0, 0]:F6}");

            // Apply gradients
            layer.UpdateWeightsAndBiasesWithScaledGradients(learningRate);

            // Get updated weights
            var nnUpdatedWeights = new float[hiddenDim * inputDim];
            for (int r = 0; r < hiddenDim; r++)
                for (int c = 0; c < inputDim; c++)
                    nnUpdatedWeights[r * inputDim + c] = layer.Weights[r, c];

            var nnUpdatedBias = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++)
                nnUpdatedBias[i] = layer.Biases[i];

            // Compare updated weights
            for (int i = 0; i < torchUpdatedWeights.Length; i++)
            {
                Assert.AreEqual(torchUpdatedWeights[i], nnUpdatedWeights[i], Tolerance,
                    $"Updated weight at index {i} does not match: TorchSharp={torchUpdatedWeights[i]}, NeuralNets={nnUpdatedWeights[i]}");
            }

            // Compare updated biases
            for (int i = 0; i < torchUpdatedBias.Length; i++)
            {
                Assert.AreEqual(torchUpdatedBias[i], nnUpdatedBias[i], Tolerance,
                    $"Updated bias at index {i} does not match: TorchSharp={torchUpdatedBias[i]}, NeuralNets={nnUpdatedBias[i]}");
            }
        }
    }
}
