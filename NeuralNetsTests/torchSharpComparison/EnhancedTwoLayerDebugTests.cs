using MatrixLibrary;
using MatrixLibrary.BaseClasses;
using NeuralNets;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using static TorchSharp.torch.nn.functional;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Linq;
using System.Text;

namespace NeuralNetsTests.torchSharpComparison
{
    [TestClass]
    public class EnhancedTwoLayerDebugTests
    {
        private const float Tolerance = 1e-4f;
        private StringBuilder debugOutput = new StringBuilder();

        private void Log(string message)
        {
            debugOutput.AppendLine(message);
            Console.WriteLine(message);
        }

        private void LogSection(string title)
        {
            Log($"\n{'='}{title}{('=' * (70 - title.Length))}");
        }

        private void LogSubSection(string title)
        {
            Log($"\n--- {title} ---");
        }

        private string FormatArray(float[] arr, int decimals = 6)
        {
            return "[" + string.Join(", ", arr.Select(x => x.ToString($"F{decimals}"))) + "]";
        }

        private string FormatMatrix(float[,] mat, int decimals = 6)
        {
            int rows = mat.GetLength(0);
            int cols = mat.GetLength(1);
            var lines = new List<string>();
            for (int i = 0; i < rows; i++)
            {
                var rowVals = Enumerable.Range(0, cols).Select(j => mat[i, j].ToString($"F{decimals}"));
                lines.Add($"    [{string.Join(", ", rowVals)}]");
            }
            return string.Join("\n", lines);
        }

        private bool CompareValues(string name, float expected, float actual, float tolerance)
        {
            float diff = System.Math.Abs(expected - actual);
            bool match = diff <= tolerance;
            string status = match ? "✓ MATCH" : "✗ MISMATCH";
            Log($"  {name}: Expected={expected:F6}, Actual={actual:F6}, Diff={diff:F6} {status}");
            return match;
        }

        private bool CompareArrays(string name, float[] expected, float[] actual, float tolerance)
        {
            Log($"\n  {name}:");
            bool allMatch = true;
            for (int i = 0; i < expected.Length; i++)
            {
                float diff = System.Math.Abs(expected[i] - actual[i]);
                bool match = diff <= tolerance;
                allMatch &= match;
                string status = match ? "✓" : "✗";
                Log($"    [{i}]: Expected={expected[i]:F6}, Actual={actual[i]:F6}, Diff={diff:F6} {status}");
            }
            return allMatch;
        }

        private bool CompareMatrices(string name, float[,] expected, float[,] actual, float tolerance)
        {
            Log($"\n  {name}:");
            int rows = expected.GetLength(0);
            int cols = expected.GetLength(1);
            bool allMatch = true;
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    float diff = System.Math.Abs(expected[i, j] - actual[i, j]);
                    bool match = diff <= tolerance;
                    allMatch &= match;
                    string status = match ? "✓" : "✗";
                    Log($"    [{i},{j}]: Expected={expected[i,j]:F6}, Actual={actual[i,j]:F6}, Diff={diff:F6} {status}");
                }
            }
            return allMatch;
        }

        [TestMethod]
        public void Debug_TwoLayer_Gradients_Enhanced()
        {
            debugOutput.Clear();
            
            // Simplest possible 2-layer network: 2 inputs -> 2 hidden -> 2 outputs
            int inputDim = 2;
            int hiddenDim = 2;
            int outputDim = 2;
            float learningRate = 0.1f;

            // Fixed input and target for complete reproducibility
            float[] inputData = { 0.5f, 0.3f };
            float[] targetData = { 1.0f, 0.0f };  // One-hot for class 0

            // Fixed weights for Layer 1 (2x2 matrix + 2 biases)
            float[,] weights1Data = new float[,] {
                { 0.1f, 0.2f },  // Neuron 0 weights
                { 0.3f, 0.4f }   // Neuron 1 weights
            };
            float[] bias1Data = { 0.1f, 0.2f };

            // Fixed weights for Layer 2 (2x2 matrix + 2 biases)
            float[,] weights2Data = new float[,] {
                { 0.15f, 0.25f },  // Neuron 0 weights
                { 0.35f, 0.45f }   // Neuron 1 weights
            };
            float[] bias2Data = { 0.05f, 0.15f };

            LogSection(" FORWARD PASS ");
            
            // Log inputs
            LogSubSection("INPUT DATA");
            Log($"  Input X: {FormatArray(inputData)}");
            Log($"  Target:  {FormatArray(targetData)} (one-hot for class 0)");

            // ========== LAYER 1: WEIGHTED ==========
            LogSubSection("LAYER 1 - WEIGHTED COMPUTATION (Z1 = W1·X + B1)");
            
            // Show weights
            Log("  Weights W1:");
            Log(FormatMatrix(weights1Data));
            Log($"  Biases B1: {FormatArray(bias1Data)}");
            
            // Manual computation
            float[] manualZ1 = new float[hiddenDim];
            Log("\n  Manual computation step-by-step:");
            for (int i = 0; i < hiddenDim; i++)
            {
                manualZ1[i] = bias1Data[i];
                string calc = $"    Z1[{i}] = B1[{i}]";
                for (int j = 0; j < inputDim; j++)
                {
                    float product = weights1Data[i, j] * inputData[j];
                    manualZ1[i] += product;
                    calc += $" + W1[{i},{j}]*X[{j}]({weights1Data[i,j]}*{inputData[j]}={product:F6})";
                }
                calc += $" = {manualZ1[i]:F6}";
                Log(calc);
            }
            Log($"\n  Manual Z1 result: {FormatArray(manualZ1)}");

            // ========== RELU ACTIVATION ==========
            LogSubSection("RELU ACTIVATION (A1 = max(0, Z1))");
            
            float[] manualA1 = new float[hiddenDim];
            float[] reluDerivative = new float[hiddenDim];
            
            Log("  Computation:");
            for (int i = 0; i < hiddenDim; i++)
            {
                manualA1[i] = System.Math.Max(0, manualZ1[i]);
                reluDerivative[i] = manualZ1[i] > 0 ? 1 : 0;
                string status = manualZ1[i] > 0 ? "active" : "dead";
                Log($"    A1[{i}] = max(0, {manualZ1[i]:F6}) = {manualA1[i]:F6} ({status})");
            }
            Log($"\n  Manual A1 result: {FormatArray(manualA1)}");
            Log($"  ReLU derivative for backprop: {FormatArray(reluDerivative)}");

            // ========== LAYER 2: WEIGHTED ==========
            LogSubSection("LAYER 2 - WEIGHTED COMPUTATION (Z2 = W2·A1 + B2)");
            
            Log("  Weights W2:");
            Log(FormatMatrix(weights2Data));
            Log($"  Biases B2: {FormatArray(bias2Data)}");
            
            // Manual computation
            float[] manualZ2 = new float[outputDim];
            Log("\n  Manual computation step-by-step:");
            for (int i = 0; i < outputDim; i++)
            {
                manualZ2[i] = bias2Data[i];
                string calc = $"    Z2[{i}] = B2[{i}]";
                for (int j = 0; j < hiddenDim; j++)
                {
                    float product = weights2Data[i, j] * manualA1[j];
                    manualZ2[i] += product;
                    calc += $" + W2[{i},{j}]*A1[{j}]({weights2Data[i,j]}*{manualA1[j]:F6}={product:F6})";
                }
                calc += $" = {manualZ2[i]:F6}";
                Log(calc);
            }
            Log($"\n  Manual Z2 result: {FormatArray(manualZ2)}");

            // ========== SOFTMAX ==========
            LogSubSection("SOFTMAX (probabilities)");
            
            float maxZ2 = manualZ2.Max();
            float[] expZ2 = new float[outputDim];
            float sumExp = 0;
            
            Log($"  max(Z2) = {maxZ2:F6}");
            Log("  exp(Z2[i] - max):");
            for (int i = 0; i < outputDim; i++)
            {
                expZ2[i] = (float)System.Math.Exp(manualZ2[i] - maxZ2);
                Log($"    exp({manualZ2[i]:F6} - {maxZ2:F6}) = exp({manualZ2[i] - maxZ2:F6}) = {expZ2[i]:F6}");
                sumExp += expZ2[i];
            }
            Log($"  sum of exp = {sumExp:F6}");
            
            float[] manualSoftmax = new float[outputDim];
            Log("  softmax[i] = exp[i] / sum:");
            for (int i = 0; i < outputDim; i++)
            {
                manualSoftmax[i] = expZ2[i] / sumExp;
                Log($"    softmax[{i}] = {expZ2[i]:F6} / {sumExp:F6} = {manualSoftmax[i]:F6}");
            }

            // ========== LOSS ==========
            LogSubSection("CROSS-ENTROPY LOSS");
            
            float manualLoss = 0;
            Log("  Loss = -sum(target[i] * log(softmax[i]))");
            for (int i = 0; i < outputDim; i++)
            {
                if (targetData[i] > 0)
                {
                    float logProb = (float)System.Math.Log(manualSoftmax[i] + 1e-7);
                    manualLoss -= targetData[i] * logProb;
                    Log($"    -{targetData[i]} * log({manualSoftmax[i]:F6}) = -{targetData[i]} * {logProb:F6} = {-targetData[i] * logProb:F6}");
                }
            }
            Log($"\n  Manual Loss: {manualLoss:F6}");

            // ========== TORCHSHARP FORWARD ==========
            LogSubSection("TORCHSHARP FORWARD PASS");
            
            // Clone arrays because torch.from_array shares memory and TorchSharp will modify them
            using var torchWeights1 = torch.from_array((float[,])weights1Data.Clone());
            using var torchWeights2 = torch.from_array((float[,])weights2Data.Clone());
            
            var torchLinear1 = Linear(inputDim, hiddenDim);
            torchLinear1.weight = torchWeights1.AsParameter();
            torchLinear1.bias = torch.from_array((float[])bias1Data.Clone()).AsParameter();
            
            var torchLinear2 = Linear(hiddenDim, outputDim);
            torchLinear2.weight = torchWeights2.AsParameter();
            torchLinear2.bias = torch.from_array((float[])bias2Data.Clone()).AsParameter();

            float[,] input2d = new float[1, inputDim];
            for (int i = 0; i < inputDim; i++) input2d[0, i] = inputData[i];
            using var torchInput = torch.from_array(input2d);
            using var torchTarget = torch.from_array(new long[] { 0 });  // Class 0
            using var torchLossFn = CrossEntropyLoss(reduction: Reduction.Sum);

            // Forward
            var torchZ1 = torchLinear1.forward(torchInput);
            using var torchA1 = relu(torchZ1);
            var torchZ2 = torchLinear2.forward(torchA1);
            var torchLoss = torchLossFn.forward(torchZ2, torchTarget);

            // Extract values
            var torchZ1Data = torchZ1.cpu().data<float>().ToArray();
            var torchA1Data = torchA1.cpu().data<float>().ToArray();
            var torchZ2Data = torchZ2.cpu().data<float>().ToArray();
            var torchSoftmax = torchZ2.softmax(1).cpu().data<float>().ToArray();
            float torchLossValue = torchLoss.item<float>();

            Log($"  Torch Z1: {FormatArray(torchZ1Data)}");
            Log($"  Torch A1: {FormatArray(torchA1Data)}");
            Log($"  Torch Z2: {FormatArray(torchZ2Data)}");
            Log($"  Torch Softmax: {FormatArray(torchSoftmax)}");
            Log($"  Torch Loss: {torchLossValue:F6}");

            // ========== NEURALNETS FORWARD ==========
            LogSubSection("NEURALNETS FORWARD PASS");
            
            var weights1 = MatrixFactory.CreateMatrix(weights1Data);
            var biases1 = new AvxColumnVector(bias1Data);
            var inputShape = new InputOutputShape(1, inputDim, 1, 1);
            var weightedLayer1 = new WeightedLayer(inputShape, hiddenDim, weights1, biases1);
            var reluLayer1 = new ReLUActivaction();

            var weights2 = MatrixFactory.CreateMatrix(weights2Data);
            var biases2 = new AvxColumnVector(bias2Data);
            var weightedLayer2 = new WeightedLayer(weightedLayer1.OutputShape, outputDim, weights2, biases2);

            // Forward with debug
            var nnInput = new AvxColumnVector(inputData);
            
            Log("  Layer 1 FeedForward:");
            var nnZ1Tensor = weightedLayer1.FeedFoward(nnInput.ToTensor());
            Log($"    Z1: [{weightedLayer1.Y[0]:F6}, {weightedLayer1.Y[1]:F6}]");
            
            Log("  ReLU FeedForward:");
            var nnA1Tensor = reluLayer1.FeedFoward(nnZ1Tensor);
            Log($"    A1: [{reluLayer1.LastActivation.ToColumnVector()[0]:F6}, {reluLayer1.LastActivation.ToColumnVector()[1]:F6}]");
            
            Log("  Layer 2 FeedForward:");
            var nnZ2Tensor = weightedLayer2.FeedFoward(nnA1Tensor);
            Log($"    Z2: [{weightedLayer2.Y[0]:F6}, {weightedLayer2.Y[1]:F6}]");

            // Compute loss
            var nnTarget = new AvxColumnVector(targetData);
            var lossFunction = new CategoricalCrossEntropy();
            float nnLoss = lossFunction.ScalarLoss(nnTarget, weightedLayer2.Y);
            Log($"  NeuralNets Loss: {nnLoss:F6}");

            // ========== FORWARD COMPARISON ==========
            LogSection(" FORWARD PASS COMPARISON ");
            
            var mismatches = new List<string>();
            
            if (!CompareArrays("Z1 (Layer 1 pre-activation)", manualZ1, 
                new[] { weightedLayer1.Y[0], weightedLayer1.Y[1] }, Tolerance))
                mismatches.Add("Z1");
            
            if (!CompareArrays("Z2 (Layer 2 pre-activation)", manualZ2,
                new[] { weightedLayer2.Y[0], weightedLayer2.Y[1] }, Tolerance))
                mismatches.Add("Z2");
            
            if (!CompareValues("Loss", manualLoss, nnLoss, Tolerance))
                mismatches.Add("Loss");

            // ========== BACKWARD PASS ==========
            LogSection(" BACKWARD PASS ");

            LogSubSection("STEP 1: Initial Loss Gradient (dL/dZ2 = softmax - target)");
            float[] manualDL_dZ2 = new float[outputDim];
            Log("  Manual computation:");
            for (int i = 0; i < outputDim; i++)
            {
                manualDL_dZ2[i] = manualSoftmax[i] - targetData[i];
                Log($"    dL/dZ2[{i}] = {manualSoftmax[i]:F6} - {targetData[i]:F6} = {manualDL_dZ2[i]:F6}");
            }
            Log($"  Result: {FormatArray(manualDL_dZ2)}");

            // Torch backward
            torchLoss.backward();
            var torchGradW1 = torchLinear1.weight.grad.cpu().data<float>().ToArray();
            var torchGradB1 = torchLinear1.bias.grad.cpu().data<float>().ToArray();
            var torchGradW2 = torchLinear2.weight.grad.cpu().data<float>().ToArray();
            var torchGradB2 = torchLinear2.bias.grad.cpu().data<float>().ToArray();

            // NeuralNets backward
            var layers = new System.Collections.Generic.List<Layer> { weightedLayer1, reluLayer1, weightedLayer2 };
            var network = new GeneralFeedForwardANN(layers, learningRate, inputDim, outputDim, new CategoricalCrossEntropy());
            var trainingPair = new TrainingPair(nnInput.ToTensor(), nnTarget.ToTensor());
            var renderContext = new RenderContext(network, 1, null);
            renderContext.BackProp(trainingPair, weightedLayer2.Y);

            var nnGradW1 = weightedLayer1.LastWeightGradient;
            var nnGradB1 = weightedLayer1.LastBiasGradient;
            var nnGradW2 = weightedLayer2.LastWeightGradient;
            var nnGradB2 = weightedLayer2.LastBiasGradient;

            // Reshape for comparison
            float[,] torchGradW1_2D = new float[hiddenDim, inputDim];
            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < inputDim; j++)
                    torchGradW1_2D[i, j] = torchGradW1[i * inputDim + j];

            float[,] torchGradW2_2D = new float[outputDim, hiddenDim];
            for (int i = 0; i < outputDim; i++)
                for (int j = 0; j < hiddenDim; j++)
                    torchGradW2_2D[i, j] = torchGradW2[i * hiddenDim + j];

            float[] nnGradW1Arr = new float[hiddenDim * inputDim];
            for (int i = 0; i < hiddenDim; i++)
                for (int j = 0; j < inputDim; j++)
                    nnGradW1Arr[i * inputDim + j] = nnGradW1[i, j];

            float[] nnGradB1Arr = new float[hiddenDim];
            for (int i = 0; i < hiddenDim; i++) nnGradB1Arr[i] = nnGradB1[i];

            float[] nnGradW2Arr = new float[outputDim * hiddenDim];
            for (int i = 0; i < outputDim; i++)
                for (int j = 0; j < hiddenDim; j++)
                    nnGradW2Arr[i * hiddenDim + j] = nnGradW2[i, j];

            float[] nnGradB2Arr = new float[outputDim];
            for (int i = 0; i < outputDim; i++) nnGradB2Arr[i] = nnGradB2[i];

            LogSubSection("STEP 2: Layer 2 Weight Gradients (dL/dW2 = dL/dZ2 · A1^T)");
            Log("  Manual outer product computation:");
            float[,] manualDL_dW2 = new float[outputDim, hiddenDim];
            for (int i = 0; i < outputDim; i++)
            {
                for (int j = 0; j < hiddenDim; j++)
                {
                    manualDL_dW2[i, j] = manualDL_dZ2[i] * manualA1[j];
                    Log($"    dL/dW2[{i},{j}] = dL/dZ2[{i}]*A1[{j}] = {manualDL_dZ2[i]:F6}*{manualA1[j]:F6} = {manualDL_dW2[i,j]:F6}");
                }
            }
            Log($"  Manual dL/dW2:\n{FormatMatrix(manualDL_dW2)}");
            if (!CompareArrays("Torch dL/dW2", torchGradW2, nnGradW2Arr, Tolerance * 10))
                mismatches.Add("dL/dW2");

            LogSubSection("STEP 3: Layer 2 Bias Gradients (dL/dB2 = dL/dZ2)");
            Log($"  Manual dL/dB2: {FormatArray(manualDL_dZ2)}");
            if (!CompareArrays("Torch dL/dB2", torchGradB2, nnGradB2Arr, Tolerance * 10))
                mismatches.Add("dL/dB2");

            LogSubSection("STEP 4: Backprop through Layer 2 (dL/dA1 = W2^T · dL/dZ2)");
            float[] manualDL_dA1 = new float[hiddenDim];
            Log("  Manual computation:");
            for (int j = 0; j < hiddenDim; j++)
            {
                string calc = $"    dL/dA1[{j}] = ";
                for (int i = 0; i < outputDim; i++)
                {
                    float product = weights2Data[i, j] * manualDL_dZ2[i];
                    manualDL_dA1[j] += product;
                    calc += $"W2[{i},{j}]*dL/dZ2[{i}]({weights2Data[i,j]}*{manualDL_dZ2[i]:F6}={product:F6})";
                    if (i < outputDim - 1) calc += " + ";
                }
                calc += $" = {manualDL_dA1[j]:F6}";
                Log(calc);
            }
            Log($"  Result: {FormatArray(manualDL_dA1)}");

            LogSubSection("STEP 5: Backprop through ReLU (dL/dZ1 = dL/dA1 · ReLU'(Z1))");
            float[] manualDL_dZ1 = new float[hiddenDim];
            Log("  Manual computation:");
            for (int i = 0; i < hiddenDim; i++)
            {
                manualDL_dZ1[i] = manualDL_dA1[i] * reluDerivative[i];
                Log($"    dL/dZ1[{i}] = dL/dA1[{i}]*ReLU'[{i}] = {manualDL_dA1[i]:F6}*{reluDerivative[i]} = {manualDL_dZ1[i]:F6}");
            }
            Log($"  Result: {FormatArray(manualDL_dZ1)}");

            LogSubSection("STEP 6: Layer 1 Weight Gradients (dL/dW1 = dL/dZ1 · X^T)");
            float[,] manualDL_dW1 = new float[hiddenDim, inputDim];
            Log("  Manual outer product computation:");
            for (int i = 0; i < hiddenDim; i++)
            {
                for (int j = 0; j < inputDim; j++)
                {
                    manualDL_dW1[i, j] = manualDL_dZ1[i] * inputData[j];
                    Log($"    dL/dW1[{i},{j}] = dL/dZ1[{i}]*X[{j}] = {manualDL_dZ1[i]:F6}*{inputData[j]:F6} = {manualDL_dW1[i,j]:F6}");
                }
            }
            Log($"  Manual dL/dW1:\n{FormatMatrix(manualDL_dW1)}");
            if (!CompareArrays("Torch dL/dW1", torchGradW1, nnGradW1Arr, Tolerance * 10))
                mismatches.Add("dL/dW1");

            LogSubSection("STEP 7: Layer 1 Bias Gradients (dL/dB1 = dL/dZ1)");
            Log($"  Manual dL/dB1: {FormatArray(manualDL_dZ1)}");
            if (!CompareArrays("Torch dL/dB1", torchGradB1, nnGradB1Arr, Tolerance * 10))
                mismatches.Add("dL/dB1");

            // ========== FINAL SUMMARY ==========
            LogSection(" SUMMARY ");
            Log($"Total mismatches found: {mismatches.Count}");
            if (mismatches.Count > 0)
            {
                Log($"Mismatched components: {string.Join(", ", mismatches)}");
                Log("\n✗ TEST FAILED - See detailed output above for mismatch locations");
                Assert.Fail($"Gradient mismatches found in: {string.Join(", ", mismatches)}\n\nDetailed debug output:\n{debugOutput}");
            }
            else
            {
                Log("\n✓ ALL TESTS PASSED - All gradients match between TorchSharp and NeuralNets!");
            }
        }
    }
}
