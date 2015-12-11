using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Text;
using Encog.ML;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Factory;
using Encog.ML.Model;
using Encog;
using Encog.Persist;
using Encog.Util.CSV;
using Encog.Util.Simple;

namespace FraudDetectionCLI
{
    class Program
    {

        static void Main(string[] args)
        {
            var minAccuracy = Double.Parse(ConfigurationManager.AppSettings["MinimumAccuracyThreshold"]);
            Console.WriteLine("Minimum accuracy set to {0}%", minAccuracy*100);
            var data = LoadAndAnalyzeTrainingData();

            var model = new EncogModel(data);
            NormalizationHelper helper = data.NormHelper;

            var accuracy = 0.0;
            var iteration = 0;
            while (accuracy < minAccuracy)
            {
                IMLRegression bestMethod;
                iteration++;
                Console.WriteLine("Beginning Iteration {0}.", iteration);
                model.SelectMethod(data, MLMethodFactory.TypeFeedforward);
                //model.Report = new ConsoleStatusReportable();

                data.Normalize();
                bool persistModel = ConfigurationManager.AppSettings["PersistModel"].ToLower() == "true";
                bool retrainModel = ConfigurationManager.AppSettings["RetrainModel"].ToLower() == "true";
                string modelFilePath = ConfigurationManager.AppSettings["ModelFilePath"];
                bool persistedModelExists = File.Exists(modelFilePath);

                if (retrainModel || !persistModel || !persistedModelExists)
                {
                    bestMethod = TrainModel(model, data, helper);
                }
                else
                {
                    bestMethod = LoadModel(modelFilePath);
                }

                Console.WriteLine("Model init complete.");
                accuracy = AnalyzeFileWithNetwork(CSVFormat.English, helper, bestMethod);
                Console.WriteLine("Accuracy of model: {0}%", accuracy*100);
                if (accuracy < minAccuracy)
                {
                    Console.WriteLine("Iteration failed. Repeating...");
                }
                else
                {
                    Console.WriteLine("Iteration Success.");
                    PersistModel(modelFilePath, bestMethod);
                }
            }

            Console.WriteLine("Analysis Complete. Press enter to exit.");
            Console.ReadLine();
        }

        private static void PersistModel(string modelFilePath, IMLRegression bestMethod)
        {
            Console.WriteLine("Saving model...");
            EncogDirectoryPersistence.SaveObject(new FileInfo(modelFilePath), bestMethod);
            Console.Write("Saved.");
        }

        private static IMLRegression LoadModel(string modelFilePath)
        {
            IMLRegression bestMethod;
            Console.WriteLine("Loading model from file...");
            bestMethod = (IMLRegression)
                EncogDirectoryPersistence.LoadObject(new FileInfo(modelFilePath));
            Console.WriteLine("Loaded " + bestMethod);
            return bestMethod;
        }

        private static IMLRegression TrainModel(EncogModel model, VersatileMLDataSet data, NormalizationHelper helper)
        {
            var verboseMode = ConfigurationManager.AppSettings["VerboseMode"].ToLower() == "true";
            if (verboseMode)
            {
                model.Report = new ConsoleStatusReportable();
            }
            else
            {
                SpinAnimation.Start();
            }
            IMLRegression bestMethod;
            model.HoldBackValidation(0.3, true, DateTime.Now.Millisecond);
            model.SelectTrainingType(data);
            var numberOfFolds = Int32.Parse(ConfigurationManager.AppSettings["TrainingFolds"]);
            bestMethod = (IMLRegression) model.Crossvalidate(numberOfFolds, false);

            SpinAnimation.Stop();
            Console.WriteLine("Training error: " +
                              EncogUtility.CalculateRegressionError(bestMethod, model.TrainingDataset));
            Console.WriteLine("Validation Error: " +
                              EncogUtility.CalculateRegressionError(bestMethod, model.ValidationDataset));
            //Console.WriteLine(helper.ToString());
            Console.WriteLine("Final Model: " + bestMethod);
            return bestMethod;
        }

        private static VersatileMLDataSet LoadAndAnalyzeTrainingData()
        {
            VersatileMLDataSet data;
            var trainingFilePath = ConfigurationManager.AppSettings["TrainingFilePath"];
            var transformedTrainingFilePath = trainingFilePath.Replace(".csv", "") + "-transformed.csv";
            if (!File.Exists(transformedTrainingFilePath))
            {
                TransformCsv(trainingFilePath, transformedTrainingFilePath);
            }
            var format = CSVFormat.English;
            IVersatileDataSource source = new CSVDataSource(transformedTrainingFilePath, false, format);
            data = new VersatileMLDataSet(source) {NormHelper = {Format = format}};
            data.DefineSourceColumn("id", 0, ColumnType.Ignore);
            data.DefineSourceColumn("ip1", 1, ColumnType.Continuous);
            data.DefineSourceColumn("ip2", 2, ColumnType.Continuous);
            data.DefineSourceColumn("ip3", 3, ColumnType.Continuous);
            data.DefineSourceColumn("ip4", 4, ColumnType.Continuous);
            data.DefineSourceColumn("seconds", 5, ColumnType.Continuous);
            data.DefineSourceColumn("dayOfWeek", 6, ColumnType.Continuous);
            data.DefineSourceColumn("day", 7, ColumnType.Continuous);
            data.DefineSourceColumn("month", 8, ColumnType.Continuous);
            data.DefineSourceColumn("businessName", 9, ColumnType.Nominal);
            data.DefineSourceColumn("totalCharged", 10, ColumnType.Continuous);
            data.DefineSourceColumn("itemAmmount", 11, ColumnType.Continuous);
            data.DefineSourceColumn("email", 12, ColumnType.Nominal);
            data.DefineSourceColumn("brandName", 13, ColumnType.Nominal);
            data.DefineSourceColumn("orderStatus", 14, ColumnType.Ignore);
            ColumnDefinition outputColumn = data.DefineSourceColumn("isFraud", 15, ColumnType.Nominal);
            data.DefineSingleOutputOthersInput(outputColumn);
            data.Analyze();
            return data;
        }

        private static double AnalyzeFileWithNetwork(CSVFormat format, NormalizationHelper helper, IMLRegression bestMethod)
        {
            Console.WriteLine("Beginning Analysis...");
            SpinAnimation.Start();
            var analysisFilePath = ConfigurationManager.AppSettings["AnalysisFilePath"];
            var transformedAnalysisFilePath = analysisFilePath.Replace(".csv", "") + "-transformed.csv";
            if (!File.Exists(transformedAnalysisFilePath))
            {
                TransformCsv(analysisFilePath, transformedAnalysisFilePath);
            }
            var csv = new ReadCSV(transformedAnalysisFilePath, false, format);
            var line = new string[15];
            IMLData input = helper.AllocateInputVector();
            var falsePositive = 0;
            var falseNegative = 0;
            var correctlyIdentifiedFraud = 0;
            var totalFraud = 0;
            var totalRows = 0;
            while (csv.Next())
            {
                var result = new StringBuilder();
                for (int i = 0; i < 15; i++)
                {
                    line[i] = csv.Get(i);
                }

                string correct = csv.Get(15);
                totalFraud += Int32.Parse(correct);
                totalRows++;

                helper.NormalizeInputVector(
                    line, ((BasicMLData) input).Data, true);
                IMLData output = bestMethod.Compute(input);
                var isFraud = helper.DenormalizeOutputVectorToString(output)[0] == "1";
                //if (isFraud)
                //{
                //    Console.WriteLine("Holy shit, {0} is fraud!", line[0]);
                //}
                //result.Append(line);
                //result.Append(" -> predicted: ");
                //result.Append(isFraud ? "Fraud" : "Okay");
                //result.Append(" (correct: ");
                //result.Append(correct);
                //result.Append(")");

                if (isFraud)
                {
                    if (correct == "1")
                    {
                        correctlyIdentifiedFraud++;
                    }
                    else
                    {
                        falsePositive++;
                    }
                    //Console.WriteLine(result);
                }
                else if (correct == "1")
                {
                    falseNegative++;
                }
            }
            SpinAnimation.Stop();
            Console.WriteLine("Total fraud rows: {0}, Correctly identified fraud: {1}, False Positives: {2}, FalseNegatives: {3}, Total Rows Analyzed: {4}",
                totalFraud, correctlyIdentifiedFraud, falsePositive, falseNegative, totalRows);
            double accuratelyFoundRows = totalRows - (falsePositive + falseNegative);
            double accuracy = totalRows != 0
                ? Convert.ToDouble(accuratelyFoundRows)/Convert.ToDouble(totalRows)
                : 1;
            return accuracy;
        }

        private static void TransformCsv(string inputFilePath, string outputFilePath)
        {
            using (var sr = new StreamReader(inputFilePath))
            {
                var newCsv = new List<string>();
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    var csvLine = line.Split(',');

                    if (csvLine[1].ToUpper() != "NULL")
                    {
                        csvLine[1] = csvLine[1].Replace('.', ',');
                    }
                    else
                    {
                        csvLine[1] = "0,0,0,0";
                    }

                    if (csvLine[2].ToUpper() != "NULL")
                    {
                        var dateTime = DateTime.Parse(csvLine[2]);
                        var secondsSinceMidnight = (dateTime - dateTime.Date).TotalSeconds;
                        var dayOfTheWeek = (int)dateTime.DayOfWeek;
                        var day = dateTime.Day;
                        var month = dateTime.Month;
                        csvLine[2] = String.Format("{0},{1},{2},{3}", secondsSinceMidnight, day, dayOfTheWeek, month);
                    }
                    else
                    {
                        csvLine[2] = "0,0,0,0";
                    }

                    newCsv.Add(String.Join(",", csvLine));
                }
                File.WriteAllLines(outputFilePath, newCsv.ToArray());
            }
        }
    }
}
