﻿using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Text;
using Encog.Bot.Browse.Range;
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
            var trainingFilePath = ConfigurationManager.AppSettings["TrainingFilePath"];
            var transformedTrainingFilePath = trainingFilePath.Replace(".csv", "") + "-transformed.csv";
            if (!File.Exists(transformedTrainingFilePath))
            {
                TransformCsv(trainingFilePath, transformedTrainingFilePath);
            }
            var format = CSVFormat.English;
            IVersatileDataSource source = new CSVDataSource(transformedTrainingFilePath, false, format);
            var data = new VersatileMLDataSet(source) { NormHelper = { Format = format } };
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

            var model = new EncogModel(data);
            NormalizationHelper helper = data.NormHelper;
            IMLRegression bestMethod;

            model.SelectMethod(data, MLMethodFactory.TypeFeedforward);
            model.Report = new ConsoleStatusReportable();

            data.Normalize();
            bool persistModel = ConfigurationManager.AppSettings["PersistModel"].ToLower() == "true";
            bool retrainModel = ConfigurationManager.AppSettings["RetrainModel"].ToLower() == "true";
            if (retrainModel || !persistModel)
            {
                model.HoldBackValidation(0.3, true, DateTime.Now.Millisecond);
                model.SelectTrainingType(data);
                bestMethod = (IMLRegression) model.Crossvalidate(100, false);

                Console.WriteLine("Training error: " +
                                  EncogUtility.CalculateRegressionError(bestMethod, model.TrainingDataset));
                Console.WriteLine("Validation Error: " +
                                  EncogUtility.CalculateRegressionError(bestMethod, model.ValidationDataset));
                Console.WriteLine(helper.ToString());
                Console.WriteLine("Final Model: " + bestMethod);
                if (persistModel)
                {
                    Console.WriteLine("Saving model...");
                    EncogDirectoryPersistence.SaveObject(new FileInfo(ConfigurationManager.AppSettings["ModelFilePath"]), bestMethod);
                    Console.Write("Saved.");
                }
            }
            else
            {
                Console.WriteLine("Loading model from file...");
                bestMethod = (IMLRegression)
                    EncogDirectoryPersistence.LoadObject(new FileInfo(ConfigurationManager.AppSettings["ModelFilePath"]));
                Console.WriteLine("Loaded " + bestMethod);
            }
            Console.WriteLine("Model init finished, press enter to begin analysis.");
            Console.ReadLine();
            Console.WriteLine("Beginning Analysis...");
            var analysisFilePath = ConfigurationManager.AppSettings["AnalysisFilePath"];
            var transformedAnalysisFilePath = trainingFilePath.Replace(".csv", "") + "-transformed.csv";
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
            while (csv.Next())
            {
                var result = new StringBuilder();
                for (int i = 0; i < 15; i++)
                {
                    line[i] = csv.Get(i);

                }

                string correct = csv.Get(15);

                helper.NormalizeInputVector(
                    line, ((BasicMLData)input).Data, true);
                IMLData output = bestMethod.Compute(input);
                var isFraud = helper.DenormalizeOutputVectorToString(output)[0] == "1";
                if (isFraud)
                {
                    Console.WriteLine("Holy shit, {0} is fraud!", line[0]);
                }
                result.Append(line);
                result.Append(" -> predicted: ");
                result.Append(isFraud ? "Fraud" : "Okay");
                result.Append(" (correct: ");
                result.Append(correct);
                result.Append(")");

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
                    Console.WriteLine(result);
                }
                else if (correct == "1")
                {
                    falseNegative++;
                }
            }
            Console.WriteLine("Correctly identified fraud: {0} False Positives: {1} FalseNegatives: {2}", correctlyIdentifiedFraud, falsePositive, falseNegative);
            Console.ReadLine();
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
