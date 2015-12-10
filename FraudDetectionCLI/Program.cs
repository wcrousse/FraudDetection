using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Encog.App.Analyst.Util;
using Encog.ML;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Data.Versatile;
using Encog.ML.Data.Versatile.Columns;
using Encog.ML.Data.Versatile.Sources;
using Encog.ML.Factory;
using Encog.ML.Model;
using Encog.ML.Train;
using Encog;
using Encog.Util.CSV;
using Encog.Util.Simple;

namespace FraudDetectionCLI
{
    class Program
    {
        static void Main(string[] args)
        {
            string fileName = "C:\\incentives_dataset.csv";
            var format = new CSVFormat(',', ',');
            IVersatileDataSource source = new CSVDataSource(fileName, false, format);
            var data = new VersatileMLDataSet(source);
            data.NormHelper.Format = format;
            data.DefineSourceColumn("id",0, ColumnType.Ignore);
            data.DefineSourceColumn("ip", 1, ColumnType.Nominal);
            data.DefineSourceColumn("dateTime", 2, ColumnType.Nominal);
            data.DefineSourceColumn("businessName", 3, ColumnType.Nominal);
            data.DefineSourceColumn("totalCharged", 4, ColumnType.Continuous);
            data.DefineSourceColumn("itemAmmount", 5, ColumnType.Continuous);
            data.DefineSourceColumn("email", 6, ColumnType.Nominal);
            data.DefineSourceColumn("brandName", 7, ColumnType.Nominal);
            data.DefineSourceColumn("orderStatus", 8, ColumnType.Nominal);
            ColumnDefinition outputColumn = data.DefineSourceColumn("isFraud", 9, ColumnType.Nominal);

            data.Analyze();

            data.DefineSingleOutputOthersInput(outputColumn);
            var model = new EncogModel(data);
            
            model.SelectMethod(data, MLMethodFactory.TypeFeedforward);
            model.Report = new ConsoleStatusReportable();

            data.Normalize();

            model.HoldBackValidation(0.3, true, DateTime.Now.Millisecond);// DateTime.Now.Millisecond);
            model.SelectTrainingType(data);
            var bestMethod = (IMLRegression) model.Crossvalidate(5, true);

            Console.WriteLine("Training error: " + EncogUtility.CalculateRegressionError(bestMethod, model.TrainingDataset));
            Console.WriteLine("Validation Error: " + EncogUtility.CalculateRegressionError(bestMethod, model.ValidationDataset));
            NormalizationHelper helper = data.NormHelper;
            Console.WriteLine(helper.ToString());
            Console.WriteLine("Final Model: " + bestMethod);
            
            Console.ReadLine();

            var csv = new ReadCSV(fileName, false, format);
            var line = new string[9];
            IMLData input = helper.AllocateInputVector();
            while (csv.Next())
            {
                var result = new StringBuilder();
                for (int i = 0; i < 9; i++)
                {
                        line[i] = csv.Get(i);            
                    
                }

                string correct = csv.Get(9);
                
                helper.NormalizeInputVector(
                    line, ((BasicMLData) input).Data, false);
                IMLData output = bestMethod.Compute(input);
                string isFraud = (helper.DenormalizeOutputVectorToString(output)[0] == "1" ? "Fraud" : "Okay");
                result.Append(line);
                result.Append(" -> predicted: ");
                result.Append(isFraud);
                result.Append(" (correct: ");
                result.Append(correct);
                result.Append(")");
                Console.WriteLine(result.ToString());
                
            }
        }
    }
}
