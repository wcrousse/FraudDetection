using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSVTransformer
{
    class Program
    {
        static void Main(string[] args)
        {
            TransformCsv("C:\\Users\\wrousse\\tsys.csv");
        }

        private static string TransformCsv(string filePath)
        {
            using (var sr = new StreamReader(filePath))
            {
                var trainingCSV = new List<string>();
                var testCSV = new List<string>();
                string line;
                var rand = new Random();
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
                    if ((csvLine[9] == "0" && rand.NextDouble() < .98) || (csvLine[9] == "1" && rand.NextDouble() < .9))
                    {
                        testCSV.Add(String.Join(",", csvLine));
                    }
                    else
                    {
                        trainingCSV.Add(String.Join(",", csvLine));
                    }
                }

                var newFilePath = filePath.Replace(".csv", "") + "-testData.csv";
                File.WriteAllLines(newFilePath, testCSV.ToArray());
                newFilePath = filePath.Replace(".csv", "") + "-trainingData.csv";
                File.WriteAllLines(newFilePath, trainingCSV.ToArray());
                return newFilePath;
            }
        }
    }
}
