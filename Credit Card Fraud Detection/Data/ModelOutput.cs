﻿using Microsoft.ML.Data;

namespace CreditCardFraudDetection.Data
{
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Score { get; set; }
    }
}
