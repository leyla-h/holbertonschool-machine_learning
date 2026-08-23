# BTC Forecasting

This project builds an RNN that uses the past 24 hours of BTC price
data to forecast the closing price of the following hour, using the
raw Coinbase and Bitstamp per-minute datasets.

## Files

* `preprocess_data.py` — loads, cleans, resamples, and normalizes the
  raw datasets, then saves the result to `btc_data.npz`.
* `forecast_btc.py` — loads the preprocessed data, builds a
  `tf.data.Dataset` of sliding 24-hour windows, and trains/validates
  an LSTM model to predict the following hour's closing price.

## Preprocessing decisions

**Are all of the data points useful?**
No. The raw data is a per-minute time window; a minute with no trades
is recorded as a row of `NaN`s rather than being omitted. Since a
minute without trades didn't move the market, the price columns are
forward-filled (the last traded price is carried over) instead of
dropped, and the volume columns are filled with 0. Dropping the rows
outright would silently compress time and misalign the series with
real elapsed hours, which matters once we resample to an hourly
resolution and need consecutive, evenly-spaced hours.

**Are all of the data features useful?**
Not all seven columns carry equally useful signal for an hourly
close-price forecast. `Weighted_Price` and `Volume_(Currency)` are
largely redundant with `Close` and `Volume_(BTC)` (they're derived
from the same underlying trades), so they're dropped to keep the
feature set small and avoid feeding the model near-duplicate inputs.
The final feature set kept per hour is `Open`, `High`, `Low`, `Close`,
and `Volume_(BTC)` — enough to describe the shape of each hour's
trading activity without redundant columns. The raw `Timestamp` is
used only to index/align the data and is not fed to the model as a
feature.

**Should you rescale the data?**
Yes. Prices are on the order of thousands of USD while `Volume_(BTC)`
is much smaller, and neural nets (especially RNNs using `tanh`/
`sigmoid` gates) train far more reliably when inputs are on a similar
scale. Each column is standardized (zero mean, unit variance).
Importantly, the mean and standard deviation are computed **only**
from the training split and then applied to the validation and test
splits, to avoid leaking information about the future into training.

**Is the current time window relevant?**
The task asks for hourly granularity ("the past 24 hours... the
following hour"), but the raw data is per-minute. The per-minute data
is first cleaned, then resampled into hourly bars: `Open` is the
first per-minute open in the hour, `High`/`Low` are the max/min over
the hour, `Close` is the last per-minute close in the hour, and
`Volume_(BTC)` is summed over the hour. This turns ~1440 minutes of
history into 24 hourly bars per day, matching what the model is
actually asked to predict from.

**How should you save this preprocessed data?**
The cleaned, resampled, normalized data is split chronologically into
train / validation / test sets (never shuffled, since this is a time
series and shuffling would leak future information into the past)
and saved as a single compressed `.npz` file (`btc_data.npz`)
containing the three arrays, the column names, and the normalization
statistics (`mean`, `std`) needed to invert predictions back to USD
later. This avoids reprocessing the raw CSVs (which are large and
slow to parse) every time the model is trained or re-evaluated.

## Model

`forecast_btc.py` builds sliding windows of the last 24 hours (using
`tf.data.Dataset.window`) to predict the next hour's normalized
`Close` price. The model itself is a stacked LSTM:

```
Input (24, 5)
  -> LSTM(64, return_sequences=True)
  -> LSTM(32)
  -> Dense(1)
```

It's compiled with the Adam optimizer and mean-squared error (MSE)
loss, as required, with mean absolute error (MAE) tracked as an
additional, more interpretable metric. The model is trained on the
training windows, monitored against the validation windows each
epoch, and finally scored on the held-out test windows.
