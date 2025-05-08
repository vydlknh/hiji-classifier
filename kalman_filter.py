import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

positions = ['back', 'side', 'neck']
results = []

# impacts for 6 and 8
impacts_back_side = pd.to_datetime([
  "2025-02-21 19:50:09.420000076+00:00",
  "2025-02-21 19:50:11.160000086+00:00",
  "2025-02-21 19:50:13.109999895+00:00",
  "2025-02-21 19:50:15.049999952+00:00",
  "2025-02-21 19:50:16.789999962+00:00",
  "2025-02-21 19:50:18.529999970+00:00",
  "2025-02-21 19:50:20.970000029+00:00",
  "2025-02-21 19:50:22.779999971+00:00",
  "2025-02-21 19:50:25.279999971+00:00"
])

# impacts for 10
impact_neck = pd.to_datetime([
  "2025-02-21 20:04:33.470000029+00:00",
  "2025-02-21 20:04:35.680000067+00:00",
  "2025-02-21 20:04:37.609999895+00:00",
  "2025-02-21 20:04:39.940000057+00:00",
  "2025-02-21 20:04:41.980000019+00:00",
  "2025-02-21 20:04:44.130000114+00:00",
  "2025-02-21 20:04:45.819999933+00:00"
])

impact_3_2 = pd.to_datetime([
  "2025-03-03 04:17:38.000000000+00:00",
  "2025-03-03 04:17:40.000000000+00:00",
  "2025-03-03 04:17:41.500000000+00:00",
  "2025-03-03 04:17:43.000000000+00:00",
  "2025-03-03 04:17:44.000000000+00:00",
  "2025-03-03 04:17:46.000000000+00:00",
  "2025-03-03 04:17:48.000000000+00:00",
  "2025-03-03 04:17:49.500000000+00:00",
  "2025-03-03 04:17:51.000000000+00:00",
  "2025-03-03 04:17:52.000000000+00:00",
])

impact_3_11 = pd.to_datetime([
  "2025-03-11 22:22:27.000000000+00:00"
  "2025-03-11 22:22:29.000000000+00:00"
  "2025-03-11 22:22:30.500000000+00:00"
  "2025-03-11 22:22:33.000000000+00:00"
  "2025-03-11 22:22:34.500000000+00:00"
  "2025-03-11 22:22:36.500000000+00:00"
  "2025-03-11 22:22:38.500000000+00:00"
  "2025-03-11 22:22:40.000000000+00:00"
  "2025-03-11 22:22:42.000000000+00:00"
  "2025-03-11 22:22:44.000000000+00:00"
])

impact_4_22 = pd.to_datetime([
  "2025-04-22 23:32:10.596000+00:00"
  "2025-04-22 23:32:12.098000+00:00"
  "2025-04-22 23:32:13.760000+00:00"
  "2025-04-22 23:32:15.341000+00:00"
  "2025-04-22 23:32:16.922000+00:00"
  "2025-04-22 23:32:17.650000+00:00"
  "2025-04-22 23:32:17.650000+00:00"
  "2025-04-22 23:32:17.650000+00:00"
  "2025-04-22 23:32:17.650000+00:00"
  "2025-04-22 23:32:17.650000+00:00"])
impact_times = []

data = ["3-2", "3-11"]

# function to match true impact with nearest pred impact
def match_impacts(pred_times, true_times, tolerance=0.15):
  matched_true = set()
  matched_pred = set()

  for i, true_t in enumerate(true_times):
      for j, pred_t in enumerate(pred_times):
          if abs((true_t - pred_t).total_seconds()) <= tolerance:
              matched_true.add(i)
              matched_pred.add(j)
              break
  return len(matched_true), len(pred_times) - len(matched_pred), len(true_times) - len(matched_true)

# function to group impacts into clusters
def group_impacts(times, window=0.15):
  if len(times) == 0:
    return []
  grouped = [times[0]]
  for t in times[1:]:
    if (t - grouped[-1]).total_seconds() > window:
      grouped.append(t)
  return grouped

# for dataset in data:
#   df = pd.read_csv(f'data/raw/{dataset}.csv')
for pos in positions:
  df = pd.read_csv(f'data/raw/{pos}.csv')

  df['isoTimestamp'] = pd.to_datetime(df['isoTimestamp'])
  df = df.sort_values(by='isoTimestamp')
  df = df[df['isoTimestamp'].dt.date == pd.to_datetime('2025-02-21').date()]

  if pos == 'back' or pos == 'side':
    impact_times = impacts_back_side 
  else:
    impact_times = impact_neck

  # if dataset == "3-2":
  #   df = df[df['isoTimestamp'].dt.date == pd.to_datetime('2025-03-03').date()]
  #   impact_times = impact_3_2
  # else:
  #   df = df[df['isoTimestamp'].dt.date == pd.to_datetime('2025-03-11').date()]
  #   impact_times = impact_3_11
  
  start_time = impact_times.min() - pd.Timedelta(seconds=1)
  end_time = impact_times.max() + pd.Timedelta(seconds=1)
  df = df[(df['isoTimestamp'] >= start_time) & (df['isoTimestamp'] <= end_time)]

  # state transition matrix
  dt = 0.01  # 10ms
  F = np.array([
      [1, dt, 0.5*dt**2, 0, 0, 0],
      [0, 1, dt, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, dt, 0.5*dt**2],
      [0, 0, 0, 0, 1, dt],
      [0, 0, 0, 0, 0, 1]
  ])

  # observation matrix
  H = np.eye(6)

  # initial state
  initial_state = np.zeros(6)

  # initial covariance (uncertainty)
  initial_covariance = np.eye(6) * 0.1

  # process noise (how much we expect state to vary)
  Q = np.eye(6) * 0.05

  # measurement noise
  R = np.diag([0.05]*6)

  kf = KalmanFilter(
      transition_matrices=F,
      observation_matrices=H,
      initial_state_mean=initial_state,
      initial_state_covariance=initial_covariance,
      transition_covariance=Q,
      observation_covariance=R
  )

  observations = df[['accelX', 'accelY', 'accelZ', 'rateX', 'rateY', 'rateZ']].values
  timestamps = pd.to_datetime(df['isoTimestamp'])

  filtered_state_means, filtered_state_covariances = kf.filter(observations)

  # calculate innovation (difference between predicted and actual)
  predicted_observations = np.dot(H, filtered_state_means.T).T
  innovations = observations - predicted_observations

  # calculate innovation covariance
  S = np.zeros((len(observations), H.shape[0], H.shape[0]))
  for i in range(len(observations)):
      S[i] = np.dot(H, np.dot(filtered_state_covariances[i], H.T)) + R

  # calculate mahalanobis distance for anomaly detection
  mahalanobis_dist = np.zeros(len(observations))
  for i in range(len(observations)):
      mahalanobis_dist[i] = np.sqrt(np.dot(innovations[i], np.dot(np.linalg.inv(S[i]), innovations[i])))

  # use rolling window to detect sustained spikes
  rolling_avg = pd.Series(mahalanobis_dist).rolling(window=3, center=True).mean()
  rolling_avg.index = df.index
  impact_threshold = rolling_avg.mean() + 2 * rolling_avg.std()
  potential_impacts = (rolling_avg > impact_threshold)
  true_impacts = []
  if pos == 'back' or pos == 'side':
    true_impacts = impacts_back_side
  else:
    true_impacts = impact_neck
  # if dataset == "3-2":
  #   true_impacts = impact_3_2
  # else:
  #   true_impacts = impact_3_11

  df['true_impact'] = timestamps.apply(
    lambda t: any(abs((t - impact).total_seconds()) <= 0.1 for impact in true_impacts)
  )
  true_impacts = df['true_impact'].values
  
  # true_positives = np.sum(potential_impacts & true_impacts)
  # false_positives = np.sum(potential_impacts & ~true_impacts)
  # false_negatives = np.sum(~potential_impacts & true_impacts)

  pred_times = df.loc[potential_impacts, 'isoTimestamp']
  true_times = df.loc[df["true_impact"], 'isoTimestamp']

  grouped_pred_times = group_impacts(sorted(pred_times.tolist()), window=0.1)
  true_positives, false_positives, false_negatives = match_impacts(
      grouped_pred_times, true_times, tolerance=0.15)
  
  results.append({
      'position': pos,
      # 'dataset': dataset,
      'true_positives': true_positives,
      'false_positives': false_positives,
      'false_negatives': false_negatives
  })

  # plt.figure(figsize=(12, 4))
  # plt.plot(timestamps, rolling_avg, label='Mahalanobis')
  # rolling_avg.index = df.index
  # plt.scatter(timestamps[df['true_impact']], rolling_avg[df['true_impact']], color='red', label='True Impacts')
  # plt.axhline(y=impact_threshold, color='green', linestyle='--', label='Threshold')
  # plt.legend()
  # plt.title(f"Mahalanobis Distance - {pos} position")
  # # plt.title(f"Mahalanobis Distance - {dataset} dataset")
  # plt.xticks(rotation=45)
  # plt.tight_layout()
  # plt.show()

results_df = pd.DataFrame(results)
results_df['precision'] = results_df['true_positives'] / (
    results_df['true_positives'] + results_df['false_positives'])
results_df['recall'] = results_df['true_positives'] / (
    results_df['true_positives'] + results_df['false_negatives'])

print(results_df)

# # plt.figure(figsize=(12, 6))
# plt.subplot(3,1,1)
# plt.plot(timestamps, mahalanobis_dist, label='Mahalanobis Distance')
# # plt.scatter(timestamps[potential_impacts], mahalanobis_dist[potential_impacts], 
# #             color='red', label='Potential Impacts')
# plt.title('Impact Detection Using Kalman Filter')
# plt.xlabel('Time')
# plt.ylabel('Mahalanobis Distance')
# plt.tick_params('x', labelbottom=False)
# # plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.subplot(3,1,2)
# plt.plot(df['isoTimestamp'], df['accelX'], label='X-axis')
# plt.plot(df['isoTimestamp'], df['accelY'], label='Y-axis')
# plt.plot(df['isoTimestamp'], df['accelZ'], label='Z-axis')
# plt.xlabel('Time')
# plt.ylabel('Acceleration (g)')
# plt.tick_params('x', labelbottom=False)
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()

# plt.subplot(3,1,3)
# plt.plot(df['isoTimestamp'], df['rateX'], label='X-axis')
# plt.plot(df['isoTimestamp'], df['rateY'], label='Y-axis')
# plt.plot(df['isoTimestamp'], df['rateZ'], label='Z-axis')
# plt.xlabel('Time')
# plt.ylabel('Rotation Rate (deg/s)')
# plt.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

import time

start_time = time.time()

result = 1
for i in range(1, 100001):
  result *= i

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")