import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calc_mean_erp(trial_points, ecog_data):
    """
    Calculate the mean Event Related Potential (ERP) for finger movements based on trial points and ECoG data.

    Parameters:
        trial_points_csv (str): Path to the CSV file containing trial points with columns [start, peak, finger].
        ecog_data_csv (str): Path to the CSV file containing ECoG time series data.

    Returns:
        np.ndarray: A 5x1201 matrix containing the averaged brain response for each finger.
    """
    #Load the data

    #Ensure trial points are integers
    trial_points = trial_points.astype(int)

    #Initialize variables
    num_fingers = 5
    num_timepoints = 1201  #200ms before + 1ms start + 1000ms after
    fingers_erp_mean = np.zeros((num_fingers, num_timepoints))

    #Process each finger
    for finger in range(1, num_fingers + 1):
        #Filter trials for the current finger
        finger_trials = trial_points[trial_points.iloc[:, 2] == finger]

        #Extract and average data for this finger
        finger_data = []
        for _, row in finger_trials.iterrows():
            start_idx = row[0] - 200  # 200ms before
            end_idx = row[0] + 1000  # 1000ms after
            if start_idx >= 0 and end_idx < len(ecog_data):
                finger_data.append(ecog_data[start_idx:end_idx + 1])

        #Calculate the mean ERP for the current finger
        if finger_data:
            fingers_erp_mean[finger - 1, :] = np.mean(finger_data, axis=0)

    #Plot the averaged brain responses
    time_axis = np.arange(-200, 1001)  #Time in ms
    plt.figure(figsize=(12, 8))
    for finger in range(1, num_fingers + 1):
        plt.plot(time_axis, fingers_erp_mean[finger - 1, :], label=f'Finger {finger}')
    plt.xlabel('Time (ms)')
    plt.ylabel('Brain Response (a.u.)')
    plt.title('Averaged Brain Response for Each Finger')
    plt.legend()
    plt.grid()
    plt.show()

    return fingers_erp_mean
trial_points_csv = r"C:\Users\barel\Downloads\mini_project_2_data\mini_project_2_data\events_file_ordered.csv"
ecog_data_csv = r"C:\Users\barel\Downloads\mini_project_2_data\mini_project_2_data\brain_data_channel_one.csv"
trial_points = pd.read_csv(trial_points_csv)
ecog_data = pd.read_csv(ecog_data_csv, header=None).iloc[:, 0].values  #Assuming single column of data
#Example usage (update paths as needed)
fingers_erp_mean = calc_mean_erp(trial_points, ecog_data)
