import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque


"""
This is a simple anomaly detection algorithm that uses the K-Nearest Neighbors (KNN) algorithm to detect anomalies in a real-time data stream.
Author : Alpay Avşar
Date : 05.09.2024
"""

def generate_data(n=1000, noise_level=0.1, prob=0.01):

    """
    Simulates a continuous data stream with patterns, noise, and occasional anomalies.

    n (int): Number of data points to generate.
    noise (float): Standard deviation of the Gaussian noise.
    prob (float): Probability of introducing an anomaly in the data stream.

    Yields:
    float: A new data point in the stream.
    """
    series = []
    for i in range(n):
        seasonal_component = 10 * np.sin(i * 0.1) # Simulate seasonal patterns
        noise = np.random.normal(0, noise_level)  # Add Gaussian noise
        data_point = seasonal_component + noise
        if np.random.rand() < prob:
            data_point += np.random.normal(20, 5)  # Large anomaly

        series.append(data_point)
        yield data_point
        time.sleep(0.01)   # simulating a streaming data source

class KNN_Anomaly_detector:

    """
    A KNN-based anomaly detector for real-time data streams , i have taken data mining class last semester and i have learned about KNN algorithm for anomaly detection
    and decided to use it on this project
    """
    def __init__(self, k=3, window_size=100, threshold=1.5):
        ##Initializes the KNN anomaly detector.
        self.k = k # Number of nearest neighbors to consider
        self.window_size = window_size # Size of the sliding window
        self.threshold = threshold # Threshold for anomaly detection
        self.window = deque(maxlen=window_size)  # Initialize the sliding window

    def add_point(self,point):
        """
        Adds a new data point to the window and checks if it's an anomaly.
        point (float): The new data point.
        Returns:
        bool: True if the point is an anomaly, False otherwise.
        """

        # If the window isn't full yet, just add the point and return
        if len(self.window) < self.window_size:
            self.window.append(point)
            return False  # Not enough data yet to detect anomalies
        
        # Add the new point to the window
        self.window.append(point)

         # Check if this point is an anomaly based on KNN distance
        return self.is_anomaly(point)
    
    def is_anomaly(self, point):
        """
        Determines if a given point is an anomaly based on its distance to neighbors using euclidean distance.

        point (float): The data point to check.

        Returns:
        bool: True if the point is an anomaly, False otherwise.
        """
        # Calculate the distances to the k nearest neighbors
        distances = [abs(point - x) for x in self.window]
        distances.sort()
        avg_distance = np.mean(distances[:self.k])  # Average distance to the k-nearest neighbors
        # If the average distance exceeds the threshold, it's considered an anomaly
        return avg_distance > self.threshold
    
def plot_stream(stream,detector):
     
    """
    Plots the real-time data stream and marks anomalies as they are detected.

    stream (generator): The data stream generator.
    detector (KNNAnomalyDetector): The KNN anomaly detector.
    """

    data = []
    anomalies = []

    plt.ion()  # Turn on interactive mode
    figure , ax = plt.subplots()

    for point in stream:
        data.append(point)
        is_anomaly = detector.add_point(point)
        if is_anomaly:
            anomalies.append(len(data) - 1)  # Save the index of the anomaly

        # Clear and update the plot
        ax.clear()
        ax.plot(data, label='Data Stream')  # Plot the entire data stream

        if anomalies:
            ax.plot(anomalies, [data[i] for i in anomalies], 'ro', label='Anomalies')
        
        plt.legend() # Add a legend to the plot
        plt.pause(0.01)  # Pause to allow for smooth animation

    plt.ioff()  # Disable interactive mode
    plt.show()  # Display the final plot

if __name__ == "__main__":
    # Initialize the KNN anomaly detector
    detector = KNN_Anomaly_detector(k=3, window_size=100, threshold=1.5)
    
    # Generate a continuous data stream with seasonal patterns, noise, and occasional anomalies
    data_stream = generate_data(n=1000, noise_level=0.1, prob=0.01)
    
    # Visualize the data stream and detect anomalies in real-time
    plot_stream(data_stream, detector)

    

    