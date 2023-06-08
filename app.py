#!/usr/local/bin/python3.7

from flask import Flask, render_template, make_response, request, jsonify, redirect, url_for, session, abort, \
    send_from_directory
import os
import asyncio
import concurrent.futures
import pandas as pd
import matplotlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle
import base64

matplotlib.use('Agg')

app = Flask(__name__)
app.debug = True


def ChartResult(percentages, data):
    import matplotlib.pyplot as plt
    colors = ['skyblue','orange']

    fig, ax = plt.subplots()
    ax.pie(percentages, labels=percentages.index, autopct='%1.1f%%', colors=colors)

    # Add a title
    ax.set_title('Percentage of Value Counts')

    # Adjust the layout to fit the labels properly
    plt.tight_layout()

    #save csv file for download
    file_path1 = os.path.join('static/', 'file.csv')
    data.to_csv(file_path1, index=False)
    image_path1 = os.path.join('static/images', 'pie.png')  # Save the image in the static folder
    plt.savefig(image_path1)
    plt.close(fig)
    return image_path1, file_path1

async def clusterPQ(csv, option1, option2):
    import matplotlib.pyplot as plt
    # Specify the path to your Excel file
    file_path = csv
    # Read the Excel file into a pandas DataFrame
    data = pd.read_excel(file_path)
    # data = await asyncio.to_thread(pd.read_excel, file_path)

    if (option1 and option2):
        p_q_data = data[['P', 'Q', 'Wind data', 'Solar data']]
        # Standardize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(p_q_data)
        # Load the K-means model
        with open('wind_solar_kmeans_model.pkl', 'rb') as file:
            kmeans_loaded = pickle.load(file)
        kmeans_loaded.fit(normalized_data)
        # Get the cluster labels
        cluster_labels = kmeans_loaded.labels_

        # Add cluster labels to the dataframe
        data['Cluster_PQ'] = cluster_labels

        # Label "0" as Stable and "1" as Unstable in the dataframe
        data['Stability_PQ'] = data['Cluster_PQ'].apply(
            lambda x: 'Stable' if x == 0 else 'Unstable')
        
        # Count the number of data points in each cluster
        Stability_counts = data['Stability_PQ'].value_counts()
        #plot pie chart
        percentages = Stability_counts / len(data) * 100
        
        result_file, result_pie = ChartResult(percentages, data)

        # Visualize the clusters
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels)
        plt.xlabel('P')
        plt.ylabel('Q')
        plt.title('Clustering Analysis on Columns P and Q with Wind and Solar Data')

    elif option1:
        p_q_data = data[['P', 'Q', 'Wind data']]
        # Standardize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(p_q_data)
        # Load the K-means model
        with open('wind_kmeans_model.pkl', 'rb') as file:
            kmeans_loaded = pickle.load(file)
        kmeans_loaded.fit(normalized_data)

        # Get the cluster labels
        cluster_labels = kmeans_loaded.labels_

        # Add cluster labels to the dataframe
        data['Cluster_PQ'] = cluster_labels

        # Label "0" as Stable and "1" as Unstable in the dataframe
        data['Stability_PQ'] = data['Cluster_PQ'].apply(
            lambda x: 'Stable' if x == 0 else 'Unstable')
        
        # Count the number of data points in each cluster
        Stability_counts = data['Stability_PQ'].value_counts()
        #plot pie chart
        percentages = Stability_counts / len(data) * 100
        
        result_file, result_pie = ChartResult(percentages, data)

        # Visualize the clusters
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels)
        plt.xlabel('P')
        plt.ylabel('Q')
        plt.title('Clustering Analysis on Columns P and Q with Wind Data')

    elif option2:
        p_q_data = data[['P', 'Q', 'Solar data']]
        # Standardize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(p_q_data)
        # Load the K-means model
        with open('solar_kmeans_model.pkl', 'rb') as file:
            kmeans_loaded = pickle.load(file)
        kmeans_loaded.fit(normalized_data)
        # Get the cluster labels
        cluster_labels = kmeans_loaded.labels_

        # Add cluster labels to the dataframe
        data['Cluster_PQ'] = cluster_labels

        # Label "0" as Stable and "1" as Unstable in the dataframe
        data['Stability_PQ'] = data['Cluster_PQ'].apply(
            lambda x: 'Stable' if x == 0 else 'Unstable')
        
        # Count the number of data points in each cluster
        Stability_counts = data['Stability_PQ'].value_counts()
        #plot pie chart
        percentages = Stability_counts / len(data) * 100
        
        result_file, result_pie = ChartResult(percentages, data)

        # Visualize the clusters
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels)
        plt.xlabel('P')
        plt.ylabel('Q')
        plt.title('Clustering Analysis on Columns P and Q with Solar Data')


    else:
        # Extract columns P and Q
        p_q_data = data[['P', 'Q']]
        # Standardize the data
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(p_q_data)
        # Load the K-means model
        with open('kmeans_model.pkl', 'rb') as file:
            kmeans_loaded = pickle.load(file)
        kmeans_loaded.fit(normalized_data)
        # Get the cluster labels
        cluster_labels = kmeans_loaded.labels_

        # Add cluster labels to the dataframe
        data['Cluster_PQ'] = cluster_labels

        # Label "0" as Stable and "1" as Unstable in the dataframe
        data['Stability_PQ'] = data['Cluster_PQ'].apply(
            lambda x: 'Stable' if x == 0 else 'Unstable')
        
        # Count the number of data points in each cluster
        Stability_counts = data['Stability_PQ'].value_counts()
        #plot pie chart
        percentages = Stability_counts / len(data) * 100
        
        result_file, result_pie = ChartResult(percentages, data)

        # Visualize the clusters
        plt.scatter(normalized_data[:, 0], normalized_data[:, 1], c=cluster_labels)
        plt.xlabel('P')
        plt.ylabel('Q')
        plt.title('Clustering Analysis on Columns P and Q')


    image_path = os.path.join('static/images', 'plot.png')  # Save the image in the static folder
    plt.savefig(image_path)
    plt.close()
    # plt.show()
    return image_path

def delete_previous_files():
    # Specify the file paths to be deleted
    file_paths = ['temp_file.xlsx', 'static/images/plot.png', 'static/images/pie.png', 'static/file.csv']

    # Delete the files if they exist
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            print('files deleted')
      
def get_image_base64(image_path):
    with open(image_path, 'rb') as file:
        image_data = file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    return image_base64

@app.route('/')
def resp():
    image_path = request.args.get('image')
    if image_path:
        return render_template('index.html', image=image_path)
    else:
        return render_template('index.html')

@app.route('/api/upload', methods=['POST'])
async def upload():
    delete_previous_files()
    if request.method == 'POST':
        file = request.files['file']  # Access the uploaded file
        option1 = request.form.get('option1', False)
        option2 = request.form.get('option2', False)
        if file:
            # Save the uploaded file to a temporary location
            file_path = 'temp_file.xlsx'
            file.save(file_path)

            def generate_image():
                return asyncio.run(clusterPQ(file_path, option1, option2))

            with concurrent.futures.ThreadPoolExecutor() as executor:
                loop = asyncio.get_event_loop()
                image_path = await loop.run_in_executor(executor, generate_image)

            # Encode the image as base64
            image_base64 = get_image_base64(image_path)

            # Return the image data as JSON response
            return jsonify({'image': image_base64})
        else:
            return 'No file uploaded'
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)