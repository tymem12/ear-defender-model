import os
import csv
import os
from pydub import AudioSegment
from my_app.model_module.models.wav2vec.eval_metrics_DF import compute_eer 
import numpy as np
from sklearn.metrics import roc_curve
# Define the path to the folder containing your files
# folder_path = 'C:/Users/tymek/Videos/yt-dlp/real'

# folder_path = '../datasets/release_in_the_wild'

# Define the output CSV file
# output_csv = 'test_data_real.csv'

def create_test_file(output_csv, folder_path):

# Open the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['file', 'target'])
        
        # Iterate over the files in the specified folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Check if it's a file (not a subdirectory)
            if os.path.isfile(file_path):
                # Write the filename and target (which is always 1)
                writer.writerow([filename, 0])

    print(f"CSV file '{output_csv}' has been created with file names and target values.")




def cal_sum_time(folder_path):
    total_duration = 0  # Total duration in seconds

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load the audio file
        audio = AudioSegment.from_file(file_path)

        # Add the duration of the audio file to the total (duration is in milliseconds)
        total_duration += len(audio) / 1000  # Convert to seconds

    # Print the total duration in seconds
    print(f"Total length of all audio files: {total_duration} seconds")


def calulate_threshold_wav2vec():

    spoof_file_path = '../wav2vec_results_fake.csv'
    scores_spoof = []
    real_file_path = '../wav2vec_results_real.csv'
    scores_real = []
    spoof_file_path_1 = '../wav2vec_results_in_the_wild_fake.csv'
    real_file_path_1 = '../wav2vec_results_in_the_wild_real.csv'


    # folder = 'test_files'
    # print('wav2vec  deep voice')
    # spoof_file_path_1 = f'{folder}/deep_voice/wav2vec_deep_voice_fake.csv'
    # print(metrics.calculate_eer_from_scores_csv([], [f'{folder}/deep_voice/wav2vec_deep_voice_fake.csv']))
    output_csv = '../meso_results_in_the_wild_fake.csv'

    # with open(spoof_file_path, mode='r') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header row
    #     for row in reader:
    #         scores_spoof.append(float(row[2]))  # Append only the file name

    # with open(real_file_path, mode='r') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header row
    #     for row in reader:
    #         scores_real.append(float(row[2]))  # Append only the file name

    with open(spoof_file_path_1, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            scores_spoof.append(float(row[2]))  # Append only the file name
    with open(real_file_path_1, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            scores_real.append(float(row[2]))  # Append only the file name


    scores_spoof = np.array(scores_spoof)
    scores_real = np.array(scores_real)

    eer, threshold = compute_eer(scores_real, scores_spoof)
    print(f"EER: {eer}   threshold: {threshold}")

def calculate_eer_binary():
    spoof_file_path = '../meso_results_fake.csv'
    predictions = []
    real_file_path = '../meso_results_real.csv'
    labels = []
    spoof_file_path_2 = '../meso_results_in_the_wild_fake.csv'
    real_file_path_2 = '../meso_results_in_the_wild_real.csv'
    # with open(spoof_file_path, mode='r') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header row
    #     for row in reader:
    #         predictions.append(float(row[3]))  # Append only the file name
    #         labels.append(0)
    
    with open(spoof_file_path_2, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            predictions.append(float(row[3]))  # Append only the file name
            labels.append(0)

    # with open(real_file_path, mode='r') as file:
    #     reader = csv.reader(file)
    #     next(reader)  # Skip header row
    #     for row in reader:
    #         predictions.append(float(row[3]))  # Append only the file name
    #         labels.append(1)

    
    with open(real_file_path_2, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            predictions.append(float(row[3]))  # Append only the file name
            labels.append(1)
    # Convert inputs to numpy arrays for easier manipulation
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Calculate False Acceptance Rate (FAR) and False Rejection Rate (FRR)
    false_acceptances = np.sum((predictions == 1) & (labels == 0))  # Predicted 1 but actual is 0 (False Positive)
    false_rejections = np.sum((predictions == 0) & (labels == 1))   # Predicted 0 but actual is 1 (False Negative)
    
    total_negatives = np.sum(labels == 0)  # Actual negatives
    total_positives = np.sum(labels == 1)  # Actual positives
    
    # FAR is the proportion of false acceptances among all negatives
    if total_negatives > 0:
        far = false_acceptances / total_negatives
    else:
        far = 0.0  # Avoid division by zero
    
    # FRR is the proportion of false rejections among all positives
    if total_positives > 0:
        frr = false_rejections / total_positives
    else:
        frr = 0.0  # Avoid division by zero
    
    # EER occurs when FAR == FRR (approximately)
    eer = (far + frr) / 2
    print(f'EER{eer}')
    
    return eer

if __name__ == '__main__':
    # base_dir = 'C:/Users/tymek/Videos/yt-dlp/real'

    output_csv = '../meta.csv'
    # output_csv = 'test_data_real.csv'

    # cal_sum_time(base_dir)
    create_test_file('fake_audio_data.csv', )
    # calulate_threshold_wav2vec()
    # calculate_eer_binary()