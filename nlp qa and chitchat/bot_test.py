import csv
import requests
import time

# Path to the input text file
input_file = 'small_bot_test.csv'

# Path to the output CSV file
output_file = 'small_bot_result.csv'

# URL to send GET requests to
url = 'http://localhost:3000/chat'

# Open the input file and output file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    # Write the header row
    writer.writerow(['id', 'answer', 'time'])

    # Iterate over each line in the input file
    for row in reader:
        # Send a GET request to the URL with the line as a parameter
        start_time = time.time()
        response = requests.get(url, params={'line': row['question']})
        end_time = time.time()

        # Write the output and time taken to the output file
        writer.writerow([row['id'], response.text, end_time - start_time])