# Read the data from the source file
with open('results/master_results_cnn.csv', 'r') as file1:
    data_to_append = file1.read()

# Append the data to the destination file
with open('results/master_results.csv', 'a') as file2:
    file2.write('\n')  # Starts on a new line
    file2.write(data_to_append)
