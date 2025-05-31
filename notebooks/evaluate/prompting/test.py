import csv

# Path to your CSV file
csv_file = "./prompt_c/average_scores_skenario_4_with_time.csv"

values = []
with open(csv_file, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        # Skip empty rows or rows with non-numeric values
        if row[0] and row[0] != "processing_time_sec" and row[0] != "":
            try:
                values.append(float(row[1]))
            except ValueError:
                pass

# Optional: exclude processing_time_sec if present
average = sum(values) / len(values)
print("prompt_c")
print(f"Average: {average}")
