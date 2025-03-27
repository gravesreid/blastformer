import os


dataset_directory = "/home/reid/projects/blast_waves/dataset_processed_2"

train_directory = os.path.join(dataset_directory, "train")
test_directory = os.path.join(dataset_directory, "test")
val_directory = os.path.join(dataset_directory, "val")

if not os.path.exists(train_directory):
    os.makedirs(train_directory)
if not os.path.exists(test_directory):
    os.makedirs(test_directory)
if not os.path.exists(val_directory):
    os.makedirs(val_directory)

# Split the dataset into train, test, and validation sets
# Take first 1050 samples for training
# take next 226 samples for testing
# take next 225 samples for validation

train_samples = 1100
test_samples = 200
val_samples = 200

# Move the simulation directories to the appropriate train, test, or val directories
for i in range(1,train_samples+1,1):
    os.rename(os.path.join(dataset_directory, f"{i}"), os.path.join(train_directory, f"{i}"))
    print(f"Moved {i} to train")
for i in range(train_samples+1, train_samples + test_samples + 1):
    os.rename(os.path.join(dataset_directory, f"{i}"), os.path.join(test_directory, f"{i}"))
    print(f"Moved {i} to test")
for i in range(train_samples + test_samples + 1, train_samples + test_samples + val_samples + 1):
    os.rename(os.path.join(dataset_directory, f"{i}"), os.path.join(val_directory, f"{i}"))
    print(f"Moved {i} to val")