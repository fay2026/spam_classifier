import pandas as pd
from sklearn.model_selection import train_test_split

# Define path to the existing SMSSpamCollection file
data_file_path = "SMSSpamCollection"

# Load data into pandas DataFrame
df = pd.read_csv(
    data_file_path, sep="\t", header=None, names=["Label", "Text"]
)

# Display first few rows
print("Dataset preview:")
print(df.head())

# Display basic statistics
print("\nDataset information:")
print(f"Total samples: {len(df)}")
print(f"Spam samples: {len(df[df['Label'] == 'spam'])}")
print(f"Ham samples: {len(df[df['Label'] == 'ham'])}")

# Split into temporary train (80%) and test (20%)
temp_train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])

# Split temporary train into actual train (70%) and validation (10%)
# Since temp_train_df is 80% of the original data, we need to use test_size=0.125 (10%/80% = 0.125) 
# to get 10% of the original data for validation
train_df, val_df = train_test_split(temp_train_df, test_size=0.125, random_state=42, stratify=temp_train_df['Label'])

# Print split information
print("\nData split information:")
print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
print(f"Testing set: {len(test_df)} samples ({len(test_df)/len(df):.1%})")

# Verify class distribution across splits
print("\nClass distribution:")
print(f"Full dataset - Spam: {len(df[df['Label'] == 'spam'])/len(df):.1%}, Ham: {len(df[df['Label'] == 'ham'])/len(df):.1%}")
print(f"Training set - Spam: {len(train_df[train_df['Label'] == 'spam'])/len(train_df):.1%}, Ham: {len(train_df[train_df['Label'] == 'ham'])/len(train_df):.1%}")
print(f"Validation set - Spam: {len(val_df[val_df['Label'] == 'spam'])/len(val_df):.1%}, Ham: {len(val_df[val_df['Label'] == 'ham'])/len(val_df):.1%}")
print(f"Testing set - Spam: {len(test_df[test_df['Label'] == 'spam'])/len(test_df):.1%}, Ham: {len(test_df[test_df['Label'] == 'ham'])/len(test_df):.1%}")
