from datasets import load_dataset

# Load one sample from the validation split
sample = load_dataset('Ankit9263/medical-asr-en-split', split='validation')[0]

# View all available fields
print(sample.keys())

# View complete sample
for key, value in sample.items():
    print(f"\n--- {key} ---")
    print(value)

if __name__ == '__main__':
    pass