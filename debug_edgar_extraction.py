"""
Debug script for EDGAR extraction issues in Google Colab.
Run this in your Colab notebook to diagnose extraction problems.
"""

print("="*80)
print("EDGAR EXTRACTION DIAGNOSTICS")
print("="*80)

# STEP 1: Check directory structure
print("\n1. Checking directory structure...")
print("-"*80)

import os
from pathlib import Path

base_path = '/content/drive/MyDrive/EDGAR_Project/edgar-crawler'

# Check if base directory exists
if not os.path.exists(base_path):
    print(f"❌ Base directory does not exist: {base_path}")
    print("   Create it or adjust the path!")
else:
    print(f"✓ Base directory exists: {base_path}")

# List all subdirectories
datasets_path = f'{base_path}/datasets'
if os.path.exists(datasets_path):
    print(f"\n✓ Datasets directory exists: {datasets_path}")
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(datasets_path):
        level = root.replace(datasets_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        # Only show first 10 files per directory
        for i, file in enumerate(files[:10]):
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... ({len(files) - 10} more files)')
        if level > 3:  # Limit depth
            break
else:
    print(f"❌ Datasets directory does not exist: {datasets_path}")

# STEP 2: Count files
print("\n2. Counting files by type...")
print("-"*80)

file_counts = {
    'RAW_FILINGS (.htm)': 0,
    'RAW_FILINGS (.txt)': 0,
    'STRUCTURED_FILINGS (.json)': 0,
    'Other files': 0
}

if os.path.exists(datasets_path):
    for root, dirs, files in os.walk(datasets_path):
        for file in files:
            if 'RAW_FILINGS' in root:
                if file.endswith('.htm'):
                    file_counts['RAW_FILINGS (.htm)'] += 1
                elif file.endswith('.txt'):
                    file_counts['RAW_FILINGS (.txt)'] += 1
            elif 'STRUCTURED_FILINGS' in root:
                if file.endswith('.json'):
                    file_counts['STRUCTURED_FILINGS (.json)'] += 1
            else:
                file_counts['Other files'] += 1

    for file_type, count in file_counts.items():
        print(f"  {file_type}: {count:,}")

# STEP 3: Check for extraction output
print("\n3. Checking for extraction output...")
print("-"*80)

structured_path = f'{datasets_path}/STRUCTURED_FILINGS'
if os.path.exists(structured_path):
    # Check if there are any JSON files
    json_files = []
    for root, dirs, files in os.walk(structured_path):
        json_files.extend([os.path.join(root, f) for f in files if f.endswith('.json')])

    if json_files:
        print(f"✓ Found {len(json_files)} JSON files")
        print("\nFirst 5 files:")
        for f in json_files[:5]:
            print(f"  - {f}")

        # Check size of first file
        if json_files:
            first_file = json_files[0]
            size = os.path.getsize(first_file)
            print(f"\nFirst file size: {size:,} bytes")

            # Try to read it
            try:
                import json
                with open(first_file, 'r') as f:
                    data = json.load(f)
                print(f"✓ File is valid JSON")
                print(f"  Keys: {list(data.keys())}")
            except Exception as e:
                print(f"❌ Error reading JSON: {e}")
    else:
        print("❌ No JSON files found in STRUCTURED_FILINGS/")
        print(f"   Directory exists but is empty: {structured_path}")
else:
    print(f"❌ STRUCTURED_FILINGS directory does not exist: {structured_path}")

# STEP 4: Check for config files
print("\n4. Checking for config files...")
print("-"*80)

config_locations = [
    f'{base_path}/config.json',
    f'{base_path}/edgar-crawler/config.json',
    f'{base_path}/src/config.json'
]

for config_path in config_locations:
    if os.path.exists(config_path):
        print(f"✓ Found config: {config_path}")
        try:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"  Config keys: {list(config.keys())}")

            # Show relevant settings
            for key in ['items_to_extract', 'output_dir', 'skip_existing']:
                if key in config:
                    print(f"  {key}: {config[key]}")
        except Exception as e:
            print(f"  Error reading config: {e}")
    else:
        print(f"❌ Not found: {config_path}")

# STEP 5: Check for log files
print("\n5. Checking for log files...")
print("-"*80)

log_locations = [
    f'{base_path}/extraction.log',
    f'{base_path}/edgar-crawler.log',
    f'{datasets_path}/extraction.log'
]

for log_path in log_locations:
    if os.path.exists(log_path):
        print(f"✓ Found log: {log_path}")
        size = os.path.getsize(log_path)
        print(f"  Size: {size:,} bytes")

        # Show last 20 lines
        with open(log_path, 'r') as f:
            lines = f.readlines()
        print("\n  Last 20 lines:")
        for line in lines[-20:]:
            print(f"  {line.rstrip()}")
    else:
        print(f"❌ Not found: {log_path}")

# STEP 6: Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

total_raw = file_counts['RAW_FILINGS (.htm)'] + file_counts['RAW_FILINGS (.txt)']
total_structured = file_counts['STRUCTURED_FILINGS (.json)']

if total_raw == 0:
    print("\n⚠️  No RAW_FILINGS found!")
    print("   You need to download filings first before extracting.")
    print("   Run the download step in your notebook.")

elif total_structured == 0:
    print(f"\n⚠️  Found {total_raw:,} raw filings but 0 extracted JSON files!")
    print("   The extraction process is failing silently.")
    print("\n   Possible issues:")
    print("   1. Check extraction cell code for errors")
    print("   2. Verify output directory path in config")
    print("   3. Check if extraction process completed")
    print("   4. Look for error messages in logs")

else:
    print(f"\n✓ Found {total_raw:,} raw filings")
    print(f"✓ Found {total_structured:,} extracted JSON files")
    extraction_rate = (total_structured / total_raw) * 100
    print(f"  Extraction rate: {extraction_rate:.1f}%")

    if extraction_rate < 50:
        print("\n⚠️  Low extraction rate!")
        print("   Many filings failed to extract.")
        print("   Review error messages in the extraction log.")

print("\n" + "="*80)
print("DIAGNOSTICS COMPLETE")
print("="*80)
