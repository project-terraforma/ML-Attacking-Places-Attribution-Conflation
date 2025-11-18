# ML-Attacking-Places-Attribution-Conflation

# All about cleanData.py 

## Overview
A data cleaning and matching pipeline that integrates business records from two different data sources (OMF and Yelp) to create a unified dataset.

## Data Sources

### Input Files
- **OMF Data**: `project_b_samples_2k.parquet`
  - Contains business information in JSON format within columns
  - Fields: names, categories, confidence, websites, socials, phones, addresses

- **Yelp Data**: `yelp_academic_dataset_business.json`
  - Line-delimited JSON file from Yelp Academic Dataset
  - Fields: name, address, city, state, postal_code, categories

## Processing Pipeline

### 1. Data Loading and Initial Cleaning
- Loads both datasets using pandas
- Filters to only necessary columns
- Drops rows with missing names or addresses

### 2. OMF Data Processing

#### Helper Functions
- `get_name()`: Extracts primary name from JSON
- `get_category()`: Extracts primary category from JSON
- `get_website()`: Gets first website from list
- `get_socials()`: Gets first social media link
- `get_phones()`: Gets first phone number
- `get_addresses()`: Combines address components into single string

#### Transformations
- Parses JSON fields into regular columns
- Creates concatenated full addresses
- Applies text cleaning (lowercase, alphanumeric only)

### 3. Yelp Data Processing
- Concatenates address components (address, city, state, postal_code)
- Applies same text cleaning as OMF data
- Maintains original and cleaned versions of fields

### 4. Record Matching

#### Exact Matching
- Direct inner join on cleaned name AND cleaned address
- Produces perfect matches only
- Output: `matched_candidates.csv`

#### Fuzzy Matching
- Uses RapidFuzz library for similarity scoring
- Token set ratio comparison for both names and addresses
- Threshold: ≥85% similarity for both fields
- More computationally intensive but catches variations
- Includes progress tracking (every 50 records)
- Output: `fuzzy_matched_candidates.csv`

## Output Files

### Cleaned Data Files
1. **cleaned_omf_data.csv**
   - Original OMF fields
   - Extracted JSON values (omf_name, omf_category, etc.)
   - Cleaned versions for matching

2. **cleaned_yelp_data.csv**
   - Original Yelp fields
   - Combined full address
   - Cleaned versions for matching

### Matched Records
3. **matched_candidates.csv**
   - Exact matches between datasets
   - Includes: names, addresses, categories, confidence scores, contact info

4. **fuzzy_matched_candidates.csv**
   - Similar matches based on fuzzy string matching
   - Same fields as exact matches
   - Captures variations in spelling, formatting, etc.

## Key Features

### Data Quality
- Robust error handling with try/except blocks
- Handles missing/malformed JSON gracefully
- Preserves original data alongside cleaned versions

### Text Normalization
- `clean_text()` function:
  - Converts to lowercase
  - Removes special characters
  - Strips whitespace
  - Keeps only alphanumeric and spaces

### Metadata Preservation
- Maintains confidence scores from OMF
- Preserves websites, social media, phone numbers
- Keeps category information from both sources

## Use Cases
- **Data Integration**: Combining business listings from multiple sources
- **Deduplication**: Identifying same businesses across datasets
- **Data Enrichment**: Adding missing information from one source to another
- **Quality Assessment**: Comparing data quality between sources

## Performance Considerations
- Exact matching is fast (pandas merge operation)
- Fuzzy matching is O(n×m) complexity - can be slow for large datasets
- Progress tracking helps monitor long-running fuzzy matches
- Consider chunking or parallel processing for production use

## Potential Improvements
1. Add configurable similarity thresholds
2. Implement blocking/indexing for faster fuzzy matching
3. Add more sophisticated address parsing
4. Include additional matching criteria (phone numbers, websites)
5. Add validation and quality metrics for matches
6. Implement incremental/update processing


## Rule-Based Algorithm Performance

### v1 (Baseline)
- Overall Accuracy: 94.24%
- Name Accuracy: 71.43%
- Errors: 8 systematic failures

### v2 (Refined)
- Overall Accuracy: **97.12%** ✅
- Name Accuracy: 85.71%
- Errors: 4 (50% reduction)
- Improvement: +2.88%

### By Attribute (v2)
- Name: 85.71% (24/28)
- Address: 100% (28/28)
- Phone: 100% (27/27)
- Category: 100% (28/28)
- Website: 100% (28/28)

## Key Improvements in v2
1. Canonical brand name detection
2. Business suffix removal (LLC, Inc, etc.)
3. Word count preference limits
4. Smarter tie-breaking logic

## Dataset
- Total matches: 28 (5 exact + 23 fuzzy)
- Match rate: 1.4% (limited by geographic mismatch)
- Data sources: OMF (2K) + Yelp Academic (150K)

## Status
 Objective 3 complete (exceeds 80% target)
 Ready for Objective 4 (ML models)
EOF
