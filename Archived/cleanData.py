import pandas as pd 
import json
import re
from rapidfuzz import fuzz, process

'''
Steps:
Data Quality Assessment
1. Clean OMF data and anrrow down attribute scope


1. Load and clean OMF data
2. Load and clean Yelp data
3. Match records, simple name and address matching first, fuzzy matching next
4. Change abbreviations to increase fuzzy matching score
5. Create Golden Dataset CSV with both attributes side-by-side, SQL
'''

def get_name(value):
    try:
        if isinstance(value, str):
            # turning JSON into string so that we can access the 'name' text
            value = json.loads(value)
        # print(type(value.get('primary')))
        return value.get('primary')
    except:
        return None

def get_category(value):
    try:
        if isinstance(value, str):
            # turns into string
            value = json.loads(value)
        return value.get('primary')
    except:
        return None

def get_website(value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list) and len(value) > 0:
            return value[0]
    except:
        return None

def get_socials(value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list) and len(value) > 0:
            return value[0]
    except:
        return None

def get_phones(value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list) and len(value) > 0:
            return value[0]
    except:
        return None

def get_addresses(value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, list) and len(value) > 0:
            for i in ['freeform', 'locality', 'region', 'postcode']:
                if i not in value[0]:
                    value[0][i] = ''
            return value[0]['freeform'] + ',' + value[0]['locality'] + ',' + value[0]['region'] + ',' + value[0]['postcode']
    except:
        return None

# clean unwanted characters from text
def clean_text(text):
    if not isinstance(text, str):
        return None
    return re.sub(r'[^a-z0-9 ]', '', text.lower().strip())

if __name__ == "__main__":
    # read the data
    read_data_omf = pd.read_parquet('project_b_samples_2k.parquet')
    read_data_yelp = pd.read_json('yelp_academic_dataset_business.json', lines=True)

    # keep only necessary/wanted columns
    read_data_omf = read_data_omf[['names', 'categories', 'confidence','websites', 'socials', 'phones', 'addresses']]
    read_data_yelp = read_data_yelp[['name', 'address', 'city', 'state', 'postal_code', 'categories']]

    # save cleaned data as csv files
    # for visulaization purpose
    #omf_csv = read_data_omf.to_csv('cleaned_omf_data.csv', index=False)
    #yelp_csv = read_data_yelp.to_csv('cleaned_yelp_data.csv', index=False)
    
    #### OMF ####
    # drop rows with missing names or addresses
    read_data_omf = read_data_omf.dropna(subset=['names', 'addresses'])

    # retrieving information in non-JSON, string format
    read_data_omf["omf_name"] = read_data_omf['names'].apply(get_name)
    read_data_omf["omf_category"] = read_data_omf['categories'].apply(get_category)
    read_data_omf["omf_confidence"] = read_data_omf['confidence']
    read_data_omf["omf_websites"] = read_data_omf['websites'].apply(get_website)
    read_data_omf["omf_socials"] = read_data_omf['socials'].apply(get_socials)
    read_data_omf["omf_phones"] = read_data_omf['phones'].apply(get_phones)
    read_data_omf["omf_addresses"] = read_data_omf['addresses'].apply(get_addresses)

    # adding columns for cleaned names and addresses
    read_data_omf["clean_omf_name"] = read_data_omf['omf_name'].apply(clean_text)
    read_data_omf["clean_omf_addresses"] = read_data_omf['omf_addresses'].apply(clean_text)

    read_data_omf.to_csv("cleaned_omf_data.csv", index=False)
    

    #### YELP ####
    # drop rows with missing names or addresses
    read_data_yelp = read_data_yelp.dropna(subset=['name', 'address'])

    read_data_yelp["yelp_full_address"] = (
        read_data_yelp['address'] + ',' +
        read_data_yelp['city'] + ',' + 
        read_data_yelp['state'] + ',' + 
        read_data_yelp['postal_code']
    )

    # adding columns for cleaned names and addresses
    read_data_yelp["clean_yelp_name"] = read_data_yelp['name'].apply(clean_text)
    read_data_yelp["clean_yelp_address"] = read_data_yelp['yelp_full_address'].apply(clean_text)

    read_data_yelp.to_csv("cleaned_yelp_data.csv", index=False)


    #### CANDIDATE MATCHING ####
    # finding candidate matches between OMF and Yelp datasets
    omf = pd.read_csv("cleaned_omf_data.csv")
    yelp = pd.read_csv("cleaned_yelp_data.csv")

    pairs = omf.merge(
        yelp,
        left_on=['clean_omf_name', 'clean_omf_addresses'],
        right_on=['clean_yelp_name', 'clean_yelp_address'],
        how='inner',
        suffixes=('_omf', '_yelp')
    )
    pairs = pairs[['clean_omf_name', 'clean_yelp_name', 
                    'clean_omf_addresses', 'clean_yelp_address',
                    'omf_category', 'categories_yelp', 
                    'omf_confidence',
                    'omf_websites', 'omf_socials', 'omf_phones']]

    pairs.to_csv("matched_candidates.csv", index=False)


    #### FUZZY MATCHING ####
    # find matches for omf and yelp that are not exaclty identical
    fuzzy_matches = []

    for i, omf_row in omf.iterrows():
        omf_name = omf_row['clean_omf_name']
        omf_address = omf_row['clean_omf_addresses']

        for j, yelp_row in yelp.iterrows():
            yelp_name = yelp_row['clean_yelp_name']
            yelp_address = yelp_row['clean_yelp_address']

            name_similarity = fuzz.token_set_ratio(omf_name, yelp_name)
            address_similarity = fuzz.token_set_ratio(omf_address, yelp_address)

            if name_similarity >= 80 and address_similarity >= 80:
                fuzzy_matches.append({
                    'clean_omf_name': omf_name,
                    'clean_yelp_name': yelp_name,
                    'clean_omf_addresses': omf_address,
                    'clean_yelp_address': yelp_address,
                    'omf_category': omf_row['omf_category'],
                    'categories_yelp': yelp_row['categories'],
                    'omf_confidence': omf_row['omf_confidence'],
                    'omf_websites': omf_row['omf_websites'],
                    'omf_socials': omf_row['omf_socials'],
                    'omf_phones': omf_row['omf_phones']
                })
        
        if i % 50 == 0:  # every 50 Yelp rows
            print(f"Processed {i} Yelp records so far...")

    fuzzy_matches_df = pd.DataFrame(fuzzy_matches).to_csv("fuzzy_matched_candidates.csv", index=False)