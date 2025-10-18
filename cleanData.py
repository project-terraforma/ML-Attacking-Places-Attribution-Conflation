import pandas as pd 
import json

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
    # retrieving information in non-JSON format
    read_data_omf["omf_name"] = read_data_omf['names'].apply(get_name)
    read_data_omf["omf_category"] = read_data_omf['categories'].apply(get_category)
    read_data_omf["omf_confidence"] = read_data_omf['confidence']
    read_data_omf["omf_websites"] = read_data_omf['websites'].apply(get_website)
    read_data_omf["omf_socials"] = read_data_omf['socials'].apply(get_socials)
    read_data_omf["omf_phones"] = read_data_omf['phones'].apply(get_phones)
    read_data_omf["omf_addresses"] = read_data_omf['addresses'].apply(get_addresses)
    
    csv = read_data_omf.to_csv("cleaned_omf_data.csv", index=False)
    

    #### YELP ####