import logging

import pandas as pd

# Set up basic configuration for logging
logging.basicConfig(filename='MUD_Deduplication_log.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# This is the processijg that was done in the chunk in the previous code, process each file individually before taking union...

def process_data(data, data_name):
    logging.info(f'Processing {data_name}...')
    data_restricted = data[['Prscrbr_NPI', 'Prscrbr_First_Name',
                            'Prscrbr_Last_Org_Name', 'Prscrbr_City', 'Prscrbr_Type']]
    data_restricted = data_restricted[pd.notna(data_restricted['Prscrbr_First_Name']) & pd.notna(
        data_restricted['Prscrbr_Last_Org_Name']) & pd.notna(data_restricted['Prscrbr_NPI'])].reset_index(drop=True)
    logging.info(f'Removing rows with NaN values from {data_name}')

    data_restricted.drop_duplicates(subset=["Prscrbr_NPI"], inplace=True)
    data_restricted['Prscrbr_NPI'] = pd.to_numeric(
        data_restricted['Prscrbr_NPI'])  # Convert to integer to save space!
    return data_restricted


try:
    logging.info('Reading MUD datasets...')
    MUD1 = pd.read_csv("Data/MUP_NPI_2013.csv", low_memory=False)
    MUD2 = pd.read_csv("Data/MUP_NPI_2021.csv", low_memory=False)

    processed_MUD1 = process_data(MUD1, "MUP_NPI_2013.csv")
    processed_MUD2 = process_data(MUD2, "MUP_NPI_2021.csv")

    logging.info('Combining datasets...')
    combined_MUD = pd.concat([processed_MUD1, processed_MUD2])

    logging.info('Removing duplicates...')
    unique_MUD = combined_MUD.drop_duplicates(
        subset=["Prscrbr_NPI"]).reset_index(drop=True)

    logging.info(f'Number of unique rows: {len(unique_MUD)}')

    logging.info('Saving the unique combined data...')
    unique_MUD.to_csv("Data/2013-2014_unique_accumulated.csv", index=False)

    logging.info('Deduplication and saving process completed successfully.')

except Exception as e:
    logging.error(
        f"An error occurred during the deduplication process: {str(e)}")
