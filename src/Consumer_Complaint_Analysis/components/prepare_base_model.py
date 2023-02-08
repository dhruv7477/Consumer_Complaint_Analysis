import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader
from Consumer_Complaint_Analysis.entity import PrepareBaseModelConfig
from Consumer_Complaint_Analysis import logger
import torch.nn as nn

class DataPreProcessing:
    def __init__(self, config):
        self.config = config

    def pre_process_data(self, path):
        """get pre processed data
    
        Args:
            path (str or Path): path of the file
    
        Returns:
            df: Pre Processed Data Frame
        """
    
        df = pd.read_csv(path)
    
        # Pre-processing
        # Replacing the NaN values with the most frequent value in each column
        for column in df.columns:
            df[column].fillna(df[column].mode()[0], inplace=True)

        # Convert the target column to 0 or 1
        df['Consumer disputed?'] = df['Consumer disputed?'].map({'No': 0, 'Yes': 1})
        df['Consumer disputed?'] = df['Consumer disputed?'].astype(int)
        
        return df


class PrepareBaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def analyze_firm_size_market_share(self, df):
        firm_size = df.groupby(['Company'])['Consumer complaint narrative'].count().reset_index(name='Complaint Count')
        firm_size['Market Share'] = firm_size['Complaint Count'] / firm_size['Complaint Count'].sum()
        return firm_size
    
    def analyze_population_of_state(self, df):
        state_population = df.groupby(['State'])['Consumer complaint narrative'].count().reset_index()
        state_population = state_population.rename(columns={'Consumer complaint narrative': 'Complaint Count'})
        state_population['Complaint per Capita'] = state_population['Complaint Count'] / df['State'].value_counts()
        self.state_population = state_population
        return state_population
        
    def analyze_population_of_ZIP_code(self, df):
        ZIP_code_population = df.groupby(['ZIP code'])['Consumer complaint narrative'].count().reset_index()
        ZIP_code_population = ZIP_code_population.rename(columns={'Consumer complaint narrative': 'Complaint Count'})
        ZIP_code_population['Complaint per Capita'] = ZIP_code_population['Complaint Count'] / df['ZIP code'].value_counts()
        self.ZIP_code_population = ZIP_code_population
        return ZIP_code_population

    def truncate_sequence(self,sequence, max_length=512):
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        return sequence
        
    def get_base_model(self, df):
        df = df.head(1000)
        #df = df.sample(frac=0.5)

        # Tokenization
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        sequences = df['Consumer complaint narrative'].tolist()
        truncated_sequences = [self.truncate_sequence(seq, max_length=512) for seq in sequences]
        encoded_data = tokenizer.batch_encode_plus(truncated_sequences, pad_to_max_length=True, return_attention_mask=True)
        input_ids = torch.tensor(encoded_data['input_ids'])
        attention_mask = torch.tensor(encoded_data['attention_mask'])
        labels = torch.tensor(df['Consumer disputed?'].tolist())
        train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids, labels, test_size=self.config.params_test_size, random_state=self.config.params_random_state)
        train_masks, test_masks, _, _ = train_test_split(attention_mask, input_ids, test_size=self.config.params_test_size, random_state=self.config.params_random_state)

        # Analyze firm size and market share
        self.analyze_firm_size_market_share(df)
        
        # Analyze population of a state
        self.analyze_population_of_state(df)
        
        # Analyze population of a ZIP code
        self.analyze_population_of_ZIP_code(df)

        # Creating a TensorDataset
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)

        # Data Loaders
        train_dataloader = DataLoader(train_data, batch_size=self.config.params_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=self.config.params_batch_size, shuffle=False)

        self.dataloaders = {"train": train_dataloader, "val": test_dataloader}
        
        # Model Configuration
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=self.config.params_num_labels)
        self.model = model
        self.save_model(self.config.base_model_path, self.model)
        
    @staticmethod
    def save_model(path, model):
        torch.save(model.state_dict(),path)