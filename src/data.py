import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import os
from datetime import datetime
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
nltk.download('punkt', force=True)
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

class WildChatDataProcessor:
    """
    Class for loading, preprocessing, and extracting AI references from the WildChat dataset.
    """
    
    def __init__(self, data_dir='data', use_sample=True, sample_size=10000, random_seed=42):
        """
        Initialize the WildChat data processor.
        
        Args:
            data_dir (str): Directory to store processed data
            use_sample (bool): Whether to use a sample of the dataset
            sample_size (int): Size of the sample to use
            random_seed (int): Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.use_sample = use_sample
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        # Create directories if they don't exist
        os.makedirs(os.path.join(data_dir, 'raw'), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'processed'), exist_ok=True)
        
        # Define regex patterns for AI references
        self.ai_patterns = {
            'direct': re.compile(r'\b(chatgpt|gpt|ai|artificial intelligence|language model|llm)\b', re.IGNORECASE),
            'indirect': re.compile(r'\b(you|your|yourself)\b', re.IGNORECASE),
        }
    
    def load_dataset(self, force_reload=False):
        """
        Load the WildChat dataset from HuggingFace.
        
        Args:
            force_reload (bool): Whether to force reload the dataset
            
        Returns:
            Dataset: The WildChat dataset
        """
        # Check if we already have the processed DataFrame
        processed_path = os.path.join(self.data_dir, 'processed', 
                                     f"wildchat_{'sample' if self.use_sample else 'full'}.parquet")
        
        if os.path.exists(processed_path) and not force_reload:
            print(f"Loading processed dataset from {processed_path}")
            self.df = pd.read_parquet(processed_path)
            return self.df
        
        print("Loading WildChat dataset from HuggingFace...")
        self.dataset = load_dataset("allenai/WildChat-1M", split="train")
        
        if self.use_sample:
            print(f"Using a sample of {self.sample_size} conversations")
            self.dataset = self.dataset.shuffle(seed=self.random_seed).select(range(self.sample_size))
        
        print(f"Loaded dataset with {len(self.dataset)} conversations")
        
        # Convert to DataFrame for easier analysis
        self.df = self._convert_to_dataframe()
        
        # Save the processed DataFrame
        self._save_processed_data()
        
        return self.df
    
    def _convert_to_dataframe(self):
        """
        Convert the HuggingFace dataset to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: Processed DataFrame with relevant fields
        """
        print("Converting dataset to DataFrame...")
        
        data = []
        for i, conv in enumerate(tqdm(self.dataset, desc="Processing conversations")):
            conv_id = i  # Use index as a conversation ID
            
            # Extract metadata
            country = None
            state = None
            language = None
            timestamp = None
            toxic = False
            user_messages = []
            assistant_messages = []
            
            # Process the conversation to extract needed fields
            conversation = conv.get('conversation', [])
            
            # Extract timestamp
            if 'timestamp' in conv and conv['timestamp']:
                try:
                    if isinstance(conv['timestamp'], (int, float)):
                        timestamp = datetime.fromtimestamp(conv['timestamp']/1000)  # Convert ms to seconds
                    else:
                        timestamp = conv['timestamp']
                except Exception as e:
                    print(f"Error parsing timestamp: {e}")
            
            # Process messages
            for msg in conversation:
                if msg.get('role', '').lower() == 'user':
                    user_messages.append(msg.get('content', ''))
                    
                    # Get metadata from first user message
                    if not country and 'country' in msg:
                        country = msg.get('country')
                    if not state and 'state' in msg:
                        state = msg.get('state')
                    if not language and 'language' in msg:
                        language = msg.get('language')
                    if not toxic and msg.get('toxic', False):
                        toxic = True
                        
                elif msg.get('role', '').lower() == 'assistant':
                    assistant_messages.append(msg.get('content', ''))
            
            # Extract message count
            num_messages = len(conversation)
            
            # First messages (if available)
            first_user_msg = user_messages[0] if user_messages else ""
            first_assistant_msg = assistant_messages[0] if assistant_messages else ""
            
            # All messages combined
            all_user_msgs = "\n\n".join(user_messages)
            all_assistant_msgs = "\n\n".join(assistant_messages)
            
            # Add to data list
            data.append({
                'conv_id': conv_id,
                'country': country,
                'state': state,
                'language': language,
                'timestamp': timestamp,
                'num_messages': num_messages,
                'toxic': toxic,
                'first_user_message': first_user_msg,
                'first_assistant_message': first_assistant_msg,
                'all_user_messages': all_user_msgs,
                'all_assistant_messages': all_assistant_msgs,
                'conversation_hash': conv.get('conversation_hash', None),
                'model': conv.get('model', None)
            })
        
        return pd.DataFrame(data)
    
    def preprocess_data(self):
        """
        Preprocess the data for analysis.
        
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if not hasattr(self, 'df'):
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        
        print("Preprocessing data...")
        
        # Filter out toxic content
        non_toxic_df = self.df[~self.df['toxic']]
        print(f"Removed {len(self.df) - len(non_toxic_df)} toxic conversations")
        
        # Filter for English content if language is available
        if 'language' in self.df.columns and not self.df['language'].isna().all():
            english_df = non_toxic_df[non_toxic_df['language'].fillna('').str.lower() == 'english']
            print(f"Filtered to {len(english_df)} English conversations")
        else:
            # If language not available, keep all non-toxic
            english_df = non_toxic_df
            print("Language information not available. Keeping all non-toxic conversations.")
        
        # Filter for conversations with country information if needed
        with_country_df = english_df[english_df['country'].notna()]
        print(f"Filtered to {len(with_country_df)} conversations with country information")
        
        # Store the preprocessed DataFrame
        self.preprocessed_df = with_country_df
        
        # Save the preprocessed data
        self._save_processed_data(filename="wildchat_preprocessed.parquet")
        
        return self.preprocessed_df
    
    def extract_ai_references(self):
        """
        Extract sentences referring to AI from user messages.
        
        Returns:
            pd.DataFrame: DataFrame with AI references
        """
        if not hasattr(self, 'preprocessed_df'):
            print("Preprocessed data not found. Running preprocessing...")
            self.preprocess_data()
        
        print("Extracting AI references from user messages...")
        
        # Initialize lists to store references
        references = []
        
        # Process each user message
        for idx, row in tqdm(self.preprocessed_df.iterrows(), total=len(self.preprocessed_df)):
            user_msgs = row['all_user_messages']
            if not isinstance(user_msgs, str) or not user_msgs:
                continue
                
            sentences = sent_tokenize(user_msgs)
            
            for sentence in sentences:
                reference_type = None
                
                # Check for direct AI references
                if self.ai_patterns['direct'].search(sentence):
                    reference_type = 'direct'
                # Check for indirect AI references (you, your)
                elif self.ai_patterns['indirect'].search(sentence):
                    reference_type = 'indirect'
                
                if reference_type:
                    references.append({
                        'conv_id': row['conv_id'],
                        'country': row['country'],
                        'state': row['state'],
                        'language': row['language'],
                        'timestamp': row['timestamp'],
                        'sentence': sentence,
                        'reference_type': reference_type,
                    })
        
        # Create DataFrame with references
        references_df = pd.DataFrame(references)
        print(f"Extracted {len(references_df)} sentences with AI references")
        print(f"Direct references: {(references_df['reference_type'] == 'direct').sum()}")
        print(f"Indirect references: {(references_df['reference_type'] == 'indirect').sum()}")
        
        # Save the references DataFrame
        self._save_processed_data(references_df, "wildchat_ai_references.parquet")
        
        self.references_df = references_df
        return references_df
    
    def _save_processed_data(self, df=None, filename=None):
        """
        Save processed data to disk.
        
        Args:
            df (pd.DataFrame): DataFrame to save, defaults to self.df
            filename (str): Filename to use, defaults to wildchat_(sample/full).parquet
        """
        if df is None:
            if hasattr(self, 'df'):
                df = self.df
            else:
                raise ValueError("No DataFrame to save")
        
        if filename is None:
            filename = f"wildchat_{'sample' if self.use_sample else 'full'}.parquet"
        
        save_path = os.path.join(self.data_dir, 'processed', filename)
        print(f"Saving processed data to {save_path}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as parquet for efficiency
        df.to_parquet(save_path, index=False)
    
    def load_processed_data(self, filename=None):
        """
        Load processed data from disk.
        
        Args:
            filename (str): Filename to load, defaults to wildchat_(sample/full).parquet
            
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        if filename is None:
            filename = f"wildchat_{'sample' if self.use_sample else 'full'}.parquet"
        
        load_path = os.path.join(self.data_dir, 'processed', filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Processed data file {load_path} not found")
        
        print(f"Loading processed data from {load_path}")
        return pd.read_parquet(load_path)

# Simple usage example
if __name__ == "__main__":
    processor = WildChatDataProcessor(use_sample=True, sample_size=5000)
    df = processor.load_dataset()
    print(f"Loaded dataset with {len(df)} conversations")
    
    # Preprocess data
    preprocessed_df = processor.preprocess_data()
    print(f"Preprocessed dataset has {len(preprocessed_df)} conversations")
    
    # Extract AI references
    references_df = processor.extract_ai_references()
    print(f"Extracted {len(references_df)} AI references")