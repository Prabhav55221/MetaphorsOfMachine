import os

class Config:
    """
    Configuration for the Metaphors of the Machine project.
    """
    
    # Data paths
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    
    # Results paths
    RESULTS_DIR = "results"
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
    TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
    MODELS_DIR = os.path.join(RESULTS_DIR, "models")
    
    # Dataset settings
    DATASET_NAME = "allenai/WildChat-1M"
    USE_SAMPLE = False  # Set to False for full dataset
    SAMPLE_SIZE = 1000  # Number of conversations to sample
    RANDOM_SEED = 42
    
    # Model settings
    METAPHOR_MODEL = "CreativeLang/metaphor_detection_roberta_seq"
    NOVEL_METAPHOR_MODEL = "CreativeLang/novel_metaphors"
    FRAME_MODEL = "liyucheng/frame_finder"
    DEVICE = 'cuda'  # None for auto-selection, or 'cuda', 'cpu', 'mps'
    
    # Preprocessing settings
    MIN_SENTENCE_LENGTH = 5
    FILTER_TOXIC = True
    FILTER_ENGLISH = True
    
    # Clustering settings
    MIN_CLUSTERS = 10
    MAX_CLUSTERS = 30
    OPTIMAL_CLUSTERS = None  # None for auto-detection
    
    # Analysis settings
    MIN_COUNTRY_COUNT = 3  # Minimum entries for a country to be included
    TOP_FRAMES = 20  # Number of top frames to analyze
    
    # Create directories
    @classmethod
    def create_directories(cls):
        """Create necessary directories for the project."""
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.FIGURES_DIR, exist_ok=True)
        os.makedirs(cls.TABLES_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        
        print(f"Created project directories in {os.getcwd()}")
    
    @classmethod
    def get_frame_df_path(cls):
        """Get the path to the processed frames DataFrame."""
        return os.path.join(cls.PROCESSED_DATA_DIR, "extracted_frames.parquet")
    
    @classmethod
    def get_references_path(cls):
        """Get the path to the processed AI references DataFrame."""
        return os.path.join(cls.PROCESSED_DATA_DIR, "wildchat_ai_references.parquet")
    
    @classmethod
    def get_embeddings_path(cls):
        """Get the path to the sentence embeddings."""
        return os.path.join(cls.PROCESSED_DATA_DIR, "sentence_embeddings.npy")
    
    @classmethod
    def get_cluster_results_path(cls):
        """Get the path to the cluster results DataFrame."""
        return os.path.join(cls.TABLES_DIR, "cluster_results.parquet")
    
    @classmethod
    def print_config(cls):
        """Print the current configuration."""
        print("\n=== Metaphors of the Machine Configuration ===")
        print(f"Dataset: {cls.DATASET_NAME}")
        print(f"Using sample: {cls.USE_SAMPLE} ({cls.SAMPLE_SIZE if cls.USE_SAMPLE else 'N/A'})")
        print(f"Device: {cls.DEVICE or 'Auto'}")
        print(f"Frame model: {cls.FRAME_MODEL}")
        print(f"Clustering: {cls.OPTIMAL_CLUSTERS or 'Auto'} clusters")
        print(f"Min country count: {cls.MIN_COUNTRY_COUNT}")
        print(f"Top frames: {cls.TOP_FRAMES}")
        print("=" * 45 + "\n")