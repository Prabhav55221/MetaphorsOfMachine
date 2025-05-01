import os
import argparse
import time
from datetime import datetime

from config import Config
# from src.data import WildChatDataProcessor
# from src.frames import FrameExtractor
# from src.clustering import ClusterAnalyzer
from src.analysis import DemographicAnalyzer

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Metaphors of the Machine: AI Framing Analysis")
    
    parser.add_argument("--full", action="store_true", 
                        help="Use full dataset instead of sample")
    parser.add_argument("--sample-size", type=int, default=Config.SAMPLE_SIZE,
                        help="Number of conversations to sample")
    parser.add_argument("--force-reload", action="store_true", 
                        help="Force reload of dataset even if processed data exists")
    parser.add_argument("--skip-frames", action="store_true", 
                        help="Skip frame extraction (use existing data)")
    parser.add_argument("--skip-clustering", action="store_true", 
                        help="Skip clustering (use existing data)")
    parser.add_argument("--skip-analysis", action="store_true", 
                        help="Skip demographic analysis")
    parser.add_argument("--report-only", action="store_true", 
                        help="Only generate the final report from existing data")
    
    return parser.parse_args()

def setup_config(args):
    """Set up configuration based on command-line arguments."""
    Config.USE_SAMPLE = not args.full
    if args.sample_size:
        Config.SAMPLE_SIZE = args.sample_size
    
    # Create necessary directories
    Config.create_directories()
    
    # Print configuration
    Config.print_config()

def process_data(force_reload=False):
    """
    Load and process the WildChat dataset.
    
    Args:
        force_reload (bool): Force reload of dataset
        
    Returns:
        DataFrame: Processed AI references
    """
    print("\n=== Processing WildChat Dataset ===")
    start_time = time.time()
    
    # Initialize data processor
    data_processor = WildChatDataProcessor(
        data_dir=Config.DATA_DIR,
        use_sample=Config.USE_SAMPLE,
        sample_size=Config.SAMPLE_SIZE,
        random_seed=Config.RANDOM_SEED
    )
    
    # Load dataset
    try:
        if force_reload:
            raise FileNotFoundError("Forcing reload")
            
        references_df = data_processor.load_processed_data(
            filename="wildchat_ai_references.parquet"
        )
        print("Loaded existing processed data")
    except FileNotFoundError:
        print("Processing dataset from scratch")
        data_processor.load_dataset()
        data_processor.preprocess_data()
        references_df = data_processor.extract_ai_references()
    
    elapsed_time = time.time() - start_time
    print(f"Data processing completed in {elapsed_time:.2f} seconds")
    
    return references_df

def extract_frames(references_df, skip_frames=True):
    """
    Extract frames from AI references.
    
    Args:
        references_df (DataFrame): AI references
        skip_frames (bool): Skip frame extraction
        
    Returns:
        DataFrame: Extracted frames
    """
    print("\n=== Extracting Semantic Frames ===")
    start_time = time.time()
    
    # Initialize frame extractor
    frame_extractor = FrameExtractor(
        data_dir=Config.DATA_DIR,
        model_dir=os.path.join(Config.MODELS_DIR, "framebert"),
        device=Config.DEVICE
    )
    
    # Extract frames
    try:
        if skip_frames:
            raise FileNotFoundError("Skipping frame extraction")
            
        frames_df = frame_extractor.load_processed_data()
        print("Loaded existing frame data")
    except FileNotFoundError:
        print("Extracting frames from references")
        frames_df = frame_extractor.process_ai_references(references_df)
    
    elapsed_time = time.time() - start_time
    print(f"Frame extraction completed in {elapsed_time:.2f} seconds")
    
    return frames_df

def cluster_frames(skip_clustering=False):
    """
    Cluster extracted frames.
    
    Args:
        skip_clustering (bool): Skip clustering
        
    Returns:
        DataFrame: Clustering results
    """
    print("\n=== Clustering Frame Patterns ===")
    start_time = time.time()
    
    # Initialize cluster analyzer
    cluster_analyzer = ClusterAnalyzer(
        data_dir=Config.PROCESSED_DATA_DIR,
        results_dir=Config.RESULTS_DIR
    )
    
    # Load frame data
    cluster_analyzer.load_data()
    
    # Run clustering
    try:
        if skip_clustering:
            raise FileNotFoundError("Skipping clustering")
            
        # Try to load existing cluster model
        cluster_model = cluster_analyzer.load_model()
        cluster_results = cluster_analyzer.cluster_results
        print("Loaded existing cluster model")
    except FileNotFoundError:
        print("Running clustering")
        cluster_results = cluster_analyzer.cluster_embeddings(
            n_clusters=Config.OPTIMAL_CLUSTERS
        )
        
        # Save cluster model
        cluster_analyzer.save_model()
    
    elapsed_time = time.time() - start_time
    print(f"Clustering completed in {elapsed_time:.2f} seconds")
    
    return cluster_results

def analyze_patterns(skip_analysis=False):
    """
    Analyze demographic patterns in frames.
    
    Args:
        skip_analysis (bool): Skip analysis
    """
    print("\n=== Analyzing Demographic Patterns ===")
    start_time = time.time()
    
    # Initialize demographic analyzer
    analyzer = DemographicAnalyzer(
        data_dir=Config.PROCESSED_DATA_DIR,
        results_dir=Config.RESULTS_DIR
    )
    
    # Load frame data
    analyzer.load_data()
    
    # Run analysis
    if not skip_analysis:
        analyzer.analyze_all(
            min_country_count=Config.MIN_COUNTRY_COUNT,
            top_frames=Config.TOP_FRAMES
        )
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")

def main():
    """Main execution function."""

    print('Hello Boss!')

    # Parse command-line arguments
    args = parse_args()
    
    # Set up configuration
    setup_config(args)

    print('Bye Boss!')
    
    # Record start time
    overall_start_time = time.time()
    print(f"Starting Metaphors of the Machine analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not args.report_only:
        # Process dataset
        # references_df = process_data(force_reload=args.force_reload)
        
        # Extract frames
        # frames_df = extract_frames(references_df, skip_frames=True)
        
        # Cluster frames
        # cluster_results = cluster_frames(skip_clustering=args.skip_clustering)
        
        # Analyze patterns
        print('Here Boss!')
        analyze_patterns(skip_analysis=args.skip_analysis)
    else:
        # Generate report only
        print("\n=== Generating Report from Existing Data ===")
        analyzer = DemographicAnalyzer(
            data_dir=Config.PROCESSED_DATA_DIR,
            results_dir=Config.RESULTS_DIR
        )
        analyzer.load_data()
        analyzer._generate_report()
    
    # Calculate total execution time
    overall_elapsed_time = time.time() - overall_start_time
    minutes, seconds = divmod(overall_elapsed_time, 60)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
    print(f"Results saved to {Config.RESULTS_DIR}")
    print(f"Report saved to {os.path.join(Config.RESULTS_DIR, 'analysis_report.md')}")

main()