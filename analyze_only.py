import os
import argparse
import time
from datetime import datetime
from src.analysis import DemographicAnalyzer

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
        data_dir=os.path.join("data", "processed"),
        results_dir="results"
    )
    
    # Load frame data
    analyzer.load_data()
    
    # Run analysis
    if not skip_analysis:
        analyzer.analyze_all(
            min_country_count=3,
            top_frames=20
        )
    
    elapsed_time = time.time() - start_time
    print(f"Analysis completed in {elapsed_time:.2f} seconds")()

analyze_patterns()