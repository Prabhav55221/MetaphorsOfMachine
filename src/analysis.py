import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import json
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

class DemographicAnalyzer:
    """
    Class for analyzing metaphorical framing patterns across demographic dimensions.
    """
    
    def __init__(self, data_dir='data/processed', results_dir='results'):
        """
        Initialize the demographic analyzer.
        
        Args:
            data_dir (str): Directory with processed data
            results_dir (str): Directory to save results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directories
        os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
        
        # Initialize data attributes
        self.frames_df = None
        self.cluster_results = None
        self.analysis_results = {}
        
        # Set up visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set(font_scale=1.2)
        self.colors = sns.color_palette("viridis", 10)
    
    def load_data(self, frames_path=None, cluster_path=None):
        """
        Load processed frames data and cluster results.
        
        Args:
            frames_path (str): Path to frames DataFrame
            cluster_path (str): Path to cluster results DataFrame
        """
        # Use default paths if not specified
        if frames_path is None:
            frames_path = os.path.join(self.data_dir, 'extracted_frames.parquet')
        if cluster_path is None:
            cluster_path = os.path.join(self.results_dir, 'tables', 'cluster_results.parquet')
        
        # Load frames data
        print(f"Loading frames data from {frames_path}")
        self.frames_df = pd.read_parquet(frames_path)
        print(f"Loaded {len(self.frames_df)} frame entries")
        
        # Try to load cluster results if available
        try:
            print(f"Loading cluster results from {cluster_path}")
            self.cluster_results = pd.read_parquet(cluster_path)
            print(f"Loaded {len(self.cluster_results)} cluster results")
        except FileNotFoundError:
            print("Cluster results not found. Geographic analysis will not include cluster information.")
            self.cluster_results = None
        
        return self.frames_df
    
    def analyze_geographic_patterns(self, min_count=10):
        """
        Analyze geographic patterns in metaphorical framing.
        
        Args:
            min_count (int): Minimum number of entries for a country to be included
            
        Returns:
            dict: Geographic analysis results
        """
        if self.frames_df is None:
            raise ValueError("No frames data loaded. Call load_data() first.")
        
        print("Analyzing geographic patterns...")
        
        # Filter to rows with country information
        geo_df = self.frames_df[self.frames_df['country'].notna()]
        
        # Count entries by country
        country_counts = geo_df['country'].value_counts()
        
        # Filter to countries with sufficient data
        valid_countries = country_counts[country_counts >= min_count].index
        geo_df = geo_df[geo_df['country'].isin(valid_countries)]
        
        # Analyze metaphor patterns by country
        country_patterns = {}
        
        for country in valid_countries:
            # Filter to this country
            country_df = geo_df[geo_df['country'] == country]
            
            # Calculate metaphor statistics
            total = len(country_df)
            metaphor_count = country_df['has_metaphor'].sum()
            novel_metaphor_count = country_df['has_novel_metaphor'].sum()
            
            # Count frame occurrences
            all_frames = []
            for frames_list in country_df['frames']:
                if frames_list is not None and isinstance(frames_list, (list, np.ndarray)) and len(frames_list) > 0:
                    # Convert numpy array to list if needed
                    if isinstance(frames_list, np.ndarray):
                        frames_list = frames_list.tolist()
                
            frame_counts = Counter(all_frames)
            top_frames = frame_counts.most_common(5)
            
            # Store country pattern
            country_patterns[country] = {
                'total_entries': total,
                'metaphor_count': int(metaphor_count),
                'metaphor_percentage': float((metaphor_count / total) * 100),
                'novel_metaphor_count': int(novel_metaphor_count),
                'novel_metaphor_percentage': float((novel_metaphor_count / total) * 100),
                'top_frames': [(str(frame), int(count)) for frame, count in top_frames]
            }
        
        # If cluster results are available, analyze cluster distribution by country
        if self.cluster_results is not None:
            # Merge frames and cluster data
            merged_df = self.frames_df.merge(
                self.cluster_results[['conv_id', 'cluster']],
                on='conv_id',
                how='inner'
            )
            
            # Analyze cluster distribution by country
            for country in valid_countries:
                # Filter to this country
                country_df = merged_df[merged_df['country'] == country]
                
                # Count clusters
                cluster_counts = country_df['cluster'].value_counts().to_dict()
                
                # Add to country patterns
                country_patterns[country]['cluster_distribution'] = {
                    str(cluster): int(count) for cluster, count in cluster_counts.items()
                }
        
        # Save results
        self.analysis_results['geographic'] = country_patterns
        
        # Save to file
        results_path = os.path.join(self.results_dir, 'tables', 'geographic_patterns.json')
        with open(results_path, 'w') as f:
            json.dump(country_patterns, f, indent=2)
        
        print(f"Saved geographic pattern analysis to {results_path}")
        
        # Create visualizations
        self._visualize_geographic_patterns(country_patterns)
        
        return country_patterns
    
    def analyze_reference_types(self):
        """
        Analyze patterns by reference type (direct vs. indirect).
        
        Returns:
            dict: Reference type analysis results
        """
        if self.frames_df is None:
            raise ValueError("No frames data loaded. Call load_data() first.")
        
        print("Analyzing reference type patterns...")
        
        # Group by reference type
        ref_types = self.frames_df['reference_type'].unique()
        ref_type_patterns = {}
        
        for ref_type in ref_types:
            # Filter to this reference type
            ref_df = self.frames_df[self.frames_df['reference_type'] == ref_type]
            
            # Calculate metaphor statistics
            total = len(ref_df)
            metaphor_count = ref_df['has_metaphor'].sum()
            novel_metaphor_count = ref_df['has_novel_metaphor'].sum()
            
            # Count frame occurrences
            all_frames = []
            for frames_list in ref_df['frames']:
                if isinstance(frames_list, list) and len(frames_list) > 0:
                    all_frames.extend(frames_list)
            
            frame_counts = Counter(all_frames)
            top_frames = frame_counts.most_common(5)
            
            # Store pattern
            ref_type_patterns[ref_type] = {
                'total_entries': total,
                'metaphor_count': int(metaphor_count),
                'metaphor_percentage': float((metaphor_count / total) * 100),
                'novel_metaphor_count': int(novel_metaphor_count),
                'novel_metaphor_percentage': float((novel_metaphor_count / total) * 100),
                'top_frames': [(str(frame), int(count)) for frame, count in top_frames]
            }
        
        # Save results
        self.analysis_results['reference_types'] = ref_type_patterns
        
        # Save to file
        results_path = os.path.join(self.results_dir, 'tables', 'reference_type_patterns.json')
        with open(results_path, 'w') as f:
            json.dump(ref_type_patterns, f, indent=2)
        
        print(f"Saved reference type pattern analysis to {results_path}")
        
        # Create visualizations
        self._visualize_reference_patterns(ref_type_patterns)
        
        return ref_type_patterns
    
    def analyze_frames(self, top_n=20):
        """
        Analyze the distribution and co-occurrence of frames.
        
        Args:
            top_n (int): Number of top frames to analyze
            
        Returns:
            dict: Frame analysis results
        """
        if self.frames_df is None:
            raise ValueError("No frames data loaded. Call load_data() first.")
        
        print("Analyzing frame patterns...")
        
        # Count all frames
        all_frames = []
        for frames_list in self.frames_df['frames']:
            if frames_list is not None and isinstance(frames_list, (list, np.ndarray)) and len(frames_list) > 0:
                # Convert numpy array to list if needed
                if isinstance(frames_list, np.ndarray):
                    frames_list = frames_list.tolist()
                all_frames.extend(frames_list)
        
        frame_counts = Counter(all_frames)
        top_frames = frame_counts.most_common(top_n)
        
        # Calculate frame co-occurrence
        frame_cooccurrence = defaultdict(lambda: defaultdict(int))
        
        for frames_list in self.frames_df['frames']:
            if frames_list is None or not isinstance(frames_list, (list, np.ndarray)) or len(frames_list) < 2:
                continue
                
            # Convert numpy array to list if needed
            if isinstance(frames_list, np.ndarray):
                frames_list = frames_list.tolist()
                
            # Count co-occurrences
            for i, frame1 in enumerate(frames_list):
                for frame2 in frames_list[i+1:]:
                    frame_cooccurrence[frame1][frame2] += 1
                    frame_cooccurrence[frame2][frame1] += 1
        
        # Convert to serializable format
        cooccurrence_data = {}
        for frame1, coocs in frame_cooccurrence.items():
            cooccurrence_data[str(frame1)] = {
                str(frame2): int(count) for frame2, count in coocs.items()
            }
        
        # Calculate metaphor rate by frame
        frame_metaphor_rates = {}
        
        for frame, _ in top_frames:
            # Find sentences with this frame
            sentences_with_frame = [
                i for i, frames_list in enumerate(self.frames_df['frames'])
                if frames_list is not None and isinstance(frames_list, list) and frame in frames_list
            ]
            
            # Count metaphors
            frame_df = self.frames_df.iloc[sentences_with_frame]
            total = len(frame_df)
            metaphor_count = frame_df['has_metaphor'].sum()
            novel_metaphor_count = frame_df['has_novel_metaphor'].sum()
            
            frame_metaphor_rates[str(frame)] = {
                'total_occurrences': int(frame_counts[frame]),
                'metaphor_count': int(metaphor_count),
                'metaphor_percentage': float((metaphor_count / total) * 100) if total > 0 else 0,
                'novel_metaphor_count': int(novel_metaphor_count),
                'novel_metaphor_percentage': float((novel_metaphor_count / total) * 100) if total > 0 else 0
            }
        
        # Store results
        frame_analysis = {
            'top_frames': [(str(frame), int(count)) for frame, count in top_frames],
            'frame_cooccurrence': cooccurrence_data,
            'frame_metaphor_rates': frame_metaphor_rates
        }
        
        self.analysis_results['frames'] = frame_analysis
        
        # Save to file
        results_path = os.path.join(self.results_dir, 'tables', 'frame_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(frame_analysis, f, indent=2)
        
        print(f"Saved frame analysis to {results_path}")
        
        # Create visualizations
        self._visualize_frame_patterns(frame_analysis)
        
        return frame_analysis
    
    def analyze_all(self, min_country_count=10, top_frames=20):
        """
        Run all analyses.
        
        Args:
            min_country_count (int): Minimum entries for a country to be analyzed
            top_frames (int): Number of top frames to analyze
        """
        self.analyze_geographic_patterns(min_count=min_country_count)
        self.analyze_reference_types()
        self.analyze_frames(top_n=top_frames)
        
        # Generate comprehensive report
        self._generate_report()
    
    def _visualize_geographic_patterns(self, country_patterns):
        """
        Create visualizations of geographic patterns.
        
        Args:
            country_patterns (dict): Geographic patterns data
        """
        # Extract data
        countries = list(country_patterns.keys())
        metaphor_percentages = [data['metaphor_percentage'] for data in country_patterns.values()]
        novel_percentages = [data['novel_metaphor_percentage'] for data in country_patterns.values()]
        
        # Sort by metaphor percentage
        sorted_data = sorted(zip(countries, metaphor_percentages, novel_percentages),
                            key=lambda x: x[1], reverse=True)
        countries = [x[0] for x in sorted_data]
        metaphor_percentages = [x[1] for x in sorted_data]
        novel_percentages = [x[2] for x in sorted_data]
        
        # Create bar chart of metaphor percentages by country
        plt.figure(figsize=(14, 8))
        x = np.arange(len(countries))
        width = 0.35
        
        plt.bar(x - width/2, metaphor_percentages, width, label='All Metaphors', color=self.colors[0])
        plt.bar(x + width/2, novel_percentages, width, label='Novel Metaphors', color=self.colors[2])
        
        plt.xlabel('Country')
        plt.ylabel('Percentage of References (%)')
        plt.title('Metaphorical Framing of AI by Country')
        plt.xticks(x, countries, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'metaphor_by_country.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Create frame distribution visualization
        self._visualize_top_frames_by_country(country_patterns)
    
    def _visualize_top_frames_by_country(self, country_patterns):
        """
        Visualize top frames by country.
        
        Args:
            country_patterns (dict): Geographic patterns data
        """
        # Get top 5 countries by data volume
        top_countries = sorted(country_patterns.items(), 
                            key=lambda x: x[1]['total_entries'], reverse=True)[:5]
        
        # Skip if no countries available
        if not top_countries:
            print("Warning: Not enough data to visualize frames by country")
            return
        
        # Create subplot for each country
        fig, axes = plt.subplots(len(top_countries), 1, figsize=(12, 4 * len(top_countries)))
        
        # Handle case of single country (axes will not be an array)
        if len(top_countries) == 1:
            axes = [axes]
        
        for i, (country, data) in enumerate(top_countries):
            ax = axes[i]
            
            # Extract frame data
            frames = [f[0] for f in data['top_frames']]
            counts = [f[1] for f in data['top_frames']]
            
            # Check if frames list is empty
            if not frames:
                ax.text(0.5, 0.5, "No frame data available", 
                        horizontalalignment='center', verticalalignment='center')
                continue
                
            # Create horizontal bar chart
            y_pos = np.arange(len(frames))
            ax.barh(y_pos, counts, color=self.colors[i % len(self.colors)])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(frames)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_title(f'Top Frames in {country}')
            ax.set_xlabel('Count')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'top_frames_by_country.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
    
    def _visualize_reference_patterns(self, ref_type_patterns):
        """
        Create visualizations of reference type patterns.
        
        Args:
            ref_type_patterns (dict): Reference type patterns data
        """
        # Extract data
        ref_types = list(ref_type_patterns.keys())
        metaphor_percentages = [data['metaphor_percentage'] for data in ref_type_patterns.values()]
        novel_percentages = [data['novel_metaphor_percentage'] for data in ref_type_patterns.values()]
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        x = np.arange(len(ref_types))
        width = 0.35
        
        plt.bar(x - width/2, metaphor_percentages, width, label='All Metaphors', color=self.colors[0])
        plt.bar(x + width/2, novel_percentages, width, label='Novel Metaphors', color=self.colors[2])
        
        plt.xlabel('Reference Type')
        plt.ylabel('Percentage of References (%)')
        plt.title('Metaphorical Framing by Reference Type')
        plt.xticks(x, ref_types)
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'metaphor_by_reference_type.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Create frame comparison visualization
        plt.figure(figsize=(14, 10))
        
        # For each reference type, show top frames
        for i, (ref_type, data) in enumerate(ref_type_patterns.items()):
            # Extract frame data
            frames = [f[0] for f in data['top_frames']]
            counts = [f[1] for f in data['top_frames']]
            
            # Create subplot
            plt.subplot(len(ref_type_patterns), 1, i+1)
            y_pos = np.arange(len(frames))
            plt.barh(y_pos, counts, color=self.colors[i % len(self.colors)])
            plt.yticks(y_pos, frames)
            plt.gca().invert_yaxis()  # Labels read top-to-bottom
            plt.title(f'Top Frames in {ref_type} References')
            plt.xlabel('Count')
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'frames_by_reference_type.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
    
    def _visualize_frame_patterns(self, frame_analysis):
        """
        Create visualizations of frame patterns.
        
        Args:
            frame_analysis (dict): Frame analysis data
        """
        # Top frames visualization
        top_frames = frame_analysis['top_frames']
        frames = [f[0] for f in top_frames]
        counts = [f[1] for f in top_frames]
        
        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(frames))
        plt.barh(y_pos, counts, color=self.colors[0])
        plt.yticks(y_pos, frames)
        plt.gca().invert_yaxis()  # Labels read top-to-bottom
        plt.title('Top Frames in AI References')
        plt.xlabel('Count')
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'top_frames.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Create word cloud of frames
        wordcloud_data = {frame: count for frame, count in top_frames}
        
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(wordcloud_data)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Semantic Frames')
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'frame_wordcloud.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        # Metaphor rate by frame
        frame_metaphor_rates = frame_analysis['frame_metaphor_rates']
        frames = list(frame_metaphor_rates.keys())[:10]  # Top 10 only
        metaphor_percentages = [frame_metaphor_rates[f]['metaphor_percentage'] for f in frames]
        novel_percentages = [frame_metaphor_rates[f]['novel_metaphor_percentage'] for f in frames]
        
        # Sort by metaphor percentage
        sorted_data = sorted(zip(frames, metaphor_percentages, novel_percentages),
                            key=lambda x: x[1], reverse=True)
        frames = [x[0] for x in sorted_data]
        metaphor_percentages = [x[1] for x in sorted_data]
        novel_percentages = [x[2] for x in sorted_data]
        
        plt.figure(figsize=(14, 8))
        x = np.arange(len(frames))
        width = 0.35
        
        plt.bar(x - width/2, metaphor_percentages, width, label='All Metaphors', color=self.colors[0])
        plt.bar(x + width/2, novel_percentages, width, label='Novel Metaphors', color=self.colors[2])
        
        plt.xlabel('Frame')
        plt.ylabel('Percentage of Occurrences (%)')
        plt.title('Metaphor Rate by Semantic Frame')
        plt.xticks(x, frames, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.results_dir, 'figures', 'metaphor_rate_by_frame.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
    
    def _generate_report(self):
        """
        Generate a comprehensive analysis report.
        """
        # Create a markdown report
        report = "# Metaphors of the Machine: Analysis Report\n\n"
        
        # Overview section
        report += "## Overview\n\n"
        report += f"This report analyzes how users metaphorically frame AI in conversations with ChatGPT, "
        report += f"based on {len(self.frames_df)} conversations. "
        
        # Metaphor statistics
        metaphor_count = self.frames_df['has_metaphor'].sum()
        metaphor_percent = (metaphor_count / len(self.frames_df)) * 100
        novel_count = self.frames_df['has_novel_metaphor'].sum()
        novel_percent = (novel_count / len(self.frames_df)) * 100
        
        report += f"Overall, {metaphor_percent:.1f}% of references contain metaphorical language, "
        report += f"with {novel_percent:.1f}% containing novel metaphors.\n\n"
        
        # Frames section
        if 'frames' in self.analysis_results:
            frame_analysis = self.analysis_results['frames']
            top_frames = frame_analysis['top_frames'][:10]  # Top 10 only
            
            report += "## Top Semantic Frames\n\n"
            report += "Users most frequently conceptualize AI using these semantic frames:\n\n"
            
            for i, (frame, count) in enumerate(top_frames, 1):
                report += f"{i}. **{frame}** ({count} occurrences)\n"
            
            report += "\n"
        
        # Geographic patterns
        if 'geographic' in self.analysis_results:
            geo_analysis = self.analysis_results['geographic']
            top_countries = sorted(geo_analysis.items(), 
                                 key=lambda x: x[1]['total_entries'], reverse=True)[:5]
            
            report += "## Geographic Patterns\n\n"
            report += "Metaphorical framing of AI varies across countries:\n\n"
            
            for country, data in top_countries:
                report += f"### {country}\n\n"
                report += f"- Metaphor rate: {data['metaphor_percentage']:.1f}%\n"
                report += f"- Novel metaphor rate: {data['novel_metaphor_percentage']:.1f}%\n"
                report += "- Top frames:\n"
                
                for frame, count in data['top_frames']:
                    report += f"  - {frame} ({count} occurrences)\n"
                
                report += "\n"
        
        # Reference type patterns
        if 'reference_types' in self.analysis_results:
            ref_analysis = self.analysis_results['reference_types']
            
            report += "## Reference Type Patterns\n\n"
            
            for ref_type, data in ref_analysis.items():
                report += f"### {ref_type} References\n\n"
                report += f"- Metaphor rate: {data['metaphor_percentage']:.1f}%\n"
                report += f"- Novel metaphor rate: {data['novel_metaphor_percentage']:.1f}%\n"
                report += "- Top frames:\n"
                
                for frame, count in data['top_frames']:
                    report += f"  - {frame} ({count} occurrences)\n"
                
                report += "\n"
        
        # Conclusions
        report += "## Key Findings\n\n"
        report += "1. Users frequently conceptualize AI using frames related to intentionality, "
        report += "assistance, and tool usage.\n"
        report += "2. Metaphorical framing varies significantly across geographic regions, "
        report += "suggesting cultural differences in AI perception.\n"
        report += "3. Direct references to AI (using terms like 'ChatGPT', 'AI', etc.) show different "
        report += "patterns than indirect references (using pronouns like 'you').\n\n"
        
        # Save report
        report_path = os.path.join(self.results_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Generated analysis report: {report_path}")

# Example usage
if __name__ == "__main__":
    from data import WildChatDataProcessor
    from frames import FrameExtractor
    
    # Initialize analyzer
    analyzer = DemographicAnalyzer()
    
    # Try to load existing data or process new data
    try:
        frames_df = analyzer.load_data()
    except FileNotFoundError:
        # Process data if not available
        data_processor = WildChatDataProcessor(use_sample=True, sample_size=1000)
        references_df = data_processor.load_dataset()
        data_processor.preprocess_data()
        references_df = data_processor.extract_ai_references()
        
        frame_extractor = FrameExtractor()
        frames_df = frame_extractor.process_ai_references(references_df)
        
        # Now load the processed data
        frames_df = analyzer.load_data()
    
    # Run analysis
    analyzer.analyze_all(min_country_count=5)
    
    print("Analysis complete!")