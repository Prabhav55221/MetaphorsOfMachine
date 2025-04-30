import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import Counter, defaultdict

class ClusterAnalyzer:
    """
    Class for clustering frame embeddings and analyzing metaphorical patterns.
    """
    
    def __init__(self, data_dir='data/processed', results_dir='results'):
        """
        Initialize the cluster analyzer.
        
        Args:
            data_dir (str): Directory with processed data
            results_dir (str): Directory to save results
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        
        # Create results directories
        os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'tables'), exist_ok=True)
        
        # Initialize attributes
        self.frames_df = None
        self.embeddings = None
        self.cluster_model = None
        self.pca_model = None
        self.cluster_results = None
    
    def load_data(self, frames_path=None, embeddings_path=None):
        """
        Load processed frames data and embeddings.
        
        Args:
            frames_path (str): Path to frames DataFrame
            embeddings_path (str): Path to embeddings array
        """
        # Use default paths if not specified
        if frames_path is None:
            frames_path = os.path.join(self.data_dir, 'extracted_frames.parquet')
        if embeddings_path is None:
            embeddings_path = os.path.join(self.data_dir, 'sentence_embeddings.npy')
        
        # Load data
        print(f"Loading frames data from {frames_path}")
        self.frames_df = pd.read_parquet(frames_path)
        
        print(f"Loading embeddings from {embeddings_path}")
        try:
            self.embeddings = np.load(embeddings_path)
            print(f"Loaded embeddings with shape {self.embeddings.shape}")
        except FileNotFoundError:
            print("Warning: No embeddings found. Will use frames data only.")
            self.embeddings = None
        
        return self.frames_df
    
    def extract_features(self):
        """
        Extract features for clustering from processed frames.
        
        Returns:
            DataFrame: Features for clustering
        """
        print("Extracting features for clustering...")
        
        # Create a DataFrame for clustering
        features = []
        
        # Process each entry in frames_df
        for _, row in self.frames_df.iterrows():
            # Skip entries without frames
            if row['frames'] is None or isinstance(row['frames'], (list, np.ndarray)) and len(row['frames']) == 0:
                continue
            
            # Count frames in the sentence
            frame_counts = Counter(row['frames'])
            
            # Create a feature dictionary
            feature = {
                'conv_id': row['conv_id'],
                'sentence': row['sentence'],
                'reference_type': row['reference_type'],
                'country': row['country'],
                'state': row['state'],
                'has_metaphor': row['has_metaphor'],
                'has_novel_metaphor': row['has_novel_metaphor'],
                'embedding_idx': row['embedding_idx'],
                'frame_counts': dict(frame_counts),
                'top_frame': max(frame_counts.items(), key=lambda x: x[1])[0] if frame_counts else None,
                'num_frames': len(row['frames']),
                'unique_frames': len(frame_counts)
            }
            
            features.append(feature)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features)
        print(f"Extracted features for {len(features_df)} sentences with frames")
        
        return features_df
    
    def find_optimal_clusters(self, min_clusters=2, max_clusters=20, random_state=42):
        """
        Find the optimal number of clusters using silhouette scores.
        
        Args:
            min_clusters (int): Minimum number of clusters to try
            max_clusters (int): Maximum number of clusters to try
            random_state (int): Random seed for reproducibility
            
        Returns:
            int: Optimal number of clusters
        """
        if self.embeddings is None:
            raise ValueError("No embeddings available for clustering")
        
        print("Finding optimal number of clusters...")
        
        # Extract features subset with frames
        features_df = self.extract_features()
        
        # Get embeddings for sentences with frames
        embedding_indices = features_df['embedding_idx'].values
        embeddings_subset = self.embeddings[embedding_indices]
        
        # Calculate silhouette scores for different cluster counts
        silhouette_scores = []
        
        for n_clusters in tqdm(range(min_clusters, max_clusters + 1), desc="Testing cluster counts"):
            # Skip if too few samples
            if len(embeddings_subset) <= n_clusters:
                silhouette_scores.append(0)
                continue
                
            # Run k-means
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings_subset)
            
            # Calculate silhouette score
            score = silhouette_score(embeddings_subset, cluster_labels)
            silhouette_scores.append(score)
            
            print(f"  {n_clusters} clusters: silhouette score = {score:.4f}")
        
        # Find the best number of clusters
        best_n_clusters = min_clusters + np.argmax(silhouette_scores)
        best_score = silhouette_scores[np.argmax(silhouette_scores)]
        
        print(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.4f})")
        
        # Plot silhouette scores
        plt.figure(figsize=(10, 6))
        plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score for Different Cluster Counts')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'figures', 'silhouette_scores.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        return best_n_clusters
    
    def cluster_embeddings(self, n_clusters=None, random_state=42):
        """
        Cluster embeddings using K-means.
        
        Args:
            n_clusters (int): Number of clusters (if None, find optimal)
            random_state (int): Random seed for reproducibility
            
        Returns:
            DataFrame: Clustering results
        """
        # Extract features subset with frames
        features_df = self.extract_features()
        
        # Check if embeddings are available
        if self.embeddings is None:
            print("No embeddings available. Generating text features for clustering...")
            
            # Generate bag-of-words features as fallback
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Use sentence text as features
            sentences = features_df['sentence'].tolist()
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Use TF-IDF features instead of embeddings
            embeddings_subset = tfidf_matrix.toarray()
            print(f"Generated TF-IDF features with shape {embeddings_subset.shape}")
        else:
            # Use pre-computed embeddings
            embedding_indices = features_df['embedding_idx'].values
            embeddings_subset = self.embeddings[embedding_indices]
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            print("Finding optimal number of clusters...")
            
            # Calculate silhouette scores for different cluster counts
            silhouette_scores = []
            
            min_clusters = min(2, len(features_df) - 1)
            max_clusters = min(20, len(features_df) - 1)
            
            if max_clusters <= min_clusters:
                print(f"Too few samples ({len(features_df)}) for meaningful clustering")
                n_clusters = min(2, len(features_df))
            else:
                for n in tqdm(range(min_clusters, max_clusters + 1), desc="Testing cluster counts"):
                    # Skip if too few samples
                    if len(embeddings_subset) <= n:
                        silhouette_scores.append(0)
                        continue
                        
                    # Run k-means
                    kmeans = KMeans(n_clusters=n, random_state=random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings_subset)
                    
                    # Calculate silhouette score
                    score = silhouette_score(embeddings_subset, cluster_labels)
                    silhouette_scores.append(score)
                    
                    print(f"  {n} clusters: silhouette score = {score:.4f}")
                
                # Find the best number of clusters
                best_n_clusters = min_clusters + np.argmax(silhouette_scores)
                best_score = silhouette_scores[np.argmax(silhouette_scores)]
                
                print(f"Optimal number of clusters: {best_n_clusters} (score: {best_score:.4f})")
                n_clusters = best_n_clusters
        
        print(f"Clustering data into {n_clusters} clusters...")
        
        # Fit k-means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_subset)
        
        # Save the model
        self.cluster_model = kmeans
        
        # Add cluster labels to features
        features_df['cluster'] = cluster_labels
        
        # Reduce dimensionality for visualization
        print("Reducing dimensionality for visualization...")
        pca = PCA(n_components=2, random_state=random_state)
        embeddings_2d = pca.fit_transform(embeddings_subset)
        
        # Save PCA model
        self.pca_model = pca
        
        # Add 2D coordinates
        features_df['x'] = embeddings_2d[:, 0]
        features_df['y'] = embeddings_2d[:, 1]
        
        # Save results
        self.cluster_results = features_df
        
        # Analyze clusters
        self._analyze_clusters()
        
        # Visualize clusters
        self._visualize_clusters()
        
        return features_df
    
    def _analyze_clusters(self):
        """
        Analyze clustering results.
        """
        if self.cluster_results is None:
            raise ValueError("No clustering results available")
        
        print("Analyzing clusters...")
        
        # Create a summary of each cluster
        cluster_summaries = []
        
        for cluster_id in sorted(self.cluster_results['cluster'].unique()):
            # Get sentences in this cluster
            cluster_df = self.cluster_results[self.cluster_results['cluster'] == cluster_id]
            
            # Count frames in this cluster
            all_frames = []
            for frames in cluster_df['frame_counts']:
                all_frames.extend(frames.keys())
            
            frame_counts = Counter(all_frames)
            # Convert to Python native types for JSON serialization
            top_frames = [(str(frame), int(count)) for frame, count in frame_counts.most_common(5)]
            
            # Count metaphor types - convert to native int
            metaphor_count = int(cluster_df['has_metaphor'].sum())
            novel_metaphor_count = int(cluster_df['has_novel_metaphor'].sum())
            
            # Count countries
            country_counts = {str(k): int(v) for k, v in cluster_df['country'].value_counts().head(3).to_dict().items()}
            
            # Create summary with native Python types
            summary = {
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_df)),
                'top_frames': top_frames,
                'metaphor_count': metaphor_count,
                'metaphor_percentage': float((metaphor_count / len(cluster_df)) * 100),
                'novel_metaphor_count': novel_metaphor_count,
                'novel_metaphor_percentage': float((novel_metaphor_count / len(cluster_df)) * 100),
                'top_countries': country_counts,
                'sample_sentences': [str(s) for s in cluster_df['sentence'].sample(min(3, len(cluster_df))).tolist()]
            }
            
            cluster_summaries.append(summary)
        
        # Save summaries
        summaries_path = os.path.join(self.results_dir, 'tables', 'cluster_summaries.json')
        with open(summaries_path, 'w') as f:
            json.dump(cluster_summaries, f, indent=2)
        
        print(f"Saved cluster summaries to {summaries_path}")
        
        # Print top frames per cluster
        print("\nTop frames per cluster:")
        for summary in cluster_summaries:
            print(f"Cluster {summary['cluster_id']} (size: {summary['size']}):")
            for frame, count in summary['top_frames']:
                print(f"  - {frame}: {count}")
            print()
    
    def _visualize_clusters(self):
        """
        Visualize clustering results.
        """
        if self.cluster_results is None:
            raise ValueError("No clustering results available")
        
        print("Visualizing clusters...")
        
        # Create scatter plot of clusters
        plt.figure(figsize=(12, 10))
        
        # Plot each cluster with a different color
        for cluster_id in sorted(self.cluster_results['cluster'].unique()):
            cluster_df = self.cluster_results[self.cluster_results['cluster'] == cluster_id]
            plt.scatter(cluster_df['x'], cluster_df['y'], label=f'Cluster {cluster_id}', alpha=0.7)
        
        # Add cluster centers
        centers = self.cluster_model.cluster_centers_
        centers_2d = self.pca_model.transform(centers)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=100, c='black', marker='X', label='Cluster Centers')
        
        plt.title('AI Reference Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'figures', 'cluster_visualization.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        print(f"Saved cluster visualization to {fig_path}")
        
        # Create additional visualizations by country
        self._visualize_by_country()
    
    def _visualize_by_country(self):
        """
        Create visualizations showing cluster distribution by country.
        """
        if self.cluster_results is None:
            raise ValueError("No clustering results available")
        
        # Get top countries
        country_counts = self.cluster_results['country'].value_counts().head(10)
        top_countries = country_counts.index.tolist()
        
        # Create cluster distribution by country
        country_cluster_counts = defaultdict(lambda: defaultdict(int))
        
        for _, row in self.cluster_results.iterrows():
            country = row['country']
            cluster = row['cluster']
            
            if country in top_countries:
                country_cluster_counts[country][cluster] += 1
        
        # Convert to DataFrame
        country_data = []
        for country in top_countries:
            for cluster in sorted(self.cluster_results['cluster'].unique()):
                count = country_cluster_counts[country][cluster]
                country_data.append({
                    'country': country,
                    'cluster': f'Cluster {cluster}',
                    'count': count
                })
        
        country_df = pd.DataFrame(country_data)
        
        # Create stacked bar chart
        plt.figure(figsize=(14, 8))
        
        # Pivot data for plotting
        pivot_df = country_df.pivot(index='country', columns='cluster', values='count').fillna(0)
        
        # Sort countries by total count
        pivot_df = pivot_df.loc[top_countries]
        
        # Create stacked bar chart
        pivot_df.plot(kind='bar', stacked=True, ax=plt.gca())
        
        plt.title('Cluster Distribution by Country')
        plt.xlabel('Country')
        plt.ylabel('Number of Sentences')
        plt.xticks(rotation=45)
        plt.legend(title='Cluster')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'figures', 'cluster_by_country.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        print(f"Saved country distribution visualization to {fig_path}")
    
    def save_model(self, filename='cluster_model.pkl'):
        """
        Save clustering model and results.
        
        Args:
            filename (str): Filename for model
        """
        if self.cluster_model is None:
            raise ValueError("No cluster model to save")
        
        # Save model using joblib
        from joblib import dump
        
        model_path = os.path.join(self.results_dir, 'models', filename)
        dump(self.cluster_model, model_path)
        
        # Save cluster results
        results_path = os.path.join(self.results_dir, 'tables', 'cluster_results.parquet')
        self.cluster_results.to_parquet(results_path, index=False)
        
        print(f"Saved cluster model to {model_path}")
        print(f"Saved cluster results to {results_path}")
    
    def load_model(self, filename='cluster_model.pkl'):
        """
        Load clustering model and results.
        
        Args:
            filename (str): Filename for model
        """
        from joblib import load
        
        model_path = os.path.join(self.results_dir, 'models', filename)
        results_path = os.path.join(self.results_dir, 'tables', 'cluster_results.parquet')
        
        if not os.path.exists(model_path) or not os.path.exists(results_path):
            raise FileNotFoundError(f"Model or results file not found: {model_path}, {results_path}")
        
        # Load model
        self.cluster_model = load(model_path)
        
        # Load results
        self.cluster_results = pd.read_parquet(results_path)
        
        print(f"Loaded cluster model from {model_path}")
        print(f"Loaded cluster results from {results_path}")
        
        return self.cluster_model

# Example usage
if __name__ == "__main__":
    from data import WildChatDataProcessor
    from frames import FrameExtractor
    
    # Initialize analyzer
    cluster_analyzer = ClusterAnalyzer()
    
    # Try to load existing data or process new data
    try:
        frames_df = cluster_analyzer.load_data()
    except FileNotFoundError:
        # Process data if not available
        data_processor = WildChatDataProcessor(use_sample=True, sample_size=1000)
        references_df = data_processor.load_dataset()
        data_processor.preprocess_data()
        references_df = data_processor.extract_ai_references()
        
        frame_extractor = FrameExtractor()
        frames_df = frame_extractor.process_ai_references(references_df)
        
        # Now load the processed data
        frames_df = cluster_analyzer.load_data()
    
    # Run clustering
    cluster_results = cluster_analyzer.cluster_embeddings()
    
    # Save model
    cluster_analyzer.save_model()
    
    print("Clustering complete!")