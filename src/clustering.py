import os
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from collections import Counter, defaultdict
from joblib import Parallel, delayed, dump, load
import time
import warnings
from sklearn.preprocessing import StandardScaler

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
        
        # Performance settings
        self.sample_size = 50000  # Maximum sample size for finding optimal clusters
        self.reduced_dims = 100   # Dimensionality reduction target
        self.batch_size = 1000    # Batch size for MiniBatchKMeans
        self.n_jobs = -1          # Number of parallel jobs (-1 = all cores)
        self.max_iter = 100       # Maximum iterations for clustering
        
        # Progress tracking
        self.start_time = None
    
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
            # Use memory mapping for large embeddings to reduce memory usage
            self.embeddings = np.load(embeddings_path, mmap_mode='r')
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
        
        # Create a DataFrame for clustering with improved efficiency
        # Process in chunks to reduce memory usage
        chunk_size = 10000
        all_features = []
        
        # Get total chunks for progress reporting
        total_chunks = (len(self.frames_df) + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(self.frames_df))
            chunk = self.frames_df.iloc[start_idx:end_idx]
            
            chunk_features = []
            for _, row in chunk.iterrows():
                # Skip entries without frames
                if row['frames'] is None or isinstance(row['frames'], (list, np.ndarray)) and len(row['frames']) == 0:
                    continue
                
                # Count frames in the sentence
                frame_counts = Counter(row['frames'])
                
                # Create a feature dictionary (only essential data)
                feature = {
                    'conv_id': row['conv_id'],
                    'sentence': row['sentence'],
                    'reference_type': row['reference_type'],
                    'country': row['country'] if pd.notnull(row['country']) else None,
                    'state': row['state'] if pd.notnull(row['state']) else None,
                    'has_metaphor': row['has_metaphor'],
                    'has_novel_metaphor': row['has_novel_metaphor'],
                    'embedding_idx': row['embedding_idx'],
                    'frame_counts': dict(frame_counts),
                    'top_frame': max(frame_counts.items(), key=lambda x: x[1])[0] if frame_counts else None,
                    'num_frames': len(row['frames']),
                    'unique_frames': len(frame_counts)
                }
                
                chunk_features.append(feature)
            
            all_features.extend(chunk_features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        print(f"Extracted features for {len(features_df)} sentences with frames")
        
        return features_df
    
    def _reduce_dimensions(self, data, target_dims=None, method='TruncatedSVD'):
        """
        Reduce dimensionality of embeddings for faster processing.
        
        Args:
            data (ndarray): Input embeddings
            target_dims (int): Target dimensions (default: self.reduced_dims)
            method (str): Reduction method ('PCA', 'TruncatedSVD')
            
        Returns:
            ndarray: Reduced embeddings
        """
        if target_dims is None:
            target_dims = min(self.reduced_dims, data.shape[1] - 1)
        
        print(f"Reducing dimensions from {data.shape[1]} to {target_dims} using {method}...")
        
        # Scale the data first for better results
        scaler = StandardScaler(with_mean=False)  # with_mean=False for sparse data compatibility
        scaled_data = scaler.fit_transform(data)
        
        if method == 'PCA':
            reducer = PCA(n_components=target_dims, random_state=42)
            reduced_data = reducer.fit_transform(scaled_data)
        elif method == 'TruncatedSVD':
            reducer = TruncatedSVD(n_components=target_dims, random_state=42)
            reduced_data = reducer.fit_transform(scaled_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"Reduced dimensions to {reduced_data.shape}")
        return reduced_data, reducer
    
    def _evaluate_clusters(self, data, labels, metric='calinski_harabasz'):
        """
        Evaluate clustering quality with different metrics.
        
        Args:
            data (ndarray): Input data
            labels (ndarray): Cluster labels
            metric (str): Evaluation metric
            
        Returns:
            float: Score
        """
        if metric == 'silhouette':
            # Sample data if too large for silhouette
            if len(data) > 10000:
                sample_idx = np.random.choice(len(data), 10000, replace=False)
                return silhouette_score(data[sample_idx], labels[sample_idx])
            return silhouette_score(data, labels)
        elif metric == 'calinski_harabasz':
            return calinski_harabasz_score(data, labels)
        elif metric == 'davies_bouldin':
            return davies_bouldin_score(data, labels)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def find_optimal_clusters(self, min_clusters=2, max_clusters=20, random_state=42):
        """
        Find the optimal number of clusters using multiple evaluation metrics.
        
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
        start_time = time.time()
        
        # Extract features subset with frames
        features_df = self.extract_features()
        
        # Sample data for faster analysis
        print("Sampling data for cluster optimization...")
        
        sample_size = min(self.sample_size, len(features_df))
        
        # Stratify by country if available
        if 'country' in features_df.columns and features_df['country'].notna().any():
            # Get count per country
            country_counts = features_df['country'].value_counts()
            # Identify countries with sufficient data for stratification
            valid_countries = country_counts.index[country_counts >= 5].tolist()
            
            # Calculate per-country sample sizes proportionally
            country_samples = {}
            other_sample = 0
            
            # Calculate total valid country count
            valid_count = features_df[features_df['country'].isin(valid_countries)].shape[0]
            other_count = len(features_df) - valid_count
            
            for country in valid_countries:
                country_pct = country_counts[country] / valid_count
                country_samples[country] = max(10, int(country_pct * sample_size * valid_count / len(features_df)))
            
            if other_count > 0:
                other_sample = sample_size - sum(country_samples.values())
            
            # Sample from each country
            sampled_indices = []
            for country, count in country_samples.items():
                country_indices = features_df[features_df['country'] == country].index.tolist()
                if len(country_indices) > count:
                    sampled_indices.extend(np.random.choice(country_indices, count, replace=False))
                else:
                    sampled_indices.extend(country_indices)
            
            # Sample from others
            if other_sample > 0:
                other_indices = features_df[~features_df['country'].isin(valid_countries)].index.tolist()
                if len(other_indices) > other_sample:
                    sampled_indices.extend(np.random.choice(other_indices, other_sample, replace=False))
                else:
                    sampled_indices.extend(other_indices)
            
            # Get final sample
            sampled_df = features_df.loc[sampled_indices]
        else:
            # Simple random sampling if no stratification possible
            sampled_df = features_df.sample(sample_size, random_state=random_state)
        
        print(f"Using {len(sampled_df)} samples for optimization")
        
        # Get embeddings for sampled sentences
        embedding_indices = sampled_df['embedding_idx'].values
        embeddings_subset = np.array([self.embeddings[i] for i in embedding_indices])
        
        # Reduce dimensions for faster clustering
        reduced_embeddings, _ = self._reduce_dimensions(embeddings_subset)
        
        # Calculate scores for different cluster counts using multiple metrics
        cluster_scores = {'calinski_harabasz': [], 'davies_bouldin': []}
        
        def evaluate_cluster_count(n_clusters):

            if len(reduced_embeddings) <= n_clusters:
                return n_clusters, {'calinski_harabasz': 0, 'davies_bouldin': float('inf')}
                
            # Run mini-batch k-means
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                batch_size=min(self.batch_size, len(reduced_embeddings)//10),
                max_iter=self.max_iter
            )
            cluster_labels = kmeans.fit_predict(reduced_embeddings)
            
            # Calculate scores
            ch_score = calinski_harabasz_score(reduced_embeddings, cluster_labels)
            db_score = davies_bouldin_score(reduced_embeddings, cluster_labels)
            
            return n_clusters, {'calinski_harabasz': ch_score, 'davies_bouldin': db_score}
        
        print(f"Testing {max_clusters - min_clusters + 1} cluster counts...")

        results = {}
        for n in tqdm(range(min_clusters, max_clusters + 1)):
            n_clusters, scores = evaluate_cluster_count(n)
            results[n_clusters] = scores

        print('Testing Done!')

        # Process results
        for k, v in results.items():
            scores = v
            n_clusters = k
            cluster_scores['calinski_harabasz'].append(scores['calinski_harabasz'])
            cluster_scores['davies_bouldin'].append(scores['davies_bouldin'])
            print(f"  {n_clusters} clusters: CH={scores['calinski_harabasz']:.2f}, DB={scores['davies_bouldin']:.2f}")
        
        # Find the best number of clusters (higher CH is better, lower DB is better)
        ch_best_idx = np.argmax(cluster_scores['calinski_harabasz'])
        db_best_idx = np.argmin(cluster_scores['davies_bouldin'])
        
        ch_best_n = min_clusters + ch_best_idx
        db_best_n = min_clusters + db_best_idx
        
        # Use Calinski-Harabasz as primary metric, Davies-Bouldin as secondary
        best_n_clusters = ch_best_n
        print(f"Best by Calinski-Harabasz: {ch_best_n} clusters (score: {cluster_scores['calinski_harabasz'][ch_best_idx]:.2f})")
        print(f"Best by Davies-Bouldin: {db_best_n} clusters (score: {cluster_scores['davies_bouldin'][db_best_idx]:.2f})")
        print(f"Selected optimal clusters: {best_n_clusters}")
        
        # Plot scores
        plt.figure(figsize=(12, 8))
        
        # Normalize scores for plotting
        ch_scores = np.array(cluster_scores['calinski_harabasz'])
        db_scores = np.array(cluster_scores['davies_bouldin'])
        
        # Min-max normalization
        ch_norm = (ch_scores - ch_scores.min()) / (ch_scores.max() - ch_scores.min() + 1e-10)
        # Invert DB so higher is better
        db_norm = 1 - (db_scores - db_scores.min()) / (db_scores.max() - db_scores.min() + 1e-10)
        
        plt.plot(range(min_clusters, max_clusters + 1), ch_norm, 'o-', label='Calinski-Harabasz (normalized)')
        plt.plot(range(min_clusters, max_clusters + 1), db_norm, 's-', label='Davies-Bouldin (inverted, normalized)')
        
        plt.axvline(x=best_n_clusters, color='red', linestyle='--', label=f'Selected: {best_n_clusters} clusters')
        
        plt.xlabel('Number of Clusters')
        plt.ylabel('Normalized Score (higher is better)')
        plt.title('Cluster Evaluation Scores')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'figures', 'cluster_scores.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        elapsed = time.time() - start_time
        print(f"Found optimal clusters in {elapsed:.2f} seconds")
        
        return best_n_clusters
    
    def cluster_embeddings(self, n_clusters=None, random_state=42):
        """
        Cluster embeddings using MiniBatchKMeans for scalability.
        
        Args:
            n_clusters (int): Number of clusters (if None, find optimal)
            random_state (int): Random seed for reproducibility
            
        Returns:
            DataFrame: Clustering results
        """
        print("Starting clustering process...")
        start_time = time.time()
        
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
            
            # Load embeddings in batches to save memory
            print(f"Loading {len(embedding_indices)} embeddings...")
            
            # Use batch loading to avoid memory issues
            batch_size = 100000  # Adjust based on available memory
            embeddings_subset = None
            
            for i in tqdm(range(0, len(embedding_indices), batch_size), desc="Loading embedding batches"):
                batch_indices = embedding_indices[i:i+batch_size]
                batch_embeddings = np.array([self.embeddings[idx] for idx in batch_indices])
                
                if embeddings_subset is None:
                    embeddings_subset = batch_embeddings
                else:
                    embeddings_subset = np.vstack((embeddings_subset, batch_embeddings))
        
        # Find optimal number of clusters if not specified
        if n_clusters is None:
            try:
                print("Finding optimal number of clusters...")
                n_clusters = self.find_optimal_clusters(
                    min_clusters=min(2, len(features_df) - 1),
                    max_clusters=min(20, len(features_df) // 500)  # Limit max clusters for very large datasets
                )
            except Exception as e:
                # Fallback if optimization fails
                print(f"Cluster optimization error: {str(e)}")
                n_clusters = min(10, len(features_df) // 1000)
                print(f"Falling back to {n_clusters} clusters")
        
        print(f"Reducing dimensions for {len(embeddings_subset)} samples...")
        
        # Reduce dimensions for full dataset
        reduced_embeddings, dim_reducer = self._reduce_dimensions(
            embeddings_subset, 
            target_dims=min(self.reduced_dims, embeddings_subset.shape[1] - 1, n_clusters * 3),
            method='TruncatedSVD'
        )
        
        print(f"Clustering data into {n_clusters} clusters using MiniBatchKMeans...")
        
        # Use MiniBatchKMeans for large datasets
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=min(self.batch_size, len(reduced_embeddings) // 10),
            random_state=random_state,
            max_iter=self.max_iter,
            verbose=1,  # Show progress
            compute_labels=True
        )
        
        # Fit the model with progress tracking
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        # Save the model
        self.cluster_model = kmeans
        
        # Add cluster labels to features
        features_df['cluster'] = cluster_labels
        
        # Reduce dimensionality for visualization
        print("Reducing dimensionality for visualization...")
        pca = PCA(n_components=2, random_state=random_state)
        
        # Sample data for PCA if very large
        if len(reduced_embeddings) > 100000:
            sample_size = 100000
            sample_idx = np.random.choice(len(reduced_embeddings), sample_size, replace=False)
            sample_embeddings = reduced_embeddings[sample_idx]
            pca.fit(sample_embeddings)
            embeddings_2d = pca.transform(reduced_embeddings)
        else:
            embeddings_2d = pca.fit_transform(reduced_embeddings)
        
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
        
        elapsed = time.time() - start_time
        print(f"Clustering completed in {elapsed:.2f} seconds!")
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
            
            # Count frames in this cluster (process in batches for large clusters)
            all_frames = []
            batch_size = 5000
            
            for i in range(0, len(cluster_df), batch_size):
                batch = cluster_df.iloc[i:i+batch_size]
                for frames in batch['frame_counts']:
                    if isinstance(frames, dict) and frames:
                        all_frames.extend(frames.keys())
            
            frame_counts = Counter(all_frames)
            # Convert to Python native types for JSON serialization
            top_frames = [(str(frame), int(count)) for frame, count in frame_counts.most_common(5)]
            
            # Count metaphor types - convert to native int
            metaphor_count = int(cluster_df['has_metaphor'].sum())
            novel_metaphor_count = int(cluster_df['has_novel_metaphor'].sum())
            
            # Count countries
            country_counts = {str(k): int(v) for k, v in cluster_df['country'].value_counts().head(3).to_dict().items()}
            
            # Sample sentences (avoid memory issues with very large clusters)
            if len(cluster_df) > 1000:
                sample_indices = np.random.choice(cluster_df.index, min(3, len(cluster_df)), replace=False)
                sample_sentences = [str(s) for s in cluster_df.loc[sample_indices, 'sentence'].tolist()]
            else:
                sample_sentences = [str(s) for s in cluster_df['sentence'].sample(min(3, len(cluster_df))).tolist()]
            
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
                'sample_sentences': sample_sentences
            }
            
            cluster_summaries.append(summary)
        
        # Save summaries
        summaries_path = os.path.join(self.results_dir, 'tables', 'cluster_summaries.json')
        with open(summaries_path, 'w') as f:
            json.dump(cluster_summaries, f, indent=2)
        
        print(f"Saved cluster summaries to {summaries_path}")
        
        # Print top frames per cluster
        print("Top frames per cluster:")
        for summary in cluster_summaries:
            print(f"Cluster {summary['cluster_id']} (size: {summary['size']}):")
            for frame, count in summary['top_frames']:
                print(f"  - {frame}: {count}")
    
    def _visualize_clusters(self):
        """
        Visualize clustering results.
        """
        if self.cluster_results is None:
            raise ValueError("No clustering results available")
        
        print("Visualizing clusters...")
        
        # Sample data for visualization if very large
        if len(self.cluster_results) > 10000:
            sample_size = 10000
            visualize_df = self.cluster_results.sample(sample_size, random_state=42)
            print(f"Sampled {sample_size} points for visualization (from {len(self.cluster_results)} total)")
        else:
            visualize_df = self.cluster_results
        
        # Create scatter plot of clusters
        plt.figure(figsize=(12, 10))
        
        # Plot each cluster with a different color
        for cluster_id in sorted(visualize_df['cluster'].unique()):
            cluster_df = visualize_df[visualize_df['cluster'] == cluster_id]
            plt.scatter(cluster_df['x'], cluster_df['y'], label=f'Cluster {cluster_id}', alpha=0.7, s=10)
        
        # Add cluster centers if available
        if hasattr(self.cluster_model, 'cluster_centers_') and self.pca_model is not None:
            centers = self.cluster_model.cluster_centers_
            centers_2d = self.pca_model.transform(centers)
            plt.scatter(centers_2d[:, 0], centers_2d[:, 1], s=100, c='black', marker='X', label='Cluster Centers')
        
        plt.title('AI Reference Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'figures', 'cluster_visualization.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
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
        
        # Skip if no countries
        if not top_countries:
            print("No country data available for country visualization")
            return
        
        # Create cluster distribution by country
        country_cluster_counts = defaultdict(lambda: defaultdict(int))
        
        # Process in batches for memory efficiency
        batch_size = 10000
        for i in range(0, len(self.cluster_results), batch_size):
            batch = self.cluster_results.iloc[i:i+batch_size]
            for _, row in batch.iterrows():
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
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.results_dir, 'figures', 'cluster_by_country.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
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