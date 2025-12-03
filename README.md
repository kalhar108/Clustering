# Clustering Algorithms - Complete Implementation Suite

A comprehensive collection of 9 Google Colab notebooks demonstrating various clustering algorithms, from classical approaches to state-of-the-art deep learning embeddings.

## 📚 Table of Contents

- [Overview](#overview)
- [Notebooks Description](#notebooks-description)
- [Installation & Usage](#installation--usage)
- [Datasets Used](#datasets-used)
- [Key Features](#key-features)
- [Evaluation Metrics](#evaluation-metrics)
- [Requirements](#requirements)
- [Learning Outcomes](#learning-outcomes)
- [Resources](#resources)

---

## 🎯 Overview

This repository contains 9 production-ready Jupyter notebooks covering:
- **Classical Clustering**: K-Means, Hierarchical, GMM, DBSCAN
- **Specialized Applications**: Time series, documents, images, audio
- **Advanced Techniques**: LLM embeddings, multimodal models, anomaly detection

All notebooks are designed to run seamlessly on Google Colab with minimal setup.

---

## 📓 Notebooks Description

### 1️⃣ a_kmeans_from_scratch.ipynb
**K-Means Clustering - Built from Scratch**

- ✨ Complete implementation without sklearn.KMeans
- 📊 Dataset: Iris (150 samples, 4 features, 3 classes)
- 🔧 Features:
  - Custom KMeans class with Euclidean distance
  - Elbow method for optimal K
  - Convergence visualization
  - Cluster quality metrics
- 📈 Metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz, Inertia
- ⏱️ Runtime: ~2 minutes

**Key Learning**: Understand the mathematics behind K-Means clustering

---

### 2️⃣ b_hierarchical_clustering.ipynb
**Hierarchical Clustering with Dendrograms**

- ✨ Multiple linkage methods comparison
- 📊 Dataset: Wine (178 samples, 13 features, 3 classes)
- 🔧 Features:
  - Ward, Complete, Average, Single linkage
  - Beautiful dendrograms
  - Linkage method comparison
  - Optimal cluster selection
- 📈 Metrics: Silhouette scores across methods
- ⏱️ Runtime: ~2 minutes

**Key Learning**: Hierarchical structure and linkage strategies

---

### 3️⃣ c_gaussian_mixture_models.ipynb
**GMM - Probabilistic Clustering**

- ✨ Soft clustering with probability distributions
- 📊 Dataset: Synthetic blobs (500 samples, 2 features, 4 clusters)
- 🔧 Features:
  - Full, Tied, Diagonal, Spherical covariance types
  - BIC/AIC model selection
  - Uncertainty visualization
  - Cluster probability heatmaps
- 📈 Metrics: BIC, AIC, Log-Likelihood, Silhouette
- ⏱️ Runtime: ~3 minutes

**Key Learning**: Probabilistic approach and model selection criteria

---

### 4️⃣ d_dbscan_pycaret.ipynb
**DBSCAN - Density-Based Clustering with PyCaret**

- ✨ Automated ML pipeline with PyCaret
- 📊 Dataset: Moon shapes with outliers (320 samples)
- 🔧 Features:
  - Automatic noise detection
  - Parameter tuning (eps, min_samples)
  - Model comparison
  - Outlier identification
- 📈 Metrics: Noise percentage, cluster count, silhouette
- ⏱️ Runtime: ~3 minutes

**Key Learning**: Density-based clustering and outlier detection

---

### 5️⃣ e_anomaly_detection_pyod.ipynb
**Anomaly Detection with PyOD**

- ✨ Multiple anomaly detection algorithms
- 📊 Dataset: Imbalanced dataset (1000 samples, 5% anomalies)
- 🔧 Features:
  - 5 algorithms: KNN, LOF, IForest, OCSVM, ECOD
  - ROC curve analysis
  - Algorithm performance comparison
  - Anomaly score distributions
- 📈 Metrics: ROC-AUC, Precision, Recall, F1-Score
- ⏱️ Runtime: ~4 minutes

**Key Learning**: Anomaly detection techniques and evaluation

---

### 6️⃣ f_timeseries_clustering.ipynb
**Time Series Clustering with DTW**

- ✨ Shape-based similarity for temporal data
- 📊 Dataset: Synthetic time series (120 series, 100 timestamps)
- 🔧 Features:
  - Dynamic Time Warping (DTW)
  - Soft-DTW comparison
  - Cluster center visualization
  - Pattern recognition
- 📈 Metrics: Silhouette, Davies-Bouldin, ARI, NMI
- ⏱️ Runtime: ~5 minutes

**Key Learning**: Time series analysis and DTW distance metric

---

### 7️⃣ g_document_clustering.ipynb
**Document Clustering with LLM Embeddings**

- ✨ State-of-the-art Sentence-BERT embeddings
- 📊 Dataset: 20 Newsgroups (400 documents, 4 categories)
- 🔧 Features:
  - all-MiniLM-L6-v2 model (384-dim embeddings)
  - UMAP dimensionality reduction
  - Representative document extraction
  - Semantic clustering
- 📈 Metrics: Silhouette, Davies-Bouldin, ARI
- ⏱️ Runtime: ~5 minutes

**Key Learning**: Modern NLP embeddings for text clustering

---

### 8️⃣ h_image_clustering_imagebind.ipynb
**Image Clustering with ImageBind Multimodal Embeddings**

- ✨ Meta's ImageBind for unified embeddings
- 📊 Dataset: CIFAR-10 subset (500 images, 10 classes)
- 🔧 Features:
  - 1024-dimensional multimodal embeddings
  - Visual cluster analysis
  - Confusion matrix
  - Sample visualization
- 📈 Metrics: Silhouette, Davies-Bouldin, ARI
- ⏱️ Runtime: ~8 minutes (includes model download)

**Key Learning**: Multimodal deep learning for image understanding

---

### 9️⃣ i_audio_clustering_imagebind.ipynb
**Audio Clustering with ImageBind**

- ✨ Multimodal audio embeddings
- 📊 Dataset: Synthetic audio (100 samples, 4 types)
- 🔧 Features:
  - Waveform and spectrogram visualization
  - Sine, chirp, noise, pulse patterns
  - Audio feature extraction
  - Cross-modal capabilities
- 📈 Metrics: Silhouette, Davies-Bouldin, ARI
- ⏱️ Runtime: ~6 minutes

**Key Learning**: Audio signal processing and embedding-based clustering

---

## 🚀 Installation & Usage

### Option 1: Google Colab (Recommended)

1. **Upload to Google Drive**
   ```
   Upload the notebooks to your Google Drive
   ```

2. **Open in Colab**
   ```
   Right-click notebook → Open with → Google Colaboratory
   ```

3. **Run All Cells**
   ```
   Runtime → Run all (Ctrl+F9)
   ```

### Option 2: Local Jupyter

1. **Install Jupyter**
   ```bash
   pip install jupyter
   ```

2. **Install Dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   pip install pycaret pyod tslearn sentence-transformers
   pip install torch torchvision torchaudio librosa
   ```

3. **Run Notebook**
   ```bash
   jupyter notebook notebook_name.ipynb
   ```

---

## 📊 Datasets Used

| Notebook | Dataset | Source | Size | Type |
|----------|---------|--------|------|------|
| a | Iris | sklearn | 150 samples | Tabular |
| b | Wine | sklearn | 178 samples | Tabular |
| c | Synthetic Blobs | sklearn | 500 samples | Tabular |
| d | Moon Shapes | sklearn | 320 samples | Tabular |
| e | Imbalanced | sklearn | 1000 samples | Tabular |
| f | Synthetic TS | Generated | 120 series | Time Series |
| g | 20 Newsgroups | sklearn | 400 docs | Text |
| h | CIFAR-10 | torchvision | 500 images | Images |
| i | Synthetic Audio | Generated | 100 samples | Audio |

---

## ⭐ Key Features

### Common to All Notebooks:

✅ **Complete & Self-Contained**
- No external data files required
- All dependencies auto-installed
- Ready to run on Google Colab

✅ **Production-Quality Code**
- Clean, well-documented code
- Professional visualizations
- Error handling included

✅ **Comprehensive Evaluation**
- Multiple quality metrics
- Visualization of results
- Comparison with ground truth

✅ **Educational Content**
- Detailed explanations
- Algorithm summaries
- Best practices included

---

## 📈 Evaluation Metrics

### Clustering Quality Metrics

1. **Silhouette Score** (Range: -1 to 1, Higher is Better)
   - Measures cluster cohesion and separation
   - Values > 0.5 indicate good clustering

2. **Davies-Bouldin Index** (Range: 0 to ∞, Lower is Better)
   - Ratio of within-cluster to between-cluster distances
   - Values < 1.0 indicate good separation

3. **Calinski-Harabasz Index** (Range: 0 to ∞, Higher is Better)
   - Ratio of between-cluster to within-cluster dispersion
   - Higher values indicate better-defined clusters

4. **Adjusted Rand Index (ARI)** (Range: -1 to 1)
   - Agreement with ground truth labels
   - 1.0 indicates perfect agreement

5. **Normalized Mutual Information (NMI)** (Range: 0 to 1)
   - Information shared between clusterings
   - 1.0 indicates perfect correspondence

### Model Selection Metrics (GMM)

- **BIC (Bayesian Information Criterion)**: Lower is better
- **AIC (Akaike Information Criterion)**: Lower is better
- **Log-Likelihood**: Higher is better

### Anomaly Detection Metrics

- **ROC-AUC**: Area under ROC curve (0 to 1, higher is better)
- **Precision**: Accuracy of anomaly predictions
- **Recall**: Coverage of actual anomalies
- **F1-Score**: Harmonic mean of precision and recall

---

## 🔧 Requirements

### Python Version
- Python 3.7 or higher

### Core Libraries
```
numpy>=1.19.0
pandas>=1.1.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.5.0
```

### Specialized Libraries
```
# Notebook d: PyCaret
pycaret>=2.3.0

# Notebook e: PyOD
pyod>=0.9.0

# Notebook f: Time Series
tslearn>=0.5.0

# Notebook g: Document Clustering
sentence-transformers>=2.0.0
umap-learn>=0.5.0

# Notebooks h, i: Multimodal
torch>=1.10.0
torchvision>=0.11.0
torchaudio>=0.10.0
librosa>=0.9.0
```

### Hardware Requirements

| Notebook | CPU | RAM | GPU | Runtime |
|----------|-----|-----|-----|---------|
| a-e | Any | 2GB | No | 2-4 min |
| f | Any | 4GB | No | 5 min |
| g | Any | 4GB | No | 5 min |
| h | Any | 8GB | Optional | 8 min |
| i | Any | 8GB | Optional | 6 min |

**Note**: All notebooks run fine on Google Colab free tier!

---

## 🎓 Learning Outcomes

After completing these notebooks, you will understand:

### 1. Classical Clustering Algorithms
- K-Means: Centroid-based partitioning
- Hierarchical: Tree-based clustering
- GMM: Probabilistic soft clustering
- DBSCAN: Density-based clustering

### 2. Specialized Applications
- Time series pattern recognition
- Document semantic grouping
- Image content organization
- Audio signal categorization

### 3. Modern Techniques
- Transformer-based embeddings
- Multimodal learning
- Transfer learning for clustering
- Zero-shot clustering

### 4. Best Practices
- Choosing the right algorithm
- Parameter tuning strategies
- Evaluation methodology
- Visualization techniques

### 5. Real-World Skills
- End-to-end ML pipelines
- Working with different data modalities
- Production-ready implementations
- Scientific computing with Python

---

## 📖 Resources

### Documentation
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [PyCaret Documentation](https://pycaret.org/)
- [PyOD Documentation](https://pyod.readthedocs.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [ImageBind GitHub](https://github.com/facebookresearch/ImageBind)

### Papers
- K-Means: MacQueen (1967) - "Some methods for classification and analysis of multivariate observations"
- DBSCAN: Ester et al. (1996) - "A density-based algorithm for discovering clusters"
- GMM: Dempster et al. (1977) - "Maximum likelihood from incomplete data via the EM algorithm"
- DTW: Sakoe & Chiba (1978) - "Dynamic programming algorithm optimization for spoken word recognition"
- Sentence-BERT: Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- ImageBind: Girdhar et al. (2023) - "ImageBind: One Embedding Space To Bind Them All"

### Datasets
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Papers With Code Datasets](https://paperswithcode.com/datasets)
- [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add new clustering algorithms
- Enhance visualizations
- Improve documentation

---

## 📝 Citation

If you use these notebooks in your research or projects, please cite:

```bibtex
@misc{clustering_notebooks_2024,
  title={Comprehensive Clustering Algorithms Implementation Suite},
  author={ML Assignment},
  year={2024},
  howpublished={GitHub Repository}
}
```

---

## ⚖️ License

This project is provided for educational purposes. Please check individual library licenses for production use.

---

## 📧 Contact

For questions or feedback:
- Open an issue in the repository
- Check individual notebook documentation
- Refer to the official library documentation

---

## 🌟 Acknowledgments

- **Scikit-learn**: Core ML library
- **PyCaret**: Automated ML framework
- **PyOD**: Anomaly detection toolkit
- **Sentence-Transformers**: State-of-the-art text embeddings
- **Meta AI**: ImageBind multimodal model
- **Google Colab**: Free GPU/TPU access

---

## 📌 Quick Start Guide

**New to Clustering?** Start with:
1. **a_kmeans_from_scratch.ipynb** - Understand the basics
2. **b_hierarchical_clustering.ipynb** - Learn hierarchical methods
3. **c_gaussian_mixture_models.ipynb** - Explore probabilistic clustering

**Working with Specific Data?** Jump to:
- **Time Series**: f_timeseries_clustering.ipynb
- **Text**: g_document_clustering.ipynb
- **Images**: h_image_clustering_imagebind.ipynb
- **Audio**: i_audio_clustering_imagebind.ipynb

**Need Anomaly Detection?** Check out:
- **e_anomaly_detection_pyod.ipynb**

**Want Automation?** Try:
- **d_dbscan_pycaret.ipynb**

---

## 🎯 Success Tips

1. **Start Simple**: Begin with classical algorithms before advanced techniques
2. **Understand Metrics**: Know what each metric measures
3. **Visualize Results**: Always plot your clusters
4. **Compare Methods**: Try multiple algorithms on the same data
5. **Validate Results**: Use domain knowledge to verify clustering quality
6. **Parameter Tuning**: Experiment with different parameters
7. **Scale Your Data**: Normalization often improves results
8. **Handle Outliers**: Consider their impact on clustering

---

## 🔥 Advanced Topics

For further exploration:
- Spectral clustering
- Fuzzy C-means
- Affinity propagation
- OPTICS (Ordering Points To Identify Clustering Structure)
- Self-organizing maps (SOM)
- Deep clustering with autoencoders
- Graph-based clustering
- Consensus clustering

---

**Happy Clustering! 🎉**

Made with ❤️ for the ML community
