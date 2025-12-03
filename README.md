# Clustering Techniques:

**Author:** Kalhar Mayurbhai Patel  
**SJSU ID:** 019140511  
**Course Project:** Machine Learning - Clustering Methods

---

## 📋 Table of Contents

1. [Project Overview]
2. [Repository Structure]
3. [Notebooks Description]
4. [Technologies and Libraries]
5. [Installation Guide]
6. [Usage Instructions]
7. [Key Concepts and Algorithms]
8. [Results and Visualizations]
9. [Learning Outcomes]
10. [References]

---

## 🎯 Project Overview

This repository contains a comprehensive collection of **9 Jupyter notebooks** demonstrating advanced clustering techniques across various domains and data types. The project showcases implementations ranging from classical algorithms built from scratch to state-of-the-art deep learning approaches for multimodal data clustering.

### Project Highlights

- **Classical to Modern**: From scratch implementations to LLM-based clustering
- **Multimodal Data**: Text, images, audio, and time series
- **Practical Applications**: Real-world clustering scenarios and use cases
- **Comprehensive Coverage**: 9 different clustering methodologies
- **Production-Ready**: Industry-standard libraries and best practices

---

## 📁 Repository Structure

```
clustering-project/
│
├── a_kmeans_from_scratch.ipynb          # K-Means algorithm implementation
├── b_hierarchical_clustering.ipynb       # Hierarchical clustering methods
├── c_gaussian_mixture_models.ipynb       # GMM and probabilistic clustering
├── d_dbscan_pycaret.ipynb               # Density-based clustering with PyCaret
├── e_anomaly_detection_pyod.ipynb       # Outlier detection with PyOD
├── f_timeseries_clustering.ipynb        # Time series clustering techniques
├── g_document_clustering_llm.ipynb      # Document clustering with embeddings
├── h_image_clustering_imagebind.ipynb   # Image clustering with CLIP/ImageBind
├── i_audio_clustering_embeddings.ipynb  # Audio clustering with deep learning
└── README.md                             # This file
```

---

## 📚 Notebooks Description

### 1. **K-Means Clustering from Scratch** (`a_kmeans_from_scratch.ipynb`)

**Objective:** Implement the K-Means clustering algorithm from scratch to understand its fundamental mechanics.

**Key Topics:**
- Algorithm implementation without using sklearn's KMeans
- Centroid initialization strategies
- Iterative assignment and update steps
- Convergence criteria and stopping conditions
- Performance comparison with sklearn implementation

**Techniques:**
- Random centroid initialization
- Euclidean distance calculations
- Iterative optimization
- Visualization of cluster formation

**Learning Outcomes:**
- Deep understanding of K-Means internals
- Implementation of optimization algorithms
- Debugging and validating custom implementations

---

### 2. **Hierarchical Clustering** (`b_hierarchical_clustering.ipynb`)

**Objective:** Explore hierarchical clustering methods and dendrogram analysis.

**Key Topics:**
- Agglomerative (bottom-up) clustering
- Divisive (top-down) clustering
- Linkage methods (single, complete, average, Ward)
- Dendrogram visualization and interpretation
- Cophenetic correlation coefficient

**Techniques:**
- Distance matrix computation
- Linkage criteria comparison
- Dendrogram cutting at different heights
- Cluster validity assessment

**Learning Outcomes:**
- Understanding hierarchical relationships in data
- Choosing optimal linkage methods
- Interpreting dendrograms for cluster selection

---

### 3. **Gaussian Mixture Models** (`c_gaussian_mixture_models.ipynb`)

**Objective:** Implement probabilistic clustering using Gaussian Mixture Models.

**Key Topics:**
- Expectation-Maximization (EM) algorithm
- Soft vs. hard clustering
- Covariance types (full, tied, diagonal, spherical)
- Probability density estimation
- Model selection using BIC/AIC

**Techniques:**
- EM algorithm for parameter estimation
- Posterior probability calculation
- Covariance structure analysis
- Model comparison metrics

**Learning Outcomes:**
- Probabilistic approach to clustering
- Understanding mixture models
- Model selection and validation

---

### 4. **DBSCAN with PyCaret** (`d_dbscan_pycaret.ipynb`)

**Objective:** Apply density-based clustering to discover arbitrary-shaped clusters and identify outliers.

**Key Topics:**
- Density-Based Spatial Clustering (DBSCAN)
- Core points, border points, and noise
- Epsilon (eps) and MinPts parameter tuning
- PyCaret clustering module
- Handling non-spherical clusters

**Techniques:**
- Epsilon neighborhood search
- Core point identification
- Cluster expansion algorithm
- Outlier detection

**Learning Outcomes:**
- Clustering arbitrary-shaped data
- Automatic outlier detection
- Parameter sensitivity analysis
- Using AutoML tools (PyCaret)

---

### 5. **Anomaly Detection with PyOD** (`e_anomaly_detection_pyod.ipynb`)

**Objective:** Detect anomalies and outliers using multiple algorithms from the PyOD library.

**Key Topics:**
- Univariate and multivariate anomaly detection
- Multiple detection algorithms (LOF, Isolation Forest, OCSVM)
- Ensemble methods for robust detection
- Anomaly scoring and ranking
- Threshold selection strategies

**Techniques:**
- Local Outlier Factor (LOF)
- Isolation Forest
- One-Class SVM
- Auto-Encoder based detection
- COPOD (Copula-based Outlier Detection)

**Learning Outcomes:**
- Understanding different anomaly detection paradigms
- Comparing algorithm performance
- Ensemble approach for improved accuracy
- Real-world anomaly detection applications

---

### 6. **Time Series Clustering** (`f_timeseries_clustering.ipynb`)

**Objective:** Cluster temporal data using specialized distance metrics and techniques.

**Key Topics:**
- Dynamic Time Warping (DTW)
- Time series feature extraction
- Shape-based clustering
- Temporal pattern recognition
- tslearn library usage

**Techniques:**
- DTW distance computation
- Time series normalization
- K-Means with DTW metric
- Feature-based representations
- Temporal alignment

**Learning Outcomes:**
- Handling temporal dependencies
- Elastic distance measures
- Time series preprocessing
- Domain-specific clustering

---

### 7. **Document Clustering with LLM Embeddings** (`g_document_clustering_llm.ipynb`)

**Objective:** Cluster text documents using state-of-the-art language model embeddings.

**Key Topics:**
- Sentence-Transformers embeddings
- Semantic similarity in embedding space
- HDBSCAN for text clustering
- UMAP dimensionality reduction
- Document representation learning

**Techniques:**
- all-MiniLM-L6-v2 model
- all-mpnet-base-v2 model
- Cosine similarity in embedding space
- HDBSCAN clustering
- UMAP visualization

**Learning Outcomes:**
- Modern NLP embedding techniques
- Semantic document clustering
- Transfer learning for text
- Visualization of high-dimensional embeddings

---

### 8. **Image Clustering with ImageBind/CLIP** (`h_image_clustering_imagebind.ipynb`)

**Objective:** Cluster images using multimodal embeddings from vision-language models.

**Key Topics:**
- CLIP (Contrastive Language-Image Pre-training)
- ImageBind multimodal embeddings
- Vision transformer architectures
- Zero-shot image understanding
- Cross-modal embedding space

**Techniques:**
- CLIP model for image encoding
- Feature extraction from pretrained models
- K-Means on image embeddings
- Visual similarity measurement
- Zero-shot classification capabilities

**Learning Outcomes:**
- Multimodal deep learning
- Vision-language models
- Transfer learning for computer vision
- Embedding-based image analysis

---

### 9. **Audio Clustering with Deep Learning** (`i_audio_clustering_embeddings.ipynb`)

**Objective:** Cluster audio data using traditional features and deep learning embeddings.

**Key Topics:**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Wav2Vec2 embeddings
- HuBERT representations
- CLAP (Audio-Language model)
- Audio feature engineering

**Techniques:**
- Traditional audio feature extraction
- Pretrained audio model embeddings
- Librosa for audio processing
- Transformers for audio
- Acoustic feature clustering

**Learning Outcomes:**
- Audio signal processing
- Deep learning for audio
- Feature engineering for sound
- Pretrained audio models

---

## 🛠️ Technologies and Libraries

### Core Python Libraries
```python
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
```

### Clustering and ML Libraries
```python
- PyCaret
- PyOD (Python Outlier Detection)
- HDBSCAN
- tslearn (Time Series)
```

### Deep Learning and NLP
```python
- PyTorch
- Transformers (Hugging Face)
- Sentence-Transformers
- CLIP (OpenAI)
```

### Audio Processing
```python
- Librosa
- Wav2Vec2
- AudioCraft
```

### Visualization
```python
- UMAP
- Plotly
- Matplotlib
- Seaborn
```

---

## 💻 Installation Guide

### Method 1: Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n clustering-env python=3.9
conda activate clustering-env

# Install core libraries
conda install numpy pandas matplotlib seaborn scikit-learn jupyter

# Install via pip
pip install pycaret pyod hdbscan tslearn sentence-transformers
pip install transformers torch torchvision librosa umap-learn
```

### Method 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all requirements
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
pip install pycaret pyod hdbscan tslearn
pip install sentence-transformers transformers torch torchvision
pip install librosa umap-learn plotly
```

### Method 3: Using requirements.txt

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
pycaret>=3.0.0
pyod>=1.0.0
hdbscan>=0.8.0
tslearn>=0.5.0
sentence-transformers>=2.2.0
transformers>=4.25.0
torch>=1.13.0
torchvision>=0.14.0
librosa>=0.9.0
umap-learn>=0.5.0
plotly>=5.0.0
```

---

## 🚀 Usage Instructions

### Running Individual Notebooks

1. **Navigate to the project directory:**
   ```bash
   cd clustering-project
   ```

2. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open desired notebook** and run cells sequentially

### Recommended Learning Path

**For Beginners:**
1. Start with `a_kmeans_from_scratch.ipynb` - Understand basic clustering
2. Move to `b_hierarchical_clustering.ipynb` - Learn hierarchical methods
3. Try `c_gaussian_mixture_models.ipynb` - Explore probabilistic clustering
4. Practice with `d_dbscan_pycaret.ipynb` - Density-based methods

**For Intermediate Users:**
1. Begin with `e_anomaly_detection_pyod.ipynb` - Outlier detection
2. Explore `f_timeseries_clustering.ipynb` - Temporal data
3. Study `g_document_clustering_llm.ipynb` - Text clustering

**For Advanced Users:**
1. Dive into `h_image_clustering_imagebind.ipynb` - Vision models
2. Master `i_audio_clustering_embeddings.ipynb` - Audio analysis
3. Integrate multiple notebooks for multimodal projects

### Running All Notebooks Programmatically

```python
import subprocess
import glob

notebooks = sorted(glob.glob('*.ipynb'))
for notebook in notebooks:
    print(f"Executing {notebook}...")
    subprocess.run(['jupyter', 'nbconvert', '--to', 'notebook', 
                   '--execute', notebook])
```

---

## 🧠 Key Concepts and Algorithms

### Clustering Fundamentals

**1. Distance Metrics**
- Euclidean Distance
- Manhattan Distance
- Cosine Similarity
- Dynamic Time Warping (DTW)

**2. Cluster Validation**
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Elbow Method

**3. Dimensionality Reduction**
- PCA (Principal Component Analysis)
- t-SNE
- UMAP

### Algorithm Comparison

| Algorithm | Type | Pros | Cons | Best For |
|-----------|------|------|------|----------|
| K-Means | Partitioning | Fast, simple | Requires K, spherical clusters | Large datasets, known K |
| Hierarchical | Hierarchical | No K needed, dendrogram | Computationally expensive | Small datasets, hierarchy |
| GMM | Probabilistic | Soft clustering, flexible | EM convergence issues | Overlapping clusters |
| DBSCAN | Density-based | Arbitrary shapes, outliers | Parameter sensitive | Non-spherical clusters |
| HDBSCAN | Density-based | Auto parameters | Complex | Varying density |

---

## 📊 Results and Visualizations

Each notebook produces various visualizations including:

- **Scatter plots** with cluster assignments
- **Dendrograms** for hierarchical relationships
- **Silhouette plots** for cluster quality
- **Confusion matrices** for validation
- **t-SNE/UMAP plots** for high-dimensional data
- **Distance heatmaps** for similarity analysis
- **Time series plots** with cluster labels
- **Embedding visualizations** in 2D/3D space

### Example Results

**K-Means Performance:**
- Typical convergence: 10-50 iterations
- Silhouette score range: 0.4-0.7 (depending on data)

**DBSCAN Outlier Detection:**
- Noise detection rate: 5-15% of data points
- Works well with eps tuning

**Document Clustering Accuracy:**
- Semantic clustering with 80%+ coherence
- Clear separation in embedding space

---

## 🎓 Learning Outcomes

Upon completing this project, you will have gained:

### Technical Skills
- ✅ Implementation of clustering algorithms from scratch
- ✅ Usage of industry-standard ML libraries
- ✅ Working with multimodal data (text, image, audio)
- ✅ Deep learning model integration
- ✅ Feature engineering techniques
- ✅ Model evaluation and validation

### Conceptual Understanding
- ✅ Differences between clustering paradigms
- ✅ When to use which algorithm
- ✅ Parameter tuning strategies
- ✅ Handling different data types
- ✅ Evaluation metrics interpretation
- ✅ Scalability considerations

### Practical Applications
- ✅ Customer segmentation
- ✅ Anomaly detection in systems
- ✅ Document organization
- ✅ Image categorization
- ✅ Audio classification
- ✅ Time series pattern recognition

---

## 🔬 Advanced Topics Covered

### 1. **Ensemble Clustering**
- Combining multiple clustering algorithms
- Consensus clustering techniques
- Improving robustness

### 2. **Transfer Learning**
- Using pretrained models for embeddings
- Fine-tuning strategies
- Zero-shot clustering

### 3. **Multimodal Learning**
- Cross-modal embeddings (CLIP, ImageBind)
- Unified embedding spaces
- Multimodal fusion strategies

### 4. **AutoML Integration**
- PyCaret for automated clustering
- Hyperparameter optimization
- Model comparison frameworks

---

## 📝 Best Practices Demonstrated

1. **Data Preprocessing**
   - Normalization and standardization
   - Handling missing values
   - Feature scaling

2. **Model Selection**
   - Cross-validation strategies
   - Hyperparameter tuning
   - Ensemble methods

3. **Code Organization**
   - Modular function design
   - Clear documentation
   - Reproducible results

4. **Visualization**
   - Informative plots
   - Multiple perspectives
   - Interactive visualizations

---

## 🔍 Use Cases and Applications

### Business Applications
- **Customer Segmentation**: Group customers by behavior
- **Market Basket Analysis**: Product recommendation
- **Fraud Detection**: Identify unusual transactions

### Healthcare
- **Patient Stratification**: Group patients by symptoms
- **Disease Subtyping**: Discover disease variants
- **Medical Image Analysis**: Cluster similar cases

### Technology
- **Log Analysis**: Identify system anomalies
- **User Behavior**: App usage patterns
- **Network Security**: Detect intrusions

### Content Management
- **Document Organization**: Auto-categorize files
- **Image Gallery**: Automatic photo albums
- **Music Recommendation**: Similar song discovery

---

## 🐛 Troubleshooting

### Common Issues and Solutions

**1. Memory Errors with Large Datasets**
```python
# Use mini-batch K-Means
from sklearn.cluster import MiniBatchKMeans
model = MiniBatchKMeans(n_clusters=8, batch_size=1000)
```

**2. Slow Performance**
```python
# Use approximate algorithms or dimensionality reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
```

**3. Installation Issues**
- Use conda for complex dependencies
- Check CUDA compatibility for GPU operations
- Use Python 3.8+ for best compatibility

**4. Model Convergence Issues**
- Increase max_iter parameter
- Try different initialization methods
- Scale features appropriately

---

## 📚 References

### Books
1. "Pattern Recognition and Machine Learning" - Christopher Bishop
2. "Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani
3. "Hands-On Machine Learning" - Aurélien Géron

### Papers
1. MacQueen, J. (1967). "K-means clustering"
2. Ester et al. (1996). "A density-based algorithm (DBSCAN)"
3. Campello et al. (2013). "HDBSCAN: Hierarchical DBSCAN"

### Online Resources
- Scikit-learn Documentation: https://scikit-learn.org/
- PyCaret Documentation: https://pycaret.org/
- PyOD Documentation: https://pyod.readthedocs.io/
- Hugging Face Transformers: https://huggingface.co/

### Datasets
- UCI Machine Learning Repository
- Kaggle Datasets
- OpenML
- TensorFlow Datasets

---

## 👨‍💻 Author Information

**Name:** Kalhar Mayurbhai Patel  
**SJSU ID:** 019140511  
**Email:** [Your Email]  
**LinkedIn:** [Your LinkedIn]  
**GitHub:** [Your GitHub]

---

## 📄 License

This project is created for educational purposes as part of coursework at San Jose State University.

---

## 🙏 Acknowledgments

- San Jose State University, Department of Computer Science
- Course Instructor and Teaching Assistants
- Open-source community for libraries and tools
- Research papers and authors cited throughout

---

## 🔮 Future Enhancements

Potential extensions to this project:

1. **Deep Clustering Networks**
   - Implement DEC (Deep Embedded Clustering)
   - AutoEncoder-based clustering

2. **Graph Clustering**
   - Community detection algorithms
   - Spectral clustering

3. **Online/Streaming Clustering**
   - Real-time data clustering
   - Incremental learning

4. **Explainable AI**
   - Cluster interpretation tools
   - Feature importance analysis

5. **Production Deployment**
   - API development
   - Model serving with FastAPI
   - Docker containerization

---

## 📞 Contact and Support

For questions, suggestions, or collaboration:

- **Course-Related Queries:** Contact via Canvas
- **Technical Issues:** Open an issue in the repository
- **Collaborations:** Reach out via email

---

## 📊 Project Statistics

- **Total Notebooks:** 9
- **Lines of Code:** ~2000+
- **Algorithms Implemented:** 15+
- **Visualization Types:** 20+
- **Libraries Used:** 25+
- **Data Modalities:** 5 (Numerical, Text, Image, Audio, Time Series)

---

## ✅ Completion Checklist

- [x] K-Means from scratch implementation
- [x] Hierarchical clustering with dendrograms
- [x] Gaussian Mixture Models
- [x] DBSCAN with PyCaret
- [x] Anomaly detection with PyOD
- [x] Time series clustering
- [x] Document clustering with LLMs
- [x] Image clustering with CLIP
- [x] Audio clustering with embeddings
- [x] Comprehensive documentation

---

**Last Updated:** December 2024  
**Version:** 1.0  
**Status:** Complete ✅

---

*This README provides a comprehensive guide to the clustering project collection. For detailed implementation and code walkthroughs, please refer to individual notebooks.*
