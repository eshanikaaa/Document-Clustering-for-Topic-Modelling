### Project Title: Topic Modeling and Clustering on Real-world Text Data using LDA and K-Means

**Objective:** To uncover latent topics and group similar documents from the 20 Newsgroups dataset using Latent Dirichlet Allocation (LDA) and K-Means clustering. This helps in organizing and summarizing large unstructured text data efficiently.

**Dataset:** [20 Newsgroups](http://archive.ics.uci.edu/ml/datasets/Twenty+Newsgroups) — 20,000 documents from 20 categories such as sports, politics, science, etc.

**Approach:** Applied two unsupervised learning methods:  
- **K-Means** on TF-IDF vectors for document clustering  
- **LDA** on Bag-of-Words for topic modeling

**Live App**
🔗

Features
-Input any paragraph or document
-Get predicted topic cluster (KMeans) and dominant topic (LDA)
-View full topic distribution from LDA
-Clean and minimal UI powered by Streamlit
-Built using pre-trained models for instant predictions

**Tech Stack**
-Python 3
-Streamlit
-Scikit-learn
-Joblib
-TF-IDF & Count Vectorization
-KMeans Clustering
-Latent Dirichlet Allocation (LDA)
