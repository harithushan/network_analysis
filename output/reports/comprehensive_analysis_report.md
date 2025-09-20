# Disease-Symptom Network Analysis Report

## Executive Summary

This analysis examines a bipartite network of 1,097 nodes (773 diseases and 324 symptoms) connected by 5,388 edges, representing symptom-disease associations from a medical dataset.

## Key Findings

### Network Structure
- **Density**: 0.008963 (sparse network indicating specific disease-symptom relationships)
- **Average Degree**: 9.82
- **Clustering Coefficient**: 0.0089

### Most Important Nodes

#### Top Diseases (by Degree Centrality)
1. **depression**: 0.0529
2. **drug abuse**: 0.0310
3. **lymphedema**: 0.0182
4. **diaper rash**: 0.0128
5. **acute pancreatitis**: 0.0109
6. **infectious gastroenteritis**: 0.0109
7. **marijuana abuse**: 0.0109
8. **bursitis**: 0.0109
9. **spondylosis**: 0.0109
10. **injury to the arm**: 0.0109

#### Top Symptoms (by Degree Centrality)  
1. **sharp abdominal pain**: 0.1232
2. **headache**: 0.1058
3. **sharp chest pain**: 0.0949
4. **shortness of breath**: 0.0922
5. **cough**: 0.0912
6. **vomiting**: 0.0894
7. **dizziness**: 0.0858
8. **nausea**: 0.0730
9. **depressive or psychotic symptoms**: 0.0693
10. **fever**: 0.0693

### Community Structure
- **Louvain Algorithm**: 10 communities (Modularity: 0.595)
- **Leiden Algorithm**: 10 communities

The strong modularity score indicates well-defined community structure in the network.

## Network Properties

### Degree Distribution
- **Range**: 1 - 135
- **Mean**: 9.82
- **Standard Deviation**: 12.19

### Centrality Analysis
The network exhibits typical scale-free properties with a few highly connected hubs and many nodes with few connections. This suggests:

1. **Key diseases** like "depression" act as hubs with diverse symptom presentations
2. **Common symptoms** like "sharp abdominal pain" appear across multiple diseases
3. **Specialized relationships** exist for most disease-symptom pairs

## Medical Implications

### Clinical Insights
- High-degree diseases may represent complex conditions with diverse presentations
- High-degree symptoms might be common across multiple medical conditions
- Community structure could reflect medical specialties or related disease families

### Applications
- **Diagnostic Support**: Use network structure to suggest related symptoms or diseases
- **Medical Education**: Understand symptom-disease relationships
- **Research Priorities**: Focus on highly connected nodes for maximum impact

## Technical Details

### Data Processing
- **Source**: Disease-symptom association dataset
- **Network Type**: Undirected bipartite graph
- **Community Detection**: Louvain and Leiden algorithms
- **Centrality Measures**: Degree, betweenness, and closeness centrality

### Quality Metrics
- **Data Completeness**: High (all nodes connected)
- **Algorithm Convergence**: Successful for all measures
- **Community Quality**: Modularity = 0.595

---
**Report Generated**: 2025-09-20 11:15:40  
**Analysis Version**: 2.0 (Robust Processing)
