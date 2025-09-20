# Disease-Symptom Network Analysis Output

## Generated Files

### Data Files (`data/`)
- `node_analysis.csv` - Node-level metrics and community assignments
- `community_analysis.csv` - Community statistics and composition  
- `network_statistics.csv` - Overall network metrics
- `network.pkl` - Complete analysis results (Python pickle)

### Visualizations (`visualizations/`)
- `network_overview.png` - Comprehensive network visualization
- `degree_centrality_analysis.png` - Degree centrality analysis
- `betweenness_centrality_analysis.png` - Betweenness centrality analysis
- `closeness_centrality_analysis.png` - Closeness centrality analysis
- `louvain_communities.png` - Louvain community structure
- `leiden_communities.png` - Leiden community structure
- `disease_symptom_network.html` - Interactive main network

### Interactive Networks (`networks/`)
- Community network visualizations with different centrality measures
- HTML files with D3.js interactive visualizations

### Reports (`reports/`)
- `comprehensive_analysis_report.md` - Detailed analysis report
- `summary_statistics.csv` - Key metrics summary

## Usage

1. **Dashboard**: Load with `streamlit run dashboard.py`
2. **Further Analysis**: Load `network.pkl` in Python
3. **Presentations**: Use PNG files for static displays
4. **Interactive Exploration**: Open HTML files in browser

Generated: 2025-09-20 11:15:40
