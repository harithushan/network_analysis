import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import os
from pathlib import Path
import numpy as np
import streamlit.components.v1 as components
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Disease-Symptom Network Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #3b82f6;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #e5e7eb;
        padding-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border-left: 5px solid #3b82f6;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
    }
    
    .success-box {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 5px solid #10b981;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #065f46;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #92400e;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8fafc;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_processed_data():
    """Load all processed data from output directory"""
    output_dir = Path("output")
    
    data = {}
    
    try:
        # Load CSV data
        data['nodes'] = pd.read_csv(output_dir / "data" / "node_analysis.csv")
        data['communities'] = pd.read_csv(output_dir / "data" / "community_analysis.csv")
        data['statistics'] = pd.read_csv(output_dir / "data" / "network_statistics.csv")
        data['summary'] = pd.read_csv(output_dir / "reports" / "summary_statistics.csv")
        
        # Load pickle data
        with open(output_dir / "data" / "network.pkl", 'rb') as f:
            pickle_data = pickle.load(f)
            data.update(pickle_data)
        
        # Load report
        try:
            with open(output_dir / "reports" / "comprehensive_analysis_report.md", 'r', encoding='utf-8') as f:
                data['report'] = f.read()
        except:
            data['report'] = "Report not available"
        
        return data
        
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}")
        st.info("Please run the data processor script first: `python robust_processor.py`")
        return None

def load_html_file(file_path):
    """Load and return HTML file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None

def display_image_with_caption(image_path, caption):
    """Display image with caption if it exists"""
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption=caption, use_column_width=True)
    else:
        st.warning(f"Image not found: {image_path}")

def create_metrics_overview(data):
    """Create overview metrics display"""
    stats = data['statistics'].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", f"{int(stats['Total_Nodes']):,}")
        st.metric("Density", f"{stats['Density']:.6f}")
    
    with col2:
        st.metric("Total Edges", f"{int(stats['Total_Edges']):,}")
        st.metric("Avg Clustering", f"{stats['Average_Clustering']:.4f}")
    
    with col3:
        st.metric("Disease Nodes", f"{int(stats['Disease_Nodes']):,}")
        st.metric("Louvain Communities", f"{int(stats['Louvain_Communities'])}")
    
    with col4:
        st.metric("Symptom Nodes", f"{int(stats['Symptom_Nodes']):,}")
        st.metric("Modularity", f"{stats['Louvain_Modularity']:.3f}")

def create_centrality_analysis_tab(data):
    """Create centrality analysis visualizations"""
    st.markdown('<h2 class="sub-header">📈 Centrality Analysis</h2>', unsafe_allow_html=True)
    
    # Explanation section
    with st.expander("📚 Understanding Centrality Measures", expanded=False):
        st.markdown("""
        **Centrality measures help identify the most important nodes in a network:**
        
        - **🔗 Degree Centrality**: Number of direct connections - identifies "popular" nodes
        - **🌉 Betweenness Centrality**: How often a node lies on shortest paths - identifies "bridge" nodes
        - **📍 Closeness Centrality**: How close a node is to all others - identifies "central" nodes
        
        Higher values indicate more important/influential nodes in the network structure.
        """)
    
    # Centrality selection
    col1, col2 = st.columns([1, 3])
    
    with col1:
        centrality_type = st.selectbox(
            "Select Centrality Measure:",
            ["Degree", "Betweenness", "Closeness"],
            key="cent_select"
        )
        
        node_type_filter = st.selectbox(
            "Focus on:",
            ["Both", "Diseases Only", "Symptoms Only"],
            key="node_filter"
        )
        
        top_n = st.slider("Number of top nodes to show:", 5, 25, 15, key="top_n")
    
    with col2:
        # Filter data based on selection
        nodes_df = data['nodes'].copy()
        
        if node_type_filter == "Diseases Only":
            nodes_df = nodes_df[nodes_df['Type'] == 'Disease']
        elif node_type_filter == "Symptoms Only":
            nodes_df = nodes_df[nodes_df['Type'] == 'Symptom']
        
        # Select centrality column
        cent_col = f"{centrality_type}_Centrality"
        
        # Create visualization
        top_nodes = nodes_df.nlargest(top_n, cent_col)
        
        fig = px.bar(
            top_nodes,
            x=cent_col,
            y='Node',
            orientation='h',
            color=cent_col,
            color_continuous_scale='viridis',
            title=f'Top {top_n} Nodes by {centrality_type} Centrality ({node_type_filter})',
            labels={cent_col: f'{centrality_type} Centrality Score', 'Node': 'Nodes'}
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'},
            title_font_size=18,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical analysis
    st.markdown("### 📊 Centrality Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Disease centrality distribution
        disease_nodes = data['nodes'][data['nodes']['Type'] == 'Disease']
        
        fig_dist = px.histogram(
            disease_nodes,
            x=f"{centrality_type}_Centrality",
            nbins=20,
            title=f"Disease {centrality_type} Centrality Distribution",
            color_discrete_sequence=['#e74c3c']
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Symptom centrality distribution
        symptom_nodes = data['nodes'][data['nodes']['Type'] == 'Symptom']
        
        fig_dist = px.histogram(
            symptom_nodes,
            x=f"{centrality_type}_Centrality",
            nbins=20,
            title=f"Symptom {centrality_type} Centrality Distribution",
            color_discrete_sequence=['#3498db']
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Comparative analysis
    st.markdown("### 🔍 Node Search and Analysis")
    
    selected_node = st.selectbox(
        "Search for a specific node:",
        [""] + sorted(data['nodes']['Node'].tolist()),
        key="node_search"
    )
    
    if selected_node:
        node_data = data['nodes'][data['nodes']['Node'] == selected_node].iloc[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="insight-box">
            <h4>🔍 Node Analysis: {selected_node}</h4>
            <strong>Type:</strong> {node_data['Type']}<br>
            <strong>Degree:</strong> {node_data['Degree']}<br>
            <strong>Degree Centrality:</strong> {node_data['Degree_Centrality']:.4f}<br>
            <strong>Betweenness Centrality:</strong> {node_data['Betweenness_Centrality']:.4f}<br>
            <strong>Closeness Centrality:</strong> {node_data['Closeness_Centrality']:.4f}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="insight-box">
            <h4>👥 Community Information</h4>
            <strong>Louvain Community:</strong> {node_data['Louvain_Community']}<br>
            <strong>Leiden Community:</strong> {node_data['Leiden_Community']}<br><br>
            <strong>Centrality Ranking:</strong><br>
            • Degree: #{(data['nodes']['Degree_Centrality'] > node_data['Degree_Centrality']).sum() + 1}<br>
            • Betweenness: #{(data['nodes']['Betweenness_Centrality'] > node_data['Betweenness_Centrality']).sum() + 1}<br>
            • Closeness: #{(data['nodes']['Closeness_Centrality'] > node_data['Closeness_Centrality']).sum() + 1}
            </div>
            """, unsafe_allow_html=True)

def create_network_visualization_tab(data):
    """Create network visualization tab"""
    st.markdown('<h2 class="sub-header">🌐 Interactive Network Visualization</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 🎛️ Visualization Controls")
        
        # Check for available network files
        main_network_path = Path("output") / "visualizations" / "disease_symptom_network.html"
        static_network_path = Path("output") / "visualizations" / "network_overview.png"
        
        if main_network_path.exists():
            if st.button("🌐 Load Interactive Network"):
                html_content = load_html_file(main_network_path)
                if html_content:
                    st.session_state['network_loaded'] = True
                    st.session_state['network_content'] = html_content
                    st.success("✅ Interactive network loaded!")
                else:
                    st.error("❌ Could not load network file")
        else:
            st.warning("⚠️ Interactive network not available")
            
        if static_network_path.exists():
            if st.button("📸 Show Static Network"):
                st.session_state['static_network'] = True
                st.success("✅ Static network loaded!")
        
        st.markdown("---")
        
        # Network statistics
        st.markdown("### 📊 Network Properties")
        stats = data['statistics'].iloc[0]
        
        st.metric("Density", f"{stats['Density']:.6f}")
        st.metric("Avg Degree", f"{stats['Average_Degree']:.2f}")
        st.metric("Clustering", f"{stats['Average_Clustering']:.4f}")
        
        # Node type distribution
        node_type_counts = data['nodes']['Type'].value_counts()
        fig_types = px.pie(
            values=node_type_counts.values,
            names=node_type_counts.index,
            title="Node Type Distribution",
            color_discrete_sequence=['#e74c3c', '#3498db']
        )
        fig_types.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 Network Explorer")
        
        # Check what's available and display accordingly
        if 'network_loaded' in st.session_state and st.session_state['network_loaded']:
            # Display the interactive network
            components.html(st.session_state['network_content'], height=600, scrolling=True)
        elif 'static_network' in st.session_state and st.session_state['static_network'] and static_network_path.exists():
            # Display static network
            st.image(str(static_network_path), caption="Disease-Symptom Network Overview", use_column_width=True)
        else:
            # Show available visualizations
            st.info("👆 Choose a visualization option from the controls")
            
            # Show any available preview images
            viz_dir = Path("output") / "visualizations"
            if viz_dir.exists():
                available_images = list(viz_dir.glob("*.png"))
                if available_images:
                    st.markdown("### 📸 Available Visualizations")
                    selected_viz = st.selectbox("Choose visualization:", [img.name for img in available_images])
                    if selected_viz:
                        selected_path = viz_dir / selected_viz
                        st.image(str(selected_path), caption=selected_viz.replace('.png', '').replace('_', ' ').title())
    
    # Network insights
    st.markdown("### 💡 Network Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_disease = data['nodes'][data['nodes']['Type'] == 'Disease'].nlargest(1, 'Degree_Centrality')
        if not top_disease.empty:
            st.markdown(f"""
            <div class="success-box">
            <strong>🏆 Most Connected Disease</strong><br>
            {top_disease.iloc[0]['Node']}<br>
            Degree: {top_disease.iloc[0]['Degree']}<br>
            Centrality: {top_disease.iloc[0]['Degree_Centrality']:.4f}
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        top_symptom = data['nodes'][data['nodes']['Type'] == 'Symptom'].nlargest(1, 'Degree_Centrality')
        if not top_symptom.empty:
            st.markdown(f"""
            <div class="success-box">
            <strong>🎯 Most Connected Symptom</strong><br>
            {top_symptom.iloc[0]['Node']}<br>
            Degree: {top_symptom.iloc[0]['Degree']}<br>
            Centrality: {top_symptom.iloc[0]['Degree_Centrality']:.4f}
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        bridge_node = data['nodes'].nlargest(1, 'Betweenness_Centrality')
        if not bridge_node.empty:
            st.markdown(f"""
            <div class="warning-box">
            <strong>🌉 Most Important Bridge</strong><br>
            {bridge_node.iloc[0]['Node']}<br>
            Type: {bridge_node.iloc[0]['Type']}<br>
            Betweenness: {bridge_node.iloc[0]['Betweenness_Centrality']:.4f}
            </div>
            """, unsafe_allow_html=True)

def create_community_analysis_tab(data):
    """Create community analysis visualizations"""
    st.markdown('<h2 class="sub-header">👥 Community Detection Analysis</h2>', unsafe_allow_html=True)
    
    # Algorithm comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Louvain Algorithm")
        louvain_data = data['communities'][data['communities']['Algorithm'] == 'Louvain']
        st.metric("Number of Communities", len(louvain_data))
        st.metric("Modularity Score", f"{data['statistics']['Louvain_Modularity'].iloc[0]:.3f}")
        
        # Community size distribution
        fig_louvain = px.histogram(
            louvain_data,
            x='Size',
            nbins=15,
            title="Louvain Community Size Distribution",
            color_discrete_sequence=['#e74c3c']
        )
        fig_louvain.update_layout(height=300)
        st.plotly_chart(fig_louvain, use_container_width=True)
    
    with col2:
        st.markdown("### 🧬 Leiden Algorithm")
        leiden_data = data['communities'][data['communities']['Algorithm'] == 'Leiden']
        st.metric("Number of Communities", len(leiden_data))
        st.metric("Average Community Size", f"{leiden_data['Size'].mean():.1f}")
        
        # Community size distribution
        fig_leiden = px.histogram(
            leiden_data,
            x='Size',
            nbins=15,
            title="Leiden Community Size Distribution",
            color_discrete_sequence=['#3498db']
        )
        fig_leiden.update_layout(height=300)
        st.plotly_chart(fig_leiden, use_container_width=True)
    
    # Algorithm selection for detailed analysis
    st.markdown("### 🔍 Detailed Community Analysis")
    
    algorithm = st.selectbox("Choose Algorithm for Analysis:", ["Louvain", "Leiden"])
    community_data = data['communities'][data['communities']['Algorithm'] == algorithm]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_community = st.selectbox(
            f"Select {algorithm} Community:",
            sorted(community_data['Community_ID'].tolist())
        )
        
        if selected_community is not None:
            comm_info = community_data[community_data['Community_ID'] == selected_community].iloc[0]
            
            st.markdown(f"""
            <div class="insight-box">
            <h4>Community {selected_community} Details</h4>
            <strong>Total Size:</strong> {comm_info['Size']} nodes<br>
            <strong>Diseases:</strong> {comm_info['Diseases']} nodes<br>
            <strong>Symptoms:</strong> {comm_info['Symptoms']} nodes<br>
            <strong>Composition:</strong> {comm_info['Diseases']/(comm_info['Diseases']+comm_info['Symptoms'])*100:.1f}% diseases
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Community composition visualization
        if selected_community is not None:
            comm_info = community_data[community_data['Community_ID'] == selected_community].iloc[0]
            
            # Pie chart for community composition
            fig_pie = px.pie(
                values=[comm_info['Diseases'], comm_info['Symptoms']],
                names=['Diseases', 'Symptoms'],
                title=f"{algorithm} Community {selected_community} Composition",
                color_discrete_sequence=['#e74c3c', '#3498db']
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Community comparison
    st.markdown("### 📊 Algorithm Comparison")
    
    # Create comparison metrics
    comparison_data = {
        'Algorithm': ['Louvain', 'Leiden'],
        'Communities': [len(louvain_data), len(leiden_data)],
        'Avg Size': [louvain_data['Size'].mean(), leiden_data['Size'].mean()],
        'Max Size': [louvain_data['Size'].max(), leiden_data['Size'].max()],
        'Min Size': [louvain_data['Size'].min(), leiden_data['Size'].min()],
        'Std Size': [louvain_data['Size'].std(), leiden_data['Size'].std()]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Interactive network visualization section
    st.markdown("### 🌐 Interactive Community Networks")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        viz_algorithm = st.selectbox("Visualization Algorithm:", ["Louvain", "Leiden"], key="viz_algo")
        viz_centrality = st.selectbox("Node Size by:", ["Degree", "Betweenness", "Closeness"], key="viz_cent")
        
        html_filename = f"{viz_algorithm.lower()}_{viz_centrality.lower()}_communities.html"
        html_path = Path("output") / "networks" / html_filename
        
        if st.button("Load Community Visualization"):
            html_content = load_html_file(html_path)
            if html_content:
                st.session_state['community_network'] = html_content
                st.success(f"✅ Loaded {viz_algorithm} communities with {viz_centrality} centrality sizing")
            else:
                st.error(f"❌ Could not load {html_filename}")
                
                # Check for available community images
                img_path = Path("output") / "visualizations" / f"{viz_algorithm.lower()}_communities.png"
                if img_path.exists():
                    st.info("📸 Showing static community visualization instead:")
                    st.session_state['community_static'] = str(img_path)
    
    with col2:
        if 'community_network' in st.session_state:
            components.html(st.session_state['community_network'], height=600, scrolling=True)
        elif 'community_static' in st.session_state:
            st.image(st.session_state['community_static'], caption="Community Structure", use_column_width=True)
        else:
            st.info("📖 **Interactive Network Guide:**\n\n"
                    "• **Hover** over nodes to see details\n"
                    "• **Drag** nodes to rearrange\n"
                    "• **Zoom** with mouse wheel\n"
                    "• **Colors** represent communities\n"
                    "• **Sizes** represent centrality scores")
def create_advanced_analytics_tab(data):
    """Create advanced analytics and export functionality"""
    st.markdown('<h2 class="sub-header">🔬 Advanced Analytics & Export</h2>', unsafe_allow_html=True)
    
    # Degree distribution analysis
    st.markdown("### 📈 Degree Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Overall degree distribution
        fig_degree = px.histogram(
            data['nodes'],
            x='Degree',
            nbins=30,
            title="Overall Degree Distribution",
            color='Type',
            color_discrete_map={'Disease': '#e74c3c', 'Symptom': '#3498db'}
        )
        fig_degree.update_layout(height=400)
        st.plotly_chart(fig_degree, use_container_width=True)
    
    with col2:
        # Log-scale degree distribution
        degrees = data['nodes']['Degree'].values
        log_degrees = np.log10(degrees + 1)  # Add 1 to avoid log(0)
        
        fig_log = px.histogram(
            x=log_degrees,
            nbins=20,
            title="Log-Scale Degree Distribution",
            labels={'x': 'log₁₀(Degree + 1)', 'y': 'Count'}
        )
        fig_log.update_layout(height=400)
        st.plotly_chart(fig_log, use_container_width=True)
    
    # Correlation analysis
    st.markdown("### 🔗 Centrality Correlation Analysis")
    
    # Calculate correlations
    centrality_cols = ['Degree_Centrality', 'Betweenness_Centrality', 'Closeness_Centrality']
    corr_matrix = data['nodes'][centrality_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Centrality Measures Correlation Matrix",
        color_continuous_scale='RdBu'
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Disease Nodes Statistics:**")
        disease_stats = data['nodes'][data['nodes']['Type'] == 'Disease'][centrality_cols].describe()
        st.dataframe(disease_stats, use_container_width=True)
    
    with col2:
        st.markdown("**Symptom Nodes Statistics:**")
        symptom_stats = data['nodes'][data['nodes']['Type'] == 'Symptom'][centrality_cols].describe()
        st.dataframe(symptom_stats, use_container_width=True)
    
    # Export functionality
    st.markdown("### 💾 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.download_button(
            label="📊 Download Node Analysis",
            data=data['nodes'].to_csv(index=False),
            file_name="disease_symptom_node_analysis.csv",
            mime="text/csv"
        ):
            st.success("✅ Node analysis downloaded!")
    
    with col2:
        if st.download_button(
            label="👥 Download Community Data",
            data=data['communities'].to_csv(index=False),
            file_name="community_analysis.csv",
            mime="text/csv"
        ):
            st.success("✅ Community data downloaded!")
    
    with col3:
        if st.download_button(
            label="📈 Download Network Stats",
            data=data['statistics'].to_csv(index=False),
            file_name="network_statistics.csv",
            mime="text/csv"
        ):
            st.success("✅ Statistics downloaded!")
    
    # Custom analysis
    st.markdown("### 🔍 Custom Node Analysis")
    
    analysis_type = st.selectbox(
        "Choose Analysis Type:",
        ["Top Nodes by Multiple Criteria", "Community Comparison", "Node Similarity Analysis"]
    )
    
    if analysis_type == "Top Nodes by Multiple Criteria":
        col1, col2 = st.columns([1, 2])
        
        with col1:
            criteria_weights = {}
            st.markdown("**Set Importance Weights:**")
            criteria_weights['degree'] = st.slider("Degree Centrality Weight", 0.0, 1.0, 0.4)
            criteria_weights['betweenness'] = st.slider("Betweenness Weight", 0.0, 1.0, 0.3)
            criteria_weights['closeness'] = st.slider("Closeness Weight", 0.0, 1.0, 0.3)
            
            top_k = st.slider("Number of top nodes", 5, 30, 15)
        
        with col2:
            # Calculate composite score
            composite_score = (
                data['nodes']['Degree_Centrality'] * criteria_weights['degree'] +
                data['nodes']['Betweenness_Centrality'] * criteria_weights['betweenness'] +
                data['nodes']['Closeness_Centrality'] * criteria_weights['closeness']
            )
            
            data['nodes']['Composite_Score'] = composite_score
            top_composite = data['nodes'].nlargest(top_k, 'Composite_Score')
            
            fig_composite = px.bar(
                top_composite,
                x='Composite_Score',
                y='Node',
                color='Type',
                title=f"Top {top_k} Nodes by Composite Score",
                color_discrete_map={'Disease': '#e74c3c', 'Symptom': '#3498db'},
                orientation='h'
            )
            fig_composite.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_composite, use_container_width=True)
    
    elif analysis_type == "Community Comparison":
        st.markdown("**Compare Louvain vs Leiden Communities:**")
        
        # Community size comparison
        louvain_sizes = data['communities'][data['communities']['Algorithm'] == 'Louvain']['Size']
        leiden_sizes = data['communities'][data['communities']['Algorithm'] == 'Leiden']['Size']
        
        comparison_stats = pd.DataFrame({
            'Metric': ['Mean Size', 'Median Size', 'Std Dev', 'Min Size', 'Max Size'],
            'Louvain': [
                louvain_sizes.mean(),
                louvain_sizes.median(),
                louvain_sizes.std(),
                louvain_sizes.min(),
                louvain_sizes.max()
            ],
            'Leiden': [
                leiden_sizes.mean(),
                leiden_sizes.median(),
                leiden_sizes.std(),
                leiden_sizes.min(),
                leiden_sizes.max()
            ]
        })
        
        st.dataframe(comparison_stats, use_container_width=True, hide_index=True)
        
        # Box plot comparison
        all_sizes = pd.concat([
            pd.DataFrame({'Algorithm': 'Louvain', 'Size': louvain_sizes}),
            pd.DataFrame({'Algorithm': 'Leiden', 'Size': leiden_sizes})
        ])
        
        fig_box = px.box(
            all_sizes,
            x='Algorithm',
            y='Size',
            title="Community Size Distribution Comparison",
            color='Algorithm',
            color_discrete_map={'Louvain': '#e74c3c', 'Leiden': '#3498db'}
        )
        st.plotly_chart(fig_box, use_container_width=True)

def create_overview_tab(data):
    """Create overview tab with key insights"""
    st.markdown('<h2 class="sub-header">📋 Project Overview</h2>', unsafe_allow_html=True)
    
    # Key metrics overview
    create_metrics_overview(data)
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🏥 Disease-Symptom Network Analysis
        
        This comprehensive analysis explores the complex relationships between diseases and symptoms 
        using network science approaches. Our bipartite network reveals hidden patterns and structures 
        in medical data, providing insights for:
        
        - **🔍 Disease Diagnosis**: Identifying symptom patterns for different diseases
        - **🎯 Symptom Prediction**: Understanding which symptoms commonly co-occur
        - **👥 Medical Clustering**: Finding groups of related diseases and symptoms
        - **🌉 Critical Pathways**: Discovering important bridging symptoms and diseases
        
        ### 📊 Dataset Characteristics
        - **773 unique diseases** representing diverse medical conditions
        - **377 symptoms** covering comprehensive symptom profiles
        - **~246,000 samples** ensuring robust statistical analysis
        - **Bipartite structure** clearly separating diseases from symptoms
        """)
        
        # Key findings
        st.markdown("### 🔑 Key Discoveries")
        
        top_disease = data['nodes'][data['nodes']['Type'] == 'Disease'].nlargest(1, 'Degree_Centrality').iloc[0]
        top_symptom = data['nodes'][data['nodes']['Type'] == 'Symptom'].nlargest(1, 'Degree_Centrality').iloc[0]
        
        st.markdown(f"""
        <div class="insight-box">
        <h4>🏆 Most Influential Nodes</h4>
        <strong>Top Disease:</strong> {top_disease['Node']} (Degree: {top_disease['Degree']})<br>
        <strong>Top Symptom:</strong> {top_symptom['Node']} (Degree: {top_symptom['Degree']})<br><br>
        
        <h4>🌐 Network Structure</h4>
        <strong>Sparsity:</strong> {data['statistics']['Density'].iloc[0]:.6f} - indicating specific disease-symptom relationships<br>
        <strong>Communities:</strong> {data['statistics']['Louvain_Communities'].iloc[0]} distinct clusters detected<br>
        <strong>Modularity:</strong> {data['statistics']['Louvain_Modularity'].iloc[0]:.3f} - showing good community separation
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Network visualization preview
        st.markdown("### 📸 Network Preview")
        preview_path = Path("output") / "visualizations" / "louvain_communities.png"
        if preview_path.exists():
            display_image_with_caption(preview_path, "Community Structure Visualization")
        
        # Quick stats
        st.markdown("### 📈 Quick Statistics")
        stats_summary = data['summary']
        
        for _, row in stats_summary.iterrows():
            if row['metric'] in ['Total Nodes', 'Total Edges', 'Network Density', 'Louvain Communities']:
                st.metric(row['metric'], row['value'])
    
    # Methodology
    st.markdown("### 🔬 Analysis Methodology")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🌐 Network Construction**
        - Bipartite graph creation
        - Disease-symptom edge mapping
        - Node type classification
        - Graph validation
        """)
    
    with col2:
        st.markdown("""
        **📊 Centrality Analysis**
        - Degree centrality calculation
        - Betweenness centrality analysis
        - Closeness centrality measurement
        - Comparative ranking
        """)
    
    with col3:
        st.markdown("""
        **👥 Community Detection**
        - Louvain algorithm application
        - Leiden algorithm comparison
        - Modularity optimization
        - Community characterization
        """)
    
    # Navigation guide
    st.markdown("### 🧭 Dashboard Navigation Guide")
    
    st.markdown("""
    <div class="success-box">
    <strong>🚀 How to Use This Dashboard:</strong><br><br>
    
    <strong>📋 Overview:</strong> Start here for project summary and key findings<br>
    <strong>🌐 Network Visualization:</strong> Explore interactive network visualizations<br>
    <strong>📈 Centrality Analysis:</strong> Discover most important diseases and symptoms<br>
    <strong>👥 Community Detection:</strong> Examine disease-symptom clusters<br>
    <strong>🔬 Advanced Analytics:</strong> Perform custom analysis and export data
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<h1 class="main-header">🏥 Disease-Symptom Network Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_processed_data()
    
    if data is None:
        st.error("❌ Could not load processed data. Please run the data processor first.")
        st.code("python process_network_data.py", language="bash")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Dashboard Controls")
        
        # Data info
        st.markdown("### 📊 Dataset Info")
        st.info(f"**Nodes:** {data['statistics']['Total_Nodes'].iloc[0]:,.0f}\n\n"
                f"**Edges:** {data['statistics']['Total_Edges'].iloc[0]:,.0f}\n\n"
                f"**Diseases:** {data['statistics']['Disease_Nodes'].iloc[0]:,.0f}\n\n"
                f"**Symptoms:** {data['statistics']['Symptom_Nodes'].iloc[0]:,.0f}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 View All Files"):
            output_files = []
            for folder in ['data', 'visualizations', 'networks', 'reports']:
                folder_path = Path("output") / folder
                if folder_path.exists():
                    files = list(folder_path.glob("*"))
                    output_files.extend([f"{folder}/{f.name}" for f in files])
            
            if output_files:
                st.success(f"Found {len(output_files)} output files")
                with st.expander("📁 File List"):
                    for file in sorted(output_files):
                        st.text(file)
        
        # Analysis summary
        st.markdown("### 🎯 Analysis Summary")
        modularity = data['statistics']['Louvain_Modularity'].iloc[0]
        
        if modularity > 0.5:
            community_quality = "🟢 Excellent"
        elif modularity > 0.3:
            community_quality = "🟡 Good"
        else:
            community_quality = "🔴 Fair"
        
        st.markdown(f"**Community Structure:** {community_quality}")
        st.markdown(f"**Modularity:** {modularity:.3f}")
        
        density = data['statistics']['Density'].iloc[0]
        if density < 0.001:
            network_type = "Very Sparse"
        elif density < 0.01:
            network_type = "Sparse"
        else:
            network_type = "Dense"
        
        st.markdown(f"**Network Type:** {network_type}")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Overview", 
        "🌐 Network Visualization", 
        "📈 Centrality Analysis", 
        "👥 Community Detection", 
        "🔬 Advanced Analytics"
    ])
    
    with tab1:
        create_overview_tab(data)
    
    with tab2:
        create_network_visualization_tab(data)
    
    with tab3:
        create_centrality_analysis_tab(data)
    
    with tab4:
        create_community_analysis_tab(data)
    
    with tab5:
        create_advanced_analytics_tab(data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; margin-top: 2rem;'>
    <p>🔬 Network Analysis Dashboard | Built with Streamlit & NetworkX | 
    📊 Disease-Symptom Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()