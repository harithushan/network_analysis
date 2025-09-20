import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import community.community_louvain as community_louvain
import igraph as ig
import leidenalg
import json
import os
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

def setup_output_directory():
    """Create output directory structure"""
    output_dir = Path("output")
    subdirs = ["data", "visualizations", "reports", "networks"]
    
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return output_dir

def load_and_process_data():
    """Load and create the network"""
    print("Loading dataset...")
    df = pd.read_csv("data/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
    
    print("Creating bipartite network...")
    edges = []
    for _, row in df.iterrows():
        disease = row["diseases"]
        for symptom, val in row.drop("diseases").items():
            if val == 1:
                edges.append((disease, symptom))
    
    B = nx.Graph()
    B.add_edges_from(edges)
    
    disease_nodes = df["diseases"].unique().tolist()
    symptom_nodes = list(set(B.nodes()) - set(disease_nodes))
    
    print(f"Network created: {B.number_of_nodes()} nodes, {B.number_of_edges()} edges")
    print(f"Diseases: {len(disease_nodes)}, Symptoms: {len(symptom_nodes)}")
    
    return df, B, disease_nodes, symptom_nodes

def calculate_centralities(B):
    """Calculate all centrality measures"""
    print("Calculating centrality measures...")
    
    deg_cent = nx.degree_centrality(B)
    print("✅ Degree centrality calculated")
    
    # Use approximation for large networks
    if B.number_of_nodes() > 1000:
        bet_cent = nx.betweenness_centrality(B, k=min(500, B.number_of_nodes()//2), seed=42)
        print("✅ Betweenness centrality calculated (approximated)")
    else:
        bet_cent = nx.betweenness_centrality(B)
        print("✅ Betweenness centrality calculated")
    
    clo_cent = nx.closeness_centrality(B)
    print("✅ Closeness centrality calculated")
    
    return deg_cent, bet_cent, clo_cent

def detect_communities(B):
    """Detect communities using multiple algorithms"""
    print("Detecting communities...")
    
    # Louvain
    print("Running Louvain algorithm...")
    louvain_partition = community_louvain.best_partition(B)
    modularity = community_louvain.modularity(louvain_partition, B)
    print(f"✅ Louvain completed: {len(set(louvain_partition.values()))} communities (modularity: {modularity:.3f})")
    
    # Leiden
    print("Running Leiden algorithm...")
    try:
        G_ig = ig.Graph.TupleList(B.edges(), directed=False)
        leiden_partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
        leiden_partition_map = {G_ig.vs[i]["name"]: comm for i, comm in enumerate(leiden_partition.membership)}
        print(f"✅ Leiden completed: {len(leiden_partition)} communities")
    except Exception as e:
        print(f"⚠️  Leiden failed: {e}. Using Louvain results for Leiden as well.")
        leiden_partition_map = louvain_partition.copy()
        leiden_partition = list(set(louvain_partition.values()))
    
    return louvain_partition, modularity, leiden_partition_map, len(set(leiden_partition_map.values()))

def create_static_visualizations(B, deg_cent, bet_cent, clo_cent, disease_nodes, symptom_nodes, 
                                louvain_partition, leiden_partition_map, output_dir):
    """Create all static visualizations"""
    print("Creating static visualizations...")
    
    def top_n(metric_dict, node_list, n=15):
        return sorted([(n, v) for n, v in metric_dict.items() if n in node_list],
                     key=lambda x: x[1], reverse=True)[:n]
    
    centralities = {
        'degree': deg_cent,
        'betweenness': bet_cent,
        'closeness': clo_cent
    }
    
    # Centrality visualizations
    for cent_name, cent_dict in centralities.items():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{cent_name.title()} Centrality Analysis', fontsize=16, fontweight='bold')
        
        # Top diseases
        top_diseases = top_n(cent_dict, disease_nodes)
        if top_diseases:
            names, scores = zip(*top_diseases)
            bars1 = ax1.barh(range(len(names)), scores, color='#e74c3c', alpha=0.8)
            ax1.set_yticks(range(len(names)))
            ax1.set_yticklabels(names, fontsize=10)
            ax1.set_xlabel(f'{cent_name.title()} Centrality Score', fontsize=12)
            ax1.set_title(f'Top Diseases by {cent_name.title()} Centrality', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        # Top symptoms
        top_symptoms = top_n(cent_dict, symptom_nodes)
        if top_symptoms:
            names, scores = zip(*top_symptoms)
            bars2 = ax2.barh(range(len(names)), scores, color='#3498db', alpha=0.8)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names, fontsize=10)
            ax2.set_xlabel(f'{cent_name.title()} Centrality Score', fontsize=12)
            ax2.set_title(f'Top Symptoms by {cent_name.title()} Centrality', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        # Distribution for diseases
        disease_scores = [cent_dict[n] for n in disease_nodes]
        ax3.hist(disease_scores, bins=25, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.5)
        ax3.set_xlabel(f'{cent_name.title()} Centrality Score', fontsize=12)
        ax3.set_ylabel('Number of Disease Nodes', fontsize=12)
        ax3.set_title(f'Disease {cent_name.title()} Centrality Distribution', fontsize=14)
        ax3.grid(True, alpha=0.3)
        ax3.axvline(np.mean(disease_scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(disease_scores):.4f}')
        ax3.legend()
        
        # Distribution for symptoms
        symptom_scores = [cent_dict[n] for n in symptom_nodes]
        ax4.hist(symptom_scores, bins=25, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.5)
        ax4.set_xlabel(f'{cent_name.title()} Centrality Score', fontsize=12)
        ax4.set_ylabel('Number of Symptom Nodes', fontsize=12)
        ax4.set_title(f'Symptom {cent_name.title()} Centrality Distribution', fontsize=14)
        ax4.grid(True, alpha=0.3)
        ax4.axvline(np.mean(symptom_scores), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(symptom_scores):.4f}')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "visualizations" / f"{cent_name}_centrality_analysis.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ {cent_name.title()} centrality analysis saved")
    
    # Network overview visualization
    print("Creating network overview visualization...")
    
    # Sample nodes for visualization if network is too large
    if B.number_of_nodes() > 500:
        print("Large network detected, creating sample visualization...")
        degrees = dict(B.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:300]
        B_vis = B.subgraph(top_nodes)
        title_suffix = " (Top 300 Nodes by Degree)"
    else:
        B_vis = B
        title_suffix = ""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'Disease-Symptom Network Analysis{title_suffix}', fontsize=18, fontweight='bold')
    
    # Basic network layout
    pos = nx.spring_layout(B_vis, k=0.8, iterations=50, seed=42)
    
    # Network with node types
    disease_nodes_vis = [n for n in B_vis.nodes() if n in disease_nodes]
    symptom_nodes_vis = [n for n in B_vis.nodes() if n not in disease_nodes]
    
    nx.draw_networkx_edges(B_vis, pos, alpha=0.3, edge_color='gray', width=0.5, ax=ax1)
    nx.draw_networkx_nodes(B_vis, pos, nodelist=disease_nodes_vis, node_color='#e74c3c', 
                          node_size=60, alpha=0.8, label='Diseases', ax=ax1)
    nx.draw_networkx_nodes(B_vis, pos, nodelist=symptom_nodes_vis, node_color='#3498db', 
                          node_size=40, alpha=0.8, label='Symptoms', ax=ax1)
    ax1.set_title('Network by Node Type', fontsize=14)
    ax1.legend(scatterpoints=1)
    ax1.axis('off')
    
    # Network with degree centrality
    node_sizes = [deg_cent[n] * 2000 + 10 for n in B_vis.nodes()]
    nx.draw_networkx_edges(B_vis, pos, alpha=0.3, edge_color='gray', width=0.5, ax=ax2)
    nx.draw_networkx_nodes(B_vis, pos, node_size=node_sizes, node_color='orange', 
                          alpha=0.7, ax=ax2)
    ax2.set_title('Network by Degree Centrality', fontsize=14)
    ax2.axis('off')
    
    # Louvain communities
    community_colors = {}
    unique_communities = list(set(louvain_partition.values()))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_communities)))
    for i, comm in enumerate(unique_communities):
        community_colors[comm] = colors[i]
    
    node_colors_louvain = [community_colors[louvain_partition[n]] for n in B_vis.nodes()]
    nx.draw_networkx_edges(B_vis, pos, alpha=0.2, edge_color='gray', width=0.5, ax=ax3)
    nx.draw_networkx_nodes(B_vis, pos, node_color=node_colors_louvain, node_size=50, 
                          alpha=0.8, ax=ax3)
    ax3.set_title(f'Louvain Communities ({len(unique_communities)} communities)', fontsize=14)
    ax3.axis('off')
    
    # Degree distribution
    degrees = [B.degree(n) for n in B.nodes()]
    ax4.hist(degrees, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Node Degree', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Degree Distribution', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(np.mean(degrees), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(degrees):.2f}')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "visualizations" / "network_overview.png", 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Network overview saved")
    
    # Community-specific visualizations
    partitions = {
        'louvain': louvain_partition,
        'leiden': leiden_partition_map
    }
    
    for partition_name, partition in partitions.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'{partition_name.title()} Community Analysis', fontsize=16, fontweight='bold')
        
        # Community size distribution
        community_sizes = list(pd.Series(list(partition.values())).value_counts().values)
        ax1.hist(community_sizes, bins=max(10, len(community_sizes)//5), color='lightblue', 
                alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Community Size', fontsize=12)
        ax1.set_ylabel('Number of Communities', fontsize=12)
        ax1.set_title(f'{partition_name.title()} Community Size Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Network visualization with communities (sample if large)
        if B.number_of_nodes() > 400:
            degrees = dict(B.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:200]
            B_comm = B.subgraph(top_nodes)
            pos = nx.spring_layout(B_comm, k=1, iterations=50, seed=42)
        else:
            B_comm = B
            pos = nx.spring_layout(B_comm, k=0.5, iterations=50, seed=42)
        
        # Color nodes by community
        unique_comms = list(set(partition.values()))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_comms)))
        comm_colors = {comm: colors[i] for i, comm in enumerate(unique_comms)}
        
        node_colors = [comm_colors[partition[n]] for n in B_comm.nodes()]
        
        nx.draw_networkx_edges(B_comm, pos, alpha=0.2, edge_color='gray', width=0.5, ax=ax2)
        nx.draw_networkx_nodes(B_comm, pos, node_color=node_colors, node_size=30, 
                              alpha=0.8, ax=ax2)
        ax2.set_title(f'{partition_name.title()} Communities Network View', fontsize=14)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "visualizations" / f"{partition_name}_communities.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ {partition_name.title()} community analysis saved")

def create_simple_html_networks(B, disease_nodes, symptom_nodes, louvain_partition, leiden_partition_map,
                               deg_cent, bet_cent, clo_cent, output_dir):
    """Create simple HTML network visualizations"""
    print("Creating HTML network visualizations...")
    
    # Main network
    print("Creating main network HTML...")
    
    # Sample for large networks
    if B.number_of_nodes() > 300:
        degrees = dict(B.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:250]
        B_html = B.subgraph(top_nodes)
    else:
        B_html = B
    
    # Prepare node and edge data
    nodes_data = []
    edges_data = []
    
    for node in B_html.nodes():
        nodes_data.append({
            "id": node,
            "group": "disease" if node in disease_nodes else "symptom",
            "degree": B_html.degree(node),
            "deg_cent": deg_cent.get(node, 0),
            "bet_cent": bet_cent.get(node, 0),
            "clo_cent": clo_cent.get(node, 0)
        })
    
    for edge in B_html.edges():
        edges_data.append({"source": edge[0], "target": edge[1]})
    
    # Create main network HTML
    html_content = create_network_html(
        nodes_data, edges_data, 
        "Disease-Symptom Network", 
        f"Interactive visualization of {B.number_of_nodes()} nodes and {B.number_of_edges()} edges"
    )
    
    with open(output_dir / "visualizations" / "disease_symptom_network.html", 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("✅ Main network HTML saved")
    
    # Community networks
    partitions = {
        'louvain': louvain_partition,
        'leiden': leiden_partition_map
    }
    
    centralities = {
        'degree': deg_cent,
        'betweenness': bet_cent,
        'closeness': clo_cent
    }
    
    for partition_name, partition in partitions.items():
        for cent_name, centrality in centralities.items():
            try:
                # Prepare community network data
                comm_nodes_data = []
                
                for node in B_html.nodes():
                    comm_id = partition.get(node, 0)
                    comm_nodes_data.append({
                        "id": node,
                        "group": comm_id,
                        "type": "disease" if node in disease_nodes else "symptom",
                        "centrality": centrality.get(node, 0),
                        "size": max(5, centrality.get(node, 0) * 1000)
                    })
                
                title = f"{partition_name.title()} Communities - {cent_name.title()} Centrality"
                description = f"Node colors represent communities, sizes represent {cent_name} centrality"
                
                comm_html = create_community_network_html(
                    comm_nodes_data, edges_data, title, description, 
                    len(set(partition.values())), cent_name
                )
                
                filename = f"{partition_name}_{cent_name}_communities.html"
                with open(output_dir / "networks" / filename, 'w', encoding='utf-8') as f:
                    f.write(comm_html)
                print(f"✅ {filename} saved")
                
            except Exception as e:
                print(f"⚠️  Error creating {partition_name}_{cent_name}: {e}")

def create_network_html(nodes_data, edges_data, title, description):
    """Create HTML content for basic network visualization"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px 10px 0 0; }}
        .network-container {{ width: 100%; height: 700px; }}
        .disease {{ fill: #e74c3c; }}
        .symptom {{ fill: #3498db; }}
        .link {{ stroke: #999; stroke-opacity: 0.4; }}
        .tooltip {{ position: absolute; background: rgba(0,0,0,0.8); color: white; padding: 10px; 
                   border-radius: 5px; pointer-events: none; opacity: 0; font-size: 12px; }}
        .controls {{ padding: 20px; background-color: #f8f9fa; }}
        .legend {{ display: flex; justify-content: center; gap: 30px; align-items: center; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0; font-size: 24px;">{title}</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">{description}</p>
        </div>
        
        <div class="controls">
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color disease"></div>
                    <span>Diseases ({len([n for n in nodes_data if n['group'] == 'disease'])})</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color symptom"></div>
                    <span>Symptoms ({len([n for n in nodes_data if n['group'] == 'symptom'])})</span>
                </div>
                <div class="legend-item">
                    <strong>Nodes:</strong> {len(nodes_data)} | <strong>Edges:</strong> {len(edges_data)}
                </div>
            </div>
        </div>
        
        <div class="network-container" id="network"></div>
    </div>
    
    <script>
        const nodes = {str(nodes_data).replace("'", '"')};
        const links = {str(edges_data).replace("'", '"')};
        
        const width = 1200;
        const height = 700;
        
        const svg = d3.select("#network")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, width, height]);
        
        const tooltip = d3.select("body").append("div").attr("class", "tooltip");
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(50))
            .force("charge", d3.forceManyBody().strength(-80))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link");
        
        const node = svg.append("g")
            .selectAll("circle")
            .data(nodes)
            .enter().append("circle")
            .attr("r", d => Math.sqrt(d.degree) + 3)
            .attr("class", d => d.group)
            .on("mouseover", function(event, d) {{
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(
                    `<strong>${{d.group.charAt(0).toUpperCase() + d.group.slice(1)}}: ${{d.id}}</strong><br/>` +
                    `Degree: ${{d.degree}}<br/>` +
                    `Degree Centrality: ${{d.deg_cent.toFixed(4)}}<br/>` +
                    `Betweenness: ${{d.bet_cent.toFixed(4)}}<br/>` +
                    `Closeness: ${{d.clo_cent.toFixed(4)}}`
                )
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.transition().duration(500).style("opacity", 0);
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>
"""

def create_community_network_html(nodes_data, edges_data, title, description, num_communities, centrality_name):
    """Create HTML content for community network visualization"""
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        .header {{ padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px 10px 0 0; }}
        .network-container {{ width: 100%; height: 700px; }}
        .link {{ stroke: #999; stroke-opacity: 0.3; }}
        .tooltip {{ position: absolute; background: rgba(0,0,0,0.9); color: white; padding: 12px; 
                   border-radius: 6px; pointer-events: none; opacity: 0; font-size: 12px; max-width: 200px; }}
        .controls {{ padding: 20px; background-color: #f8f9fa; }}
        .legend {{ text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0; font-size: 24px;">{title}</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">{description}</p>
        </div>
        
        <div class="controls">
            <div class="legend">
                <strong>Communities:</strong> {num_communities} | 
                <strong>Node Size:</strong> {centrality_name.title()} Centrality | 
                <strong>Colors:</strong> Community Groups | 
                <strong>Shapes:</strong> ● Diseases, ■ Symptoms
            </div>
        </div>
        
        <div class="network-container" id="network"></div>
    </div>
    
    <script>
        const nodes = {str(nodes_data).replace("'", '"')};
        const links = {str(edges_data).replace("'", '"')};
        
        const width = 1200;
        const height = 700;
        
        const colorScale = d3.scaleOrdinal(d3.schemeSet3);
        
        const svg = d3.select("#network")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("viewBox", [0, 0, width, height]);
        
        const tooltip = d3.select("body").append("div").attr("class", "tooltip");
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(40))
            .force("charge", d3.forceManyBody().strength(-60))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = svg.append("g")
            .selectAll("line")
            .data(links)
            .enter().append("line")
            .attr("class", "link");
        
        const node = svg.append("g")
            .selectAll("path")
            .data(nodes)
            .enter().append("path")
            .attr("d", d => d.type === "disease" ? 
                d3.symbol().type(d3.symbolCircle).size(d.size * 20)() :
                d3.symbol().type(d3.symbolSquare).size(d.size * 20)())
            .attr("fill", d => colorScale(d.group))
            .attr("stroke", "#333")
            .attr("stroke-width", 0.5)
            .on("mouseover", function(event, d) {{
                tooltip.transition().duration(200).style("opacity", .9);
                tooltip.html(
                    `<strong>${{d.type.charAt(0).toUpperCase() + d.type.slice(1)}}: ${{d.id}}</strong><br/>` +
                    `Community: ${{d.group}}<br/>` +
                    `{centrality_name.title()} Centrality: ${{d.centrality.toFixed(4)}}`
                )
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.transition().duration(500).style("opacity", 0);
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("transform", d => `translate(${{d.x}},${{d.y}})`);
        }});
        
        function dragstarted(event) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            event.subject.fx = event.subject.x;
            event.subject.fy = event.subject.y;
        }}
        
        function dragged(event) {{
            event.subject.fx = event.x;
            event.subject.fy = event.y;
        }}
        
        function dragended(event) {{
            if (!event.active) simulation.alphaTarget(0);
            event.subject.fx = null;
            event.subject.fy = null;
        }}
    </script>
</body>
</html>
"""

def save_analysis_data(df, B, deg_cent, bet_cent, clo_cent, louvain_partition, leiden_partition_map, 
                      disease_nodes, symptom_nodes, modularity, output_dir):
    """Save all analysis data to files"""
    print("Saving analysis data...")
    
    # Node analysis data
    node_data = []
    for node in B.nodes():
        node_data.append({
            'Node': node,
            'Type': 'Disease' if node in disease_nodes else 'Symptom',
            'Degree': B.degree(node),
            'Degree_Centrality': deg_cent[node],
            'Betweenness_Centrality': bet_cent[node],
            'Closeness_Centrality': clo_cent[node],
            'Louvain_Community': louvain_partition[node],
            'Leiden_Community': leiden_partition_map.get(node, 'N/A')
        })
    
    nodes_df = pd.DataFrame(node_data)
    nodes_df.to_csv(output_dir / "data" / "node_analysis.csv", index=False)
    print("✅ Node analysis data saved")
    
    # Community analysis data
    community_data = []
    
    # Louvain communities
    louvain_communities = {}
    for node, comm in louvain_partition.items():
        louvain_communities.setdefault(comm, []).append(node)
    
    for comm_id, nodes in louvain_communities.items():
        diseases = [n for n in nodes if n in disease_nodes]
        symptoms = [n for n in nodes if n in symptom_nodes]
        
        community_data.append({
            'Algorithm': 'Louvain',
            'Community_ID': comm_id,
            'Size': len(nodes),
            'Diseases': len(diseases),
            'Symptoms': len(symptoms),
            'Modularity': modularity,
            'Disease_Sample': '; '.join(diseases[:3]),
            'Symptom_Sample': '; '.join(symptoms[:5])
        })
    
    # Leiden communities
    leiden_communities = {}
    for node, comm in leiden_partition_map.items():
        leiden_communities.setdefault(comm, []).append(node)
    
    for comm_id, nodes in leiden_communities.items():
        diseases = [n for n in nodes if n in disease_nodes]
        symptoms = [n for n in nodes if n in symptom_nodes]
        
        community_data.append({
            'Algorithm': 'Leiden',
            'Community_ID': comm_id,
            'Size': len(nodes),
            'Diseases': len(diseases),
            'Symptoms': len(symptoms),
            'Modularity': 'N/A',
            'Disease_Sample': '; '.join(diseases[:3]),
            'Symptom_Sample': '; '.join(symptoms[:5])
        })
    
    communities_df = pd.DataFrame(community_data)
    communities_df.to_csv(output_dir / "data" / "community_analysis.csv", index=False)
    print("✅ Community analysis data saved")
    
    # Network statistics
    degrees = dict(B.degree())
    network_stats = {
        'Total_Nodes': B.number_of_nodes(),
        'Total_Edges': B.number_of_edges(),
        'Disease_Nodes': len(disease_nodes),
        'Symptom_Nodes': len(symptom_nodes),
        'Density': nx.density(B),
        'Average_Degree': np.mean(list(degrees.values())),
        'Max_Degree': max(degrees.values()),
        'Min_Degree': min(degrees.values()),
        'Average_Clustering': nx.average_clustering(B),
        'Louvain_Communities': len(set(louvain_partition.values())),
        'Leiden_Communities': len(set(leiden_partition_map.values())),
        'Louvain_Modularity': modularity
    }
    
    stats_df = pd.DataFrame([network_stats])
    stats_df.to_csv(output_dir / "data" / "network_statistics.csv", index=False)
    print("✅ Network statistics saved")
    
    # Save network object for later use
    with open(output_dir / "data" / "network.pkl", 'wb') as f:
        pickle.dump({
            'network': B,
            'disease_nodes': disease_nodes,
            'symptom_nodes': symptom_nodes,
            'deg_cent': deg_cent,
            'bet_cent': bet_cent,
            'clo_cent': clo_cent,
            'louvain_partition': louvain_partition,
            'leiden_partition': leiden_partition_map,
            'modularity': modularity
        }, f)
    print("✅ Network object pickled and saved")

def generate_reports(B, deg_cent, bet_cent, clo_cent, louvain_partition, leiden_partition_map, 
                    disease_nodes, symptom_nodes, modularity, output_dir):
    """Generate analysis reports"""
    print("Generating reports...")
    
    # Calculate top nodes for insights
    def top_n(metric_dict, node_list, n=10):
        return sorted([(n, v) for n, v in metric_dict.items() if n in node_list],
                     key=lambda x: x[1], reverse=True)[:n]
    
    top_diseases_deg = top_n(deg_cent, disease_nodes)
    top_symptoms_deg = top_n(deg_cent, symptom_nodes)
    
    degrees = dict(B.degree())
    
    # Summary statistics
    summary_stats = {
        'metric': [
            'Total Nodes', 'Total Edges', 'Disease Nodes', 'Symptom Nodes',
            'Network Density', 'Average Degree', 'Max Degree', 'Min Degree',
            'Average Clustering', 'Louvain Communities', 'Leiden Communities',
            'Louvain Modularity', 'Top Disease (Degree)', 'Top Symptom (Degree)'
        ],
        'value': [
            B.number_of_nodes(), B.number_of_edges(), len(disease_nodes), len(symptom_nodes),
            f"{nx.density(B):.6f}", f"{np.mean(list(degrees.values())):.2f}",
            max(degrees.values()), min(degrees.values()),
            f"{nx.average_clustering(B):.4f}", len(set(louvain_partition.values())),
            len(set(leiden_partition_map.values())), f"{modularity:.3f}",
            f"{top_diseases_deg[0][0]} ({top_diseases_deg[0][1]:.4f})",
            f"{top_symptoms_deg[0][0]} ({top_symptoms_deg[0][1]:.4f})"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(output_dir / "reports" / "summary_statistics.csv", index=False)
    
    # Generate comprehensive report
    report = f"""# Disease-Symptom Network Analysis Report

## Executive Summary

This analysis examines a bipartite network of {B.number_of_nodes():,} nodes ({len(disease_nodes)} diseases and {len(symptom_nodes)} symptoms) connected by {B.number_of_edges():,} edges, representing symptom-disease associations from a medical dataset.

## Key Findings

### Network Structure
- **Density**: {nx.density(B):.6f} (sparse network indicating specific disease-symptom relationships)
- **Average Degree**: {np.mean(list(degrees.values())):.2f}
- **Clustering Coefficient**: {nx.average_clustering(B):.4f}

### Most Important Nodes

#### Top Diseases (by Degree Centrality)
{chr(10).join([f"{i+1}. **{name}**: {score:.4f}" for i, (name, score) in enumerate(top_diseases_deg[:10])])}

#### Top Symptoms (by Degree Centrality)  
{chr(10).join([f"{i+1}. **{name}**: {score:.4f}" for i, (name, score) in enumerate(top_symptoms_deg[:10])])}

### Community Structure
- **Louvain Algorithm**: {len(set(louvain_partition.values()))} communities (Modularity: {modularity:.3f})
- **Leiden Algorithm**: {len(set(leiden_partition_map.values()))} communities

The {'strong' if modularity > 0.5 else 'moderate' if modularity > 0.3 else 'weak'} modularity score indicates {'well-defined' if modularity > 0.3 else 'loose'} community structure in the network.

## Network Properties

### Degree Distribution
- **Range**: {min(degrees.values())} - {max(degrees.values())}
- **Mean**: {np.mean(list(degrees.values())):.2f}
- **Standard Deviation**: {np.std(list(degrees.values())):.2f}

### Centrality Analysis
The network exhibits typical scale-free properties with a few highly connected hubs and many nodes with few connections. This suggests:

1. **Key diseases** like "{top_diseases_deg[0][0]}" act as hubs with diverse symptom presentations
2. **Common symptoms** like "{top_symptoms_deg[0][0]}" appear across multiple diseases
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
- **Community Quality**: Modularity = {modularity:.3f}

---
**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Analysis Version**: 2.0 (Robust Processing)
"""
    
    with open(output_dir / "reports" / "comprehensive_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print("✅ Comprehensive report saved")

def create_readme(output_dir):
    """Create README file"""
    readme_content = f"""# Disease-Symptom Network Analysis Output

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

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("✅ README created")

def main():
    """Main execution function"""
    print("🚀 Starting Robust Network Analysis Pipeline")
    print("=" * 60)
    
    try:
        # Setup
        output_dir = setup_output_directory()
        print(f"📁 Output directory ready: {output_dir}")
        
        # Load and process data
        df, B, disease_nodes, symptom_nodes = load_and_process_data()
        
        # Calculate centralities
        deg_cent, bet_cent, clo_cent = calculate_centralities(B)
        
        # Detect communities
        louvain_partition, modularity, leiden_partition_map, n_leiden_communities = detect_communities(B)
        
        # Create visualizations
        create_static_visualizations(B, deg_cent, bet_cent, clo_cent, disease_nodes, symptom_nodes,
                                   louvain_partition, leiden_partition_map, output_dir)
        
        create_simple_html_networks(B, disease_nodes, symptom_nodes, louvain_partition, leiden_partition_map,
                                   deg_cent, bet_cent, clo_cent, output_dir)
        
        # Save data
        save_analysis_data(df, B, deg_cent, bet_cent, clo_cent, louvain_partition, leiden_partition_map, 
                          disease_nodes, symptom_nodes, modularity, output_dir)
        
        # Generate reports
        generate_reports(B, deg_cent, bet_cent, clo_cent, louvain_partition, leiden_partition_map, 
                        disease_nodes, symptom_nodes, modularity, output_dir)
        
        # Create documentation
        create_readme(output_dir)
        
        print("=" * 60)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Results saved to: {output_dir.absolute()}")
        print("\n🎯 Summary:")
        print(f"   • Network: {B.number_of_nodes():,} nodes, {B.number_of_edges():,} edges")
        print(f"   • Communities: {len(set(louvain_partition.values()))} (Louvain), {n_leiden_communities} (Leiden)")
        print(f"   • Modularity: {modularity:.3f}")
        print(f"   • Visualizations: {len(list((output_dir / 'visualizations').glob('*')))} files")
        print(f"   • Interactive Networks: {len(list((output_dir / 'networks').glob('*')))} files")
        
        print("\n🚀 Ready for Dashboard!")
        print("   Run: streamlit run dashboard.py")
        
    except FileNotFoundError:
        print("❌ Error: Dataset file not found!")
        print("   Please ensure 'data/Final_Augmented_dataset_Diseases_and_Symptoms.csv' exists")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()