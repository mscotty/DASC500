import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def analyze_correlations(df, numeric_columns):
    """Explore correlation matrix"""
    st.subheader("Correlation Matrix")

    if len(numeric_columns) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    corr_method = st.radio(
        "Select correlation method:",
        ["Pearson", "Spearman", "Kendall"],
        horizontal=True,
        key="corr_method_tab2"
    )

    num_cols_to_show = st.slider(
        "Number of columns to include:",
        min_value=2,
        max_value=min(20, len(numeric_columns)),
        value=min(10, len(numeric_columns))
    )

    selected_num_cols = st.multiselect(
        "Select columns for correlation matrix:",
        numeric_columns,
        default=numeric_columns[:num_cols_to_show]
    )

    if len(selected_num_cols) > 1:
        try:
            with st.spinner("Calculating correlations..."):
                corr_df = df[selected_num_cols].corr(method=corr_method.lower())

                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show strongest correlations
                st.subheader("Strongest Correlations")

                # Create a DataFrame with all pairwise correlations
                corr_pairs = []
                for i in range(len(selected_num_cols)):
                    for j in range(i + 1, len(selected_num_cols)):
                        col1 = selected_num_cols[i]
                        col2 = selected_num_cols[j]
                        corr_value = corr_df.loc[col1, col2]
                        corr_pairs.append(
                            {
                                "Variable 1": col1,
                                "Variable 2": col2,
                                "Correlation": corr_value,
                                "Abs Correlation": abs(corr_value),
                            }
                        )

                if corr_pairs:
                    corr_pairs_df = pd.DataFrame(corr_pairs)
                    top_corr = corr_pairs_df.sort_values("Abs Correlation", ascending=False).head(10)

                    st.dataframe(
                        top_corr[["Variable 1", "Variable 2", "Correlation"]]
                        .style.background_gradient(cmap="RdBu_r", subset=["Correlation"])
                        .format({"Correlation": "{:.4f}"})
                    )

                    # Visualize top correlation as scatter plot
                    if len(corr_pairs) > 0 and st.checkbox("Visualize top correlation"):
                        top_pair = top_corr.iloc[0]
                        var1 = top_pair["Variable 1"]
                        var2 = top_pair["Variable 2"]
                        corr_val = top_pair["Correlation"]

                        # Sample for large datasets
                        plot_df = df
                        if len(df) > 5000:
                            plot_df = df.sample(5000, random_state=42)
                            st.info("Showing a sample of 5,000 points for better performance.")

                        fig = px.scatter(
                            plot_df,
                            x=var1,
                            y=var2,
                            trendline="ols",
                            title=f"Strongest Correlation: {var1} vs {var2} (r = {corr_val:.4f})",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Network graph of correlations
                if len(selected_num_cols) > 2 and st.checkbox("Show correlation network graph"):
                    st.subheader("Correlation Network")
                    st.write("This visualization shows the strength of correlations between variables as a network.")

                    corr_threshold = st.slider(
                        "Correlation threshold (absolute value):",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.05,
                    )

                    # Create network data
                    network_nodes = []
                    network_edges = []

                    # Add nodes
                    for i, col in enumerate(selected_num_cols):
                        network_nodes.append({"id": col, "label": col, "size": 20})

                    # Add edges for correlations above threshold
                    for pair in corr_pairs:
                        if abs(pair["Correlation"]) >= corr_threshold:
                            network_edges.append(
                                {
                                    "source": pair["Variable 1"],
                                    "target": pair["Variable 2"],
                                    "value": abs(pair["Correlation"]),
                                    "color": "red" if pair["Correlation"] < 0 else "blue",
                                }
                            )

                    if not network_edges:
                        st.info(
                            f"No correlations meet the threshold of {corr_threshold}. Try lowering the threshold."
                        )
                    else:
                        # Create a network graph using plotly
                        import networkx as nx

                        G = nx.Graph()

                        # Add nodes
                        for node in network_nodes:
                            G.add_node(node["id"])

                        # Add edges
                        for edge in network_edges:
                            G.add_edge(edge["source"], edge["target"], weight=edge["value"], color=edge["color"])

                        # Use spring layout for node positions
                        pos = nx.spring_layout(G, seed=42)

                        # Create edge trace
                        edge_x = []
                        edge_y = []
                        edge_colors = []
                        edge_widths = []

                        for edge in G.edges(data=True):
                            x0, y0 = pos[edge[0]]
                            x1, y1 = pos[edge[1]]
                            edge_x.extend([x0, x1, None])
                            edge_y.extend([y0, y1, None])
                            edge_colors.append(edge[2]["color"])
                            edge_widths.append(edge[2]["weight"] * 5)  # Scale width by correlation

                        edge_trace = go.Scatter(
                            x=edge_x,
                            y=edge_y,
                            line=dict(width=1, color="#888"),
                            hoverinfo="none",
                            mode="lines",
                            line_color=edge_colors,
                            line_width=edge_widths,
                        )

                        # Create node trace
                        node_x = []
                        node_y = []
                        node_text = []

                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            node_text.append(node)

                        node_trace = go.Scatter(
                            x=node_x,
                            y=node_y,
                            mode="markers+text",
                            text=node_text,
                            textposition="top center",
                            hoverinfo="text",
                            marker=dict(
                                showscale=False,
                                color="rgba(66, 135, 245, 0.8)",
                                size=15,
                                line_width=2,
                            ),
                        )

                        # Create figure
                        fig = go.Figure(
                            data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode="closest",
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                title=f"Correlation Network (threshold: {corr_threshold})",
                            ),
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.write("**Legend:**")
                        st.write("- Blue edges: Positive correlations")
                        st.write("- Red edges: Negative correlations")
                        st.write("- Edge thickness: Strength of correlation")

        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
            import traceback

            st.error(f"Details: {traceback.format_exc()}")
    else:
        st.info("Please select at least 2 columns for correlation analysis.")
