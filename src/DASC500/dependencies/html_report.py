from datetime import datetime
import numpy as np
import pandas as pd
import os

def create_html_report(results, df, output_dir):
    """
    Creates a comprehensive HTML report for outlier detection results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing all outlier detection results
    df : pandas DataFrame
        The original DataFrame analyzed
    output_dir : str
        Directory to save the HTML report
    """
    timestamp = results.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    # Create HTML header and styling
    html_report = f"""
    <html>
    <head>
        <title>Outlier Detection Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; color: #333; }}
            h1, h2, h3, h4 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }}
            .subsection {{ margin-left: 20px; margin-bottom: 20px; }}
            .stats {{ margin-left: 20px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .warning {{ color: orange; }}
            .critical {{ color: red; }}
            .method-comparison {{ display: flex; justify-content: space-between; flex-wrap: wrap; }}
            .method-box {{ flex: 1; min-width: 300px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .highlight {{ background-color: #ffffcc; }}
            .container {{ display: flex; flex-wrap: wrap; }}
            .chart {{ flex: 1; min-width: 400px; margin: 10px; border: 1px solid #eee; border-radius: 5px; padding: 10px; }}
            .summary-stats {{ display: flex; flex-wrap: wrap; }}
            .stat-box {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin: 10px 0; }}
            .stat-label {{ font-size: 14px; color: #7f8c8d; }}
            .tabs {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
            .tabs button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
            .tabs button:hover {{ background-color: #ddd; }}
            .tabs button.active {{ background-color: #ccc; }}
            .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
            .comparison-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
            .comparison-item {{ flex: 1; min-width: 300px; margin: 10px; }}
            .before-after {{ display: flex; flex-direction: column; }}
            .before-after img {{ max-width: 100%; margin-bottom: 10px; }}
            .improvement {{ color: green; }}
            .degradation {{ color: red; }}
        </style>
        <script>
            function openTab(evt, tabName) {{
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {{
                    tabcontent[i].style.display = "none";
                }}
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {{
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }}
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }}
            
            // Default to open the first tab
            window.onload = function() {{
                document.getElementsByClassName("tablinks")[0].click();
            }};
        </script>
    </head>
    <body>
        <h1>Outlier Detection Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Dataset Overview</h2>
            <div class="stats">
                <p>Rows: {df.shape[0]:,}</p>
                <p>Columns: {df.shape[1]:,}</p>
                <p>Numerical columns analyzed: {len(results.get('columns_analyzed', []))}</p>
                <p>Methods used: {', '.join(results.get('methods', []))}</p>
            </div>
        </div>
    """
    
    # Add summary statistics section
    html_report += """
        <div class="section">
            <h2>Outlier Detection Summary</h2>
            <div class="summary-stats">
    """
    
    # Add stats for Z-score method
    if 'zscore_results' in results and results['zscore_results']:
        zscore_count = results['zscore_results'].get('outlier_count', 0)
        zscore_pct = results['zscore_results'].get('outlier_percentage', 0)
        zscore_threshold = results.get('zscore_threshold', 3.0)
        
        html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">Z-Score Outliers (threshold={zscore_threshold})</div>
                    <div class="stat-value">{zscore_count:,}</div>
                    <div class="stat-label">{zscore_pct:.2f}% of rows</div>
                </div>
        """
    
    # Add stats for IQR method
    if 'iqr_results' in results and results['iqr_results']:
        iqr_outliers_df = results['iqr_results'].get('all_outliers_df', pd.DataFrame())
        iqr_count = len(iqr_outliers_df) if not iqr_outliers_df.empty else 0
        iqr_pct = iqr_count / (df.shape[0] * len(results.get('columns_analyzed', []))) * 100 if iqr_count > 0 else 0
        iqr_k = results.get('iqr_k', 1.5)
        
        html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">IQR Outliers (k={iqr_k})</div>
                    <div class="stat-value">{iqr_count:,}</div>
                    <div class="stat-label">{iqr_pct:.2f}% of data points</div>
                </div>
        """
    
    # Add stats for PCA method if available
    if 'pca_results' in results and results['pca_results'] and 'outliers' in results['pca_results']:
        pca_outlier_count = len(results['pca_results']['outliers'].get('outlier_indices', []))
        pca_pct = pca_outlier_count / df.shape[0] * 100 if pca_outlier_count > 0 else 0
        
        html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">PCA-based Outliers</div>
                    <div class="stat-value">{pca_outlier_count:,}</div>
                    <div class="stat-label">{pca_pct:.2f}% of rows</div>
                </div>
        """
    
    # Add combined stats if available
    if 'combined_results' in results and 'agreement_counts' in results['combined_results']:
        agreement_counts = results['combined_results']['agreement_counts']
        max_agreement = max(agreement_counts.keys()) if agreement_counts else 0
        consensus_count = len(agreement_counts.get(max_agreement, [])) if agreement_counts else 0
        consensus_pct = consensus_count / df.shape[0] * 100 if consensus_count > 0 else 0
        
        html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">Consensus Outliers</div>
                    <div class="stat-value">{consensus_count:,}</div>
                    <div class="stat-label">{consensus_pct:.2f}% of rows (detected by {max_agreement} methods)</div>
                </div>
        """
    
    # Add cleaned data stats if available
    if 'cleaned_data_results' in results and 'removal_info' in results['cleaned_data_results']:
        removal_info = results['cleaned_data_results']['removal_info']
        outliers_removed = removal_info.get('outliers_removed', 0)
        removal_method = removal_info.get('method', 'N/A')
        reduction_pct = removal_info.get('reduction_percentage', 0)
        
        html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">Cleaned Data (method: {removal_method})</div>
                    <div class="stat-value">{outliers_removed:,}</div>
                    <div class="stat-label">{reduction_pct:.2f}% outliers removed</div>
                </div>
        """
    
    html_report += """
            </div>
        </div>
    """
    
    # Create tabs for different methods
    html_report += """
        <div class="section">
            <h2>Detailed Results</h2>
            
            <div class="tabs">
    """
    
    # Create a list to keep track of all tabs
    tab_ids = []
    
    # Add descriptive statistics tab first if available
    if 'descriptive_stats' in results and results['descriptive_stats']:
        html_report += """
                <button class="tablinks" onclick="openTab(event, 'stats')">Descriptive Statistics</button>
        """
        tab_ids.append('stats')
    
    # Add tab buttons for each method
    if 'zscore_results' in results and results['zscore_results']:
        html_report += """
                <button class="tablinks" onclick="openTab(event, 'zscore')">Z-Score Analysis</button>
        """
        tab_ids.append('zscore')
    
    if 'iqr_results' in results and results['iqr_results']:
        html_report += """
                <button class="tablinks" onclick="openTab(event, 'iqr')">IQR Analysis</button>
        """
        tab_ids.append('iqr')
    
    if 'pca_results' in results and results['pca_results']:
        html_report += """
                <button class="tablinks" onclick="openTab(event, 'pca')">PCA Analysis</button>
        """
        tab_ids.append('pca')
    
    if 'combined_results' in results and results['combined_results']:
        html_report += """
                <button class="tablinks" onclick="openTab(event, 'combined')">Combined Analysis</button>
        """
        tab_ids.append('combined')
    
    # Add cleaned data tab if available
    if 'cleaned_data_results' in results and 'removal_info' in results['cleaned_data_results']:
        html_report += """
                <button class="tablinks" onclick="openTab(event, 'cleaned')">Cleaned Data Analysis</button>
        """
        tab_ids.append('cleaned')
    
    html_report += """
            </div>
    """
    
    # Add Descriptive Statistics tab content
    if 'descriptive_stats' in results and results['descriptive_stats']:
        stats_results = results['descriptive_stats']
        
        html_report += """
            <div id="stats" class="tabcontent">
                <h3>Descriptive Statistics Analysis</h3>
        """
        
        # Add basic statistics table
        if 'stats' in stats_results and not stats_results['stats'].empty:
            stats_df = stats_results['stats']
            
            html_report += """
                <div class="subsection">
                    <h4>Basic and Advanced Statistics</h4>
                    <div style="overflow-x:auto;">
                        <table>
                            <tr>
                                <th>Column</th>
                                <th>Count</th>
                                <th>Mean</th>
                                <th>Std</th>
                                <th>Min</th>
                                <th>25%</th>
                                <th>50%</th>
                                <th>75%</th>
                                <th>Max</th>
                                <th>Skewness</th>
                                <th>Kurtosis</th>
                                <th>IQR</th>
                                <th>CV</th>
                                <th>Missing %</th>
                            </tr>
            """
            
            for col, row in stats_df.iterrows():
                # Highlight highly skewed or high-kurtosis columns
                skew_class = 'warning' if abs(row.get('skewness', 0)) > 1 else ''
                kurt_class = 'warning' if abs(row.get('kurtosis', 0)) > 3 else ''
                
                html_report += f"""
                            <tr>
                                <td>{col}</td>
                                <td>{row.get('count', 'N/A')}</td>
                                <td>{row.get('mean', 'N/A'):.4f}</td>
                                <td>{row.get('std', 'N/A'):.4f}</td>
                                <td>{row.get('min', 'N/A'):.4f}</td>
                                <td>{row.get('25%', 'N/A'):.4f}</td>
                                <td>{row.get('50%', 'N/A'):.4f}</td>
                                <td>{row.get('75%', 'N/A'):.4f}</td>
                                <td>{row.get('max', 'N/A'):.4f}</td>
                                <td class="{skew_class}">{row.get('skewness', 'N/A'):.4f}</td>
                                <td class="{kurt_class}">{row.get('kurtosis', 'N/A'):.4f}</td>
                                <td>{row.get('iqr', 'N/A'):.4f}</td>
                                <td>{row.get('cv', 'N/A'):.4f}</td>
                                <td>{row.get('missing_percent', 'N/A'):.2f}%</td>
                            </tr>
                """
            
            html_report += """
                        </table>
                    </div>
                </div>
            """
        
                # Add normality test results if available
        if 'normality_tests' in stats_results and not stats_results['normality_tests'].empty:
            normality_df = stats_results['normality_tests']
            
            html_report += """
                <div class="subsection">
                    <h4>Normality Tests (Shapiro-Wilk)</h4>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Test Statistic</th>
                            <th>p-value</th>
                            <th>Normality Assessment</th>
                        </tr>
            """
            
            for col, row in normality_df.iterrows():
                p_value = row.get('shapiro_p_value', np.nan)
                is_normal = row.get('is_normal', False)
                
                # Determine CSS class based on normality
                css_class = '' if pd.isna(p_value) else ('normal' if is_normal else 'warning')
                
                # Format p-value with scientific notation for very small values
                if not pd.isna(p_value) and p_value < 0.0001:
                    p_value_str = f"{p_value:.2e}"
                else:
                    p_value_str = f"{p_value:.4f}" if not pd.isna(p_value) else "N/A"
                
                html_report += f"""
                            <tr class="{css_class}">
                                <td>{col}</td>
                                <td>{row.get('shapiro_stat', 'N/A'):.4f}</td>
                                <td>{p_value_str}</td>
                                <td>{"Normal" if is_normal else "Not normal" if not pd.isna(p_value) else "Insufficient data"}</td>
                            </tr>
                """
            
            html_report += """
                        </table>
                </div>
            """
        
        # Add plots if available
        if 'plots' in stats_results:
            plot_results = stats_results['plots']
            
            # Add boxplots
            if 'boxplots' in plot_results and plot_results['boxplots']:
                boxplot_path = plot_results['boxplots'].get('boxplots', '')
                if boxplot_path:
                    # Extract just the filename from the path
                    boxplot_filename = os.path.basename(boxplot_path)
                    
                    html_report += f"""
                        <div class="subsection">
                            <h4>Box Plots</h4>
                            <div class="chart" style="flex-basis: 100%;">
                                <img src="descriptive_stats/plots/{boxplot_filename}" alt="Box Plots" style="max-width:100%;">
                            </div>
                        </div>
                    """
            
            # Add histograms
            if 'histograms' in plot_results and plot_results['histograms']:
                histogram_path = plot_results['histograms'].get('histograms', '')
                if histogram_path:
                    # Extract just the filename from the path
                    histogram_filename = os.path.basename(histogram_path)
                    
                    html_report += f"""
                        <div class="subsection">
                            <h4>Histograms</h4>
                            <div class="chart" style="flex-basis: 100%;">
                                <img src="descriptive_stats/plots/{histogram_filename}" alt="Histograms" style="max-width:100%;">
                            </div>
                        </div>
                    """
            
            # Add violin plots
            if 'violin_plots' in plot_results and plot_results['violin_plots']:
                violin_path = plot_results['violin_plots'].get('violin_plots', '')
                if violin_path:
                    # Extract just the filename from the path
                    violin_filename = os.path.basename(violin_path)
                    
                    html_report += f"""
                        <div class="subsection">
                            <h4>Violin Plots</h4>
                            <div class="chart" style="flex-basis: 100%;">
                                <img src="descriptive_stats/plots/{violin_filename}" alt="Violin Plots" style="max-width:100%;">
                            </div>
                        </div>
                    """
        
        # Add observations about the distributions
        html_report += """
                <div class="subsection">
                    <h4>Distribution Observations</h4>
                    <ul>
        """
        
        # Add observations based on skewness and kurtosis
        if 'stats' in stats_results and not stats_results['stats'].empty:
            stats_df = stats_results['stats']
            
            for col, row in stats_df.iterrows():
                skewness = row.get('skewness', 0)
                kurtosis = row.get('kurtosis', 0)
                
                observations = []
                
                # Skewness observations
                if abs(skewness) < 0.5:
                    observations.append(f"approximately symmetric (skewness = {skewness:.2f})")
                elif skewness > 0.5 and skewness < 1:
                    observations.append(f"moderately positively skewed (skewness = {skewness:.2f})")
                elif skewness > 1:
                    observations.append(f"highly positively skewed (skewness = {skewness:.2f})")
                elif skewness < -0.5 and skewness > -1:
                    observations.append(f"moderately negatively skewed (skewness = {skewness:.2f})")
                elif skewness < -1:
                    observations.append(f"highly negatively skewed (skewness = {skewness:.2f})")
                
                # Kurtosis observations
                if kurtosis < -1:
                    observations.append(f"platykurtic distribution - fewer outliers than normal (kurtosis = {kurtosis:.2f})")
                elif kurtosis > 1:
                    observations.append(f"leptokurtic distribution - more outliers than normal (kurtosis = {kurtosis:.2f})")
                
                # Add normality information if available
                if 'normality_tests' in stats_results and not stats_results['normality_tests'].empty:
                    normality_df = stats_results['normality_tests']
                    if col in normality_df.index:
                        is_normal = normality_df.loc[col, 'is_normal']
                        p_value = normality_df.loc[col, 'shapiro_p_value']
                        
                        if not pd.isna(is_normal):
                            if is_normal:
                                observations.append(f"normally distributed according to Shapiro-Wilk test (p = {p_value:.4f})")
                            else:
                                observations.append(f"not normally distributed according to Shapiro-Wilk test (p = {p_value:.4f})")
                
                if observations:
                    html_report += f"""
                        <li><strong>{col}</strong> is {', '.join(observations)}</li>
                    """
        
        html_report += """
                    </ul>
                </div>
        """
        
        html_report += """
            </div>
        """
    
    # Z-Score tab content
    if 'zscore_results' in results and results['zscore_results']:
        zscore_results = results['zscore_results']
        outliers_df = zscore_results.get('outliers_df', pd.DataFrame())
        
        html_report += """
            <div id="zscore" class="tabcontent">
                <h3>Z-Score Outlier Analysis</h3>
        """
        
        # Add outlier counts by column if available
        if 'column_outlier_counts' in zscore_results:
            html_report += """
                <div class="subsection">
                    <h4>Outliers by Column</h4>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Outlier Count</th>
                            <th>Percentage</th>
                        </tr>
            """
            
            for col, count in zscore_results['column_outlier_counts'].items():
                # Calculate percentage for this column
                col_pct = count / df.shape[0] * 100
                css_class = 'critical' if col_pct > 10 else ('warning' if col_pct > 5 else '')
                
                html_report += f"""
                        <tr class="{css_class}">
                            <td>{col}</td>
                            <td>{count:,}</td>
                            <td>{col_pct:.2f}%</td>
                        </tr>
                """
            
            html_report += """
                    </table>
                </div>
            """
        
        # Add visualization if available
        if 'plot_paths' in zscore_results and zscore_results['plot_paths']:
            html_report += """
                <div class="subsection">
                    <h4>Z-Score Distributions</h4>
                    <div class="container">
            """
            
            for col, plot_path in zscore_results['plot_paths'].items():
                # Extract just the filename from the path
                plot_filename = os.path.basename(plot_path)
                
                html_report += f"""
                        <div class="chart">
                            <h5>{col}</h5>
                            <img src="zscore/{plot_filename}" alt="Z-Score Distribution for {col}" style="max-width:100%;">
                        </div>
                """
            
            html_report += """
                    </div>
                </div>
            """
        
        # Add top outliers table if available
        if not outliers_df.empty:
            # Select a subset of columns to display
            display_cols = []
            for col in results.get('columns_analyzed', []):
                if col in outliers_df.columns:
                    display_cols.append(col)
                    # Also add the z-score column if available
                    z_col = f"{col}_zscore"
                    if z_col in outliers_df.columns:
                        display_cols.append(z_col)
            
            # Limit to first 100 outliers for display
            display_df = outliers_df.head(100)
            
            html_report += """
                <div class="subsection">
                    <h4>Top Outliers (Z-Score Method)</h4>
                    <p>Showing up to 100 outliers</p>
                    <div style="overflow-x:auto;">
                        <table>
                            <tr>
                                <th>Index</th>
            """
            
            for col in display_cols:
                html_report += f"""
                                <th>{col}</th>
                """
            
            html_report += """
                            </tr>
            """
            
            for idx, row in display_df.iterrows():
                html_report += f"""
                            <tr>
                                <td>{idx}</td>
                """
                
                for col in display_cols:
                    # Highlight extreme values
                    css_class = ''
                    if col.endswith('_zscore') and abs(row[col]) > 5:
                        css_class = 'critical'
                    elif col.endswith('_zscore') and abs(row[col]) > 3:
                        css_class = 'warning'
                    
                    value = row[col]
                                        # Format numbers nicely
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:,.0f}"
                    else:
                        formatted_value = str(value)
                    
                    html_report += f"""
                                <td class="{css_class}">{formatted_value}</td>
                    """
                
                html_report += """
                            </tr>
                """
            
            html_report += """
                        </table>
                    </div>
                </div>
            """
        
        html_report += """
            </div>
        """
    
    # IQR tab content
    if 'iqr_results' in results and results['iqr_results']:
        iqr_results = results['iqr_results']
        all_outliers_df = iqr_results.get('all_outliers_df', pd.DataFrame())
        
        html_report += """
            <div id="iqr" class="tabcontent">
                <h3>IQR Outlier Analysis</h3>
        """
        
        # Add column-specific IQR information
        if 'outliers_by_column' in iqr_results:
            html_report += """
                <div class="subsection">
                    <h4>IQR Analysis by Column</h4>
                    <table>
                        <tr>
                            <th>Column</th>
                            <th>Q1</th>
                            <th>Q3</th>
                            <th>IQR</th>
                            <th>Lower Bound</th>
                            <th>Upper Bound</th>
                            <th>Outlier Count</th>
                            <th>Percentage</th>
                        </tr>
            """
            
            for col, info in iqr_results['outliers_by_column'].items():
                if info['q1'] is not None:
                    # Calculate percentage for this column
                    col_pct = info['outlier_percentage']
                    css_class = 'critical' if col_pct > 10 else ('warning' if col_pct > 5 else '')
                    
                    html_report += f"""
                            <tr class="{css_class}">
                                <td>{col}</td>
                                <td>{info['q1']:.4f}</td>
                                <td>{info['q3']:.4f}</td>
                                <td>{info['iqr']:.4f}</td>
                                <td>{info['lower_bound']:.4f}</td>
                                <td>{info['upper_bound']:.4f}</td>
                                <td>{info['outlier_count']:,}</td>
                                <td>{col_pct:.2f}%</td>
                            </tr>
                    """
                else:
                    html_report += f"""
                            <tr>
                                <td>{col}</td>
                                <td colspan="7">Insufficient data for IQR analysis</td>
                            </tr>
                    """
            
            html_report += """
                    </table>
                </div>
            """
        
        # Add visualization if available
        if 'plots' in iqr_results:
            # Add boxplots
            if 'boxplots' in iqr_results['plots']:
                html_report += """
                    <div class="subsection">
                        <h4>IQR Boxplots</h4>
                        <div class="container">
                """
                
                # Show multi-column boxplot if available
                if 'multi_boxplot' in iqr_results['plots']:
                    html_report += """
                            <div class="chart" style="flex-basis: 100%;">
                                <h5>Multi-Column Boxplot</h5>
                                <img src="iqr/iqr_plots/multi_column_iqr_boxplot.png" alt="Multi-Column Boxplot" style="max-width:100%;">
                            </div>
                    """
                
                html_report += """
                        </div>
                    </div>
                """
            
                        # Add histograms
            if 'histograms' in iqr_results['plots']:
                html_report += """
                    <div class="subsection">
                        <h4>IQR Histograms</h4>
                        <div class="container">
                """
                
                for col in results.get('columns_analyzed', []):
                    safe_col_name = col.replace('/', '_').replace('\\', '_')
                    hist_filename = f'iqr_histogram_{safe_col_name}.png'
                    
                    html_report += f"""
                            <div class="chart">
                                <h5>{col}</h5>
                                <img src="iqr/iqr_plots/{hist_filename}" alt="IQR Histogram for {col}" style="max-width:100%;">
                            </div>
                    """
                
                html_report += """
                        </div>
                    </div>
                """
        
        # Add top outliers table if available
        if not all_outliers_df.empty:
            # Limit to first 100 outliers for display
            display_df = all_outliers_df.head(100)
            
            html_report += """
                <div class="subsection">
                    <h4>Top Outliers (IQR Method)</h4>
                    <p>Showing up to 100 outliers</p>
                    <div style="overflow-x:auto;">
                        <table>
                            <tr>
                                <th>Row Index</th>
                                <th>Column</th>
                                <th>Value</th>
                                <th>Lower Bound</th>
                                <th>Upper Bound</th>
                                <th>Type</th>
                            </tr>
            """
            
            for _, row in display_df.iterrows():
                # Determine if high or low outlier
                outlier_type = row['Type']
                css_class = 'critical' if outlier_type == 'High' else 'warning'
                
                html_report += f"""
                            <tr>
                                <td>{row['Row Index']}</td>
                                <td>{row['Column']}</td>
                                <td class="{css_class}">{row['Value']:.4f}</td>
                                <td>{row['Lower Bound']:.4f}</td>
                                <td>{row['Upper Bound']:.4f}</td>
                                <td>{outlier_type} Outlier</td>
                            </tr>
                """
            
            html_report += """
                        </table>
                    </div>
                </div>
            """
        
        html_report += """
            </div>
        """
    
    # PCA tab content
    if 'pca_results' in results and results['pca_results']:
        pca_results = results['pca_results']
        
        html_report += """
            <div id="pca" class="tabcontent">
                <h3>PCA-Based Outlier Analysis</h3>
        """
        
        # Add explained variance information
        if 'explained_variance' in pca_results:
            explained_variance = pca_results['explained_variance']
            cumulative_variance = np.cumsum(explained_variance)
            
            html_report += """
                <div class="subsection">
                    <h4>Principal Components Explained Variance</h4>
                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Explained Variance</th>
                            <th>Cumulative Variance</th>
                        </tr>
            """
            
            for i, var in enumerate(explained_variance):
                html_report += f"""
                        <tr>
                            <td>PC{i+1}</td>
                            <td>{var:.4f} ({var*100:.2f}%)</td>
                            <td>{cumulative_variance[i]:.4f} ({cumulative_variance[i]*100:.2f}%)</td>
                        </tr>
                """
            
            html_report += """
                    </table>
                </div>
            """
        
        # Add loadings information
        if 'loadings' in pca_results:
            loadings = pca_results['loadings']
            
            html_report += """
                <div class="subsection">
                    <h4>Top Feature Contributions</h4>
            """
            
            # For each principal component, show top contributing features
            for i in range(min(3, len(loadings))):  # Show first 3 PCs
                pc_name = f"PC{i+1}"
                
                html_report += f"""
                    <h5>{pc_name}</h5>
                    <table>
                        <tr>
                            <th>Feature</th>
                            <th>Loading</th>
                        </tr>
                """
                
                # Get loadings for this PC and sort by absolute value
                pc_loadings = {feature: loadings[pc_name][feature] for feature in loadings[pc_name]}
                sorted_features = sorted(pc_loadings.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Show top 5 features
                for feature, loading in sorted_features[:5]:
                    css_class = 'warning' if abs(loading) > 0.7 else ''
                    
                    html_report += f"""
                        <tr class="{css_class}">
                            <td>{feature}</td>
                            <td>{loading:.4f}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                """
            
            html_report += """
                </div>
            """
        
        # Add PCA-based outlier information
        if 'outliers' in pca_results:
            pca_outliers = pca_results['outliers']
            outlier_indices = pca_outliers.get('outlier_indices', [])
            
            html_report += """
                <div class="subsection">
                    <h4>PCA-Based Outliers</h4>
            """
            
            if outlier_indices:
                html_report += f"""
                    <p>Detected {len(outlier_indices)} outliers using Mahalanobis distance</p>
                    
                    <table>
                        <tr>
                            <th>Row Index</th>
                            <th>Mahalanobis Distance</th>
                            <th>Threshold</th>
                        </tr>
                """
                
                # Get distances and threshold
                distances = pca_outliers.get('mahalanobis_distances', {})
                threshold = pca_outliers.get('threshold', 0)
                
                for idx in outlier_indices[:100]:  # Limit to 100 for display
                    distance = distances.get(idx, 0)
                    css_class = 'critical' if distance > threshold * 1.5 else 'warning'
                    
                    html_report += f"""
                        <tr class="{css_class}">
                            <td>{idx}</td>
                            <td>{distance:.4f}</td>
                            <td>{threshold:.4f}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                """
                
                # Add PCA plot if available
                if 'plot_path' in pca_outliers:
                    plot_filename = os.path.basename(pca_outliers['plot_path'])
                    
                    html_report += f"""
                    <div class="chart" style="flex-basis: 100%;">
                        <h5>PCA Outlier Visualization</h5>
                        <img src="pca/{plot_filename}" alt="PCA Outliers" style="max-width:100%;">
                    </div>
                    """
            else:
                html_report += """
                    <p>No outliers detected using PCA-based methods.</p>
                """
            
            html_report += """
                </div>
            """
        
        html_report += """
            </div>
        """
    
    # Combined results tab
    if 'combined_results' in results and results['combined_results']:
        combined_results = results['combined_results']
        
        html_report += """
            <div id="combined" class="tabcontent">
                <h3>Combined Outlier Analysis</h3>
        """
        
        # Add method agreement information
        if 'agreement_counts' in combined_results:
            agreement_counts = combined_results['agreement_counts']
            
            html_report += """
                <div class="subsection">
                    <h4>Method Agreement</h4>
                    <table>
                        <tr>
                            <th>Number of Methods</th>
                            <th>Outlier Count</th>
                            <th>Percentage of Data</th>
                        </tr>
            """
            
            for method_count, indices in sorted(agreement_counts.items(), reverse=True):
                count = len(indices)
                percentage = count / df.shape[0] * 100
                css_class = 'critical' if method_count == len(results['methods']) else ''
                
                html_report += f"""
                        <tr class="{css_class}">
                            <td>{method_count}</td>
                            <td>{count:,}</td>
                            <td>{percentage:.2f}%</td>
                        </tr>
                """
            
            html_report += """
                    </table>
                </div>
            """
        
        # Add consensus outliers table
        if 'consensus_outliers' in combined_results:
            consensus_outliers = combined_results['consensus_outliers']
            
            if not consensus_outliers.empty:
                html_report += """
                    <div class="subsection">
                        <h4>Consensus Outliers</h4>
                        <p>Rows identified as outliers by multiple methods</p>
                        <div style="overflow-x:auto;">
                            <table>
                                <tr>
                                    <th>Row Index</th>
                                    <th>Methods</th>
                """
                
                # Add columns for the original data
                for col in results.get('columns_analyzed', [])[:10]:  # Limit to first 10 columns
                    html_report += f"""
                                    <th>{col}</th>
                    """
                
                html_report += """
                                </tr>
                """
                
                # Add rows for each consensus outlier
                for idx, row in consensus_outliers.head(100).iterrows():
                    methods = row['Methods']
                    method_count = len(methods.split(','))
                    css_class = 'critical' if method_count == len(results['methods']) else 'warning'
                    
                    html_report += f"""
                                <tr class="{css_class}">
                                    <td>{idx}</td>
                                    <td>{methods}</td>
                    """
                    
                    # Add original data values
                    for col in results.get('columns_analyzed', [])[:10]:
                        if col in df.columns:
                            value = df.loc[idx, col]
                            if isinstance(value, (int, float)):
                                formatted_value = f"{value:.4f}" if abs(value) < 1000 else f"{value:,.0f}"
                            else:
                                formatted_value = str(value)
                            
                            html_report += f"""
                                    <td>{formatted_value}</td>
                            """
                        else:
                            html_report += """
                                    <td>N/A</td>
                            """
                    
                    html_report += """
                                </tr>
                    """
                
                html_report += """
                            </table>
                        </div>
                    </div>
                """
            else:
                html_report += """
                    <div class="subsection">
                        <h4>Consensus Outliers</h4>
                        <p>No consensus outliers were found across multiple methods.</p>
                    </div>
                """
        
        # Add method comparison visualization if available
        if os.path.exists(os.path.join(output_dir, 'method_comparison_venn.png')):
            html_report += """
                <div class="subsection">
                    <h4>Method Comparison</h4>
                    <div class="chart" style="flex-basis: 100%;">
                        <img src="method_comparison_venn.png" alt="Method Comparison" style="max-width:100%;">
                    </div>
                </div>
            """
        elif os.path.exists(os.path.join(output_dir, 'method_agreement_counts.png')):
            html_report += """
                <div class="subsection">
                    <h4>Method Agreement Counts</h4>
                    <div class="chart" style="flex-basis: 100%;">
                        <img src="method_agreement_counts.png" alt="Method Agreement Counts" style="max-width:100%;">
                    </div>
                </div>
            """
        
        html_report += """
            </div>
        """
    
    # Add cleaned data analysis tab - NEW SECTION
    if 'cleaned_data_results' in results and 'removal_info' in results['cleaned_data_results']:
        removal_info = results['cleaned_data_results']['removal_info']
        
        html_report += """
            <div id="cleaned" class="tabcontent">
                <h3>Cleaned Data Analysis</h3>
        """
        
        # Add summary of outlier removal
        html_report += f"""
                <div class="subsection">
                    <h4>Outlier Removal Summary</h4>
                    <div class="summary-stats">
                        <div class="stat-box">
                            <div class="stat-label">Removal Method</div>
                            <div class="stat-value">{removal_info['method']}</div>
                            <div class="stat-label">Strategy for determining outliers</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Outliers Removed</div>
                            <div class="stat-value">{removal_info['outliers_removed']:,}</div>
                            <div class="stat-label">{removal_info.get('reduction_percentage', 0):.2f}% of original data</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Original Shape</div>
                            <div class="stat-value">{removal_info.get('original_shape', (0, 0))[0]:,} × {removal_info.get('original_shape', (0, 0))[1]}</div>
                            <div class="stat-label">Rows × Columns</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-label">Cleaned Shape</div>
                            <div class="stat-value">{removal_info.get('cleaned_shape', (0, 0))[0]:,} × {removal_info.get('cleaned_shape', (0, 0))[1]}</div>
                            <div class="stat-label">Rows × Columns</div>
                        </div>
                    </div>
                </div>
        """
        
        # Add before/after comparison plots
        if os.path.exists(os.path.join(output_dir, 'cleaned_data/before_after_comparison')):
            html_report += """
                <div class="subsection">
                    <h4>Before vs. After Outlier Removal</h4>
                    
                    <div class="comparison-container">
                        <div class="comparison-item">
                                                        <h5>Box Plot Comparison</h5>
                            <img src="cleaned_data/before_after_comparison/boxplot_comparison.png" alt="Box Plot Comparison" style="max-width:100%;">
                        </div>
                        <div class="comparison-item">
                            <h5>Histogram Comparison</h5>
                            <img src="cleaned_data/before_after_comparison/histogram_comparison.png" alt="Histogram Comparison" style="max-width:100%;">
                        </div>
                    </div>
                    
                    <h5>Summary Statistics Comparison</h5>
                    <img src="cleaned_data/before_after_comparison/summary_statistics_table.png" alt="Summary Statistics Comparison" style="max-width:100%;">
                    
                    <div class="comparison-container">
                        <div class="comparison-item">
                            <h5>Outlier Scatter Plots</h5>
                            <img src="cleaned_data/before_after_comparison/outlier_scatter_plots.png" alt="Outlier Scatter Plots" style="max-width:100%;">
                        </div>
                    </div>
                </div>
            """
        
        # Add cleaned data descriptive statistics
        if 'descriptive_stats' in results['cleaned_data_results']:
            cleaned_stats = results['cleaned_data_results']['descriptive_stats']
            
            html_report += """
                <div class="subsection">
                    <h4>Cleaned Data Statistics</h4>
            """
            
            if 'stats' in cleaned_stats and not cleaned_stats['stats'].empty:
                stats_df = cleaned_stats['stats']
                
                html_report += """
                    <div style="overflow-x:auto;">
                        <table>
                            <tr>
                                <th>Column</th>
                                <th>Count</th>
                                <th>Mean</th>
                                <th>Std</th>
                                <th>Min</th>
                                <th>25%</th>
                                <th>50%</th>
                                <th>75%</th>
                                <th>Max</th>
                                <th>Skewness</th>
                                <th>Kurtosis</th>
                            </tr>
                """
                
                for col, row in stats_df.iterrows():
                    html_report += f"""
                            <tr>
                                <td>{col}</td>
                                <td>{row.get('count', 'N/A')}</td>
                                <td>{row.get('mean', 'N/A'):.4f}</td>
                                <td>{row.get('std', 'N/A'):.4f}</td>
                                <td>{row.get('min', 'N/A'):.4f}</td>
                                <td>{row.get('25%', 'N/A'):.4f}</td>
                                <td>{row.get('50%', 'N/A'):.4f}</td>
                                <td>{row.get('75%', 'N/A'):.4f}</td>
                                <td>{row.get('max', 'N/A'):.4f}</td>
                                <td>{row.get('skewness', 'N/A'):.4f}</td>
                                <td>{row.get('kurtosis', 'N/A'):.4f}</td>
                            </tr>
                    """
                
                html_report += """
                        </table>
                    </div>
                """
            
            # Add cleaned data plots if available
            if 'plots' in cleaned_stats:
                plot_results = cleaned_stats['plots']
                
                html_report += """
                    <div class="comparison-container">
                """
                
                # Add boxplots
                if 'boxplots' in plot_results and plot_results['boxplots']:
                    boxplot_path = plot_results['boxplots'].get('boxplots', '')
                    if boxplot_path:
                        boxplot_filename = os.path.basename(boxplot_path)
                        
                        html_report += f"""
                        <div class="comparison-item">
                            <h5>Cleaned Data Box Plots</h5>
                            <img src="cleaned_data/descriptive_stats/plots/{boxplot_filename}" alt="Cleaned Data Box Plots" style="max-width:100%;">
                        </div>
                        """
                
                # Add histograms
                if 'histograms' in plot_results and plot_results['histograms']:
                    histogram_path = plot_results['histograms'].get('histograms', '')
                    if histogram_path:
                        histogram_filename = os.path.basename(histogram_path)
                        
                        html_report += f"""
                        <div class="comparison-item">
                            <h5>Cleaned Data Histograms</h5>
                            <img src="cleaned_data/descriptive_stats/plots/{histogram_filename}" alt="Cleaned Data Histograms" style="max-width:100%;">
                        </div>
                        """
                
                html_report += """
                    </div>
                """
            
            html_report += """
                </div>
            """
        
        # Add comparison of outlier detection results before and after cleaning
        html_report += """
            <div class="subsection">
                <h4>Outlier Detection After Cleaning</h4>
                <p>This section shows the results of running the same outlier detection methods on the cleaned dataset.</p>
        """
        
        # Compare Z-score results
        if 'zscore_results' in results['cleaned_data_results']:
            original_zscore_count = results['zscore_results'].get('outlier_count', 0)
            cleaned_zscore_count = results['cleaned_data_results']['zscore_results'].get('outlier_count', 0)
            zscore_reduction = 100 * (1 - cleaned_zscore_count / original_zscore_count) if original_zscore_count > 0 else 0
            
            html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">Z-Score Outliers</div>
                    <div class="stat-value">{original_zscore_count} -> {cleaned_zscore_count}</div>
                    <div class="stat-label improvement">{zscore_reduction:.2f}% reduction</div>
                </div>
            """
            
            # Add Z-score plot comparison if available
            if 'plot_paths' in results['cleaned_data_results']['zscore_results']:
                html_report += """
                <div class="comparison-container">
                    <h5>Z-Score Distributions After Cleaning</h5>
                """
                
                for col, plot_path in results['cleaned_data_results']['zscore_results']['plot_paths'].items():
                    plot_filename = os.path.basename(plot_path)
                    
                    html_report += f"""
                    <div class="comparison-item">
                        <h6>{col}</h6>
                        <img src="cleaned_data/zscore/{plot_filename}" alt="Cleaned Z-Score Distribution for {col}" style="max-width:100%;">
                    </div>
                    """
                
                html_report += """
                </div>
                """
        
        # Compare IQR results
        if 'iqr_results' in results['cleaned_data_results']:
            original_iqr_df = results['iqr_results'].get('all_outliers_df', pd.DataFrame())
            cleaned_iqr_df = results['cleaned_data_results']['iqr_results'].get('all_outliers_df', pd.DataFrame())
            
            original_iqr_count = len(original_iqr_df) if not original_iqr_df.empty else 0
            cleaned_iqr_count = len(cleaned_iqr_df) if not cleaned_iqr_df.empty else 0
            iqr_reduction = 100 * (1 - cleaned_iqr_count / original_iqr_count) if original_iqr_count > 0 else 0
            
            html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">IQR Outliers</div>
                    <div class="stat-value">{original_iqr_count} -> {cleaned_iqr_count}</div>
                    <div class="stat-label improvement">{iqr_reduction:.2f}% reduction</div>
                </div>
            """
            
            # Add IQR plot comparison if available
            if 'plots' in results['cleaned_data_results']['iqr_results'] and 'boxplots' in results['cleaned_data_results']['iqr_results']['plots']:
                html_report += """
                <div class="comparison-container">
                    <h5>IQR Boxplots After Cleaning</h5>
                    <div class="comparison-item">
                        <img src="cleaned_data/iqr/iqr_plots/multi_column_iqr_boxplot.png" alt="Cleaned IQR Boxplots" style="max-width:100%;">
                    </div>
                </div>
                """
        
        # Compare PCA results
        if 'pca_results' in results['cleaned_data_results'] and 'outliers' in results['cleaned_data_results']['pca_results']:
            original_pca_count = len(results['pca_results']['outliers'].get('outlier_indices', [])) if 'outliers' in results['pca_results'] else 0
            cleaned_pca_count = len(results['cleaned_data_results']['pca_results']['outliers'].get('outlier_indices', []))
            pca_reduction = 100 * (1 - cleaned_pca_count / original_pca_count) if original_pca_count > 0 else 0
            
            html_report += f"""
                <div class="stat-box">
                    <div class="stat-label">PCA-based Outliers</div>
                    <div class="stat-value">{original_pca_count} -> {cleaned_pca_count}</div>
                    <div class="stat-label improvement">{pca_reduction:.2f}% reduction</div>
                </div>
            """
            
            # Add PCA plot comparison if available
            if 'plot_path' in results['cleaned_data_results']['pca_results']['outliers']:
                plot_filename = os.path.basename(results['cleaned_data_results']['pca_results']['outliers']['plot_path'])
                
                html_report += f"""
                <div class="comparison-container">
                    <h5>PCA Outliers After Cleaning</h5>
                    <div class="comparison-item">
                        <img src="cleaned_data/pca/{plot_filename}" alt="Cleaned PCA Outliers" style="max-width:100%;">
                    </div>
                </div>
                """
        
        html_report += """
            </div>
        """
        
        # Add impact analysis section
        html_report += """
            <div class="subsection">
                <h4>Impact of Outlier Removal</h4>
                <p>This section summarizes the changes in data distribution and statistical properties after outlier removal.</p>
        """
        
        # Add table with changes in key statistics
        if os.path.exists(os.path.join(output_dir, 'cleaned_data/before_after_comparison/summary_statistics_comparison.csv')):
            try:
                # Read the CSV file with summary statistics comparison
                summary_comparison = pd.read_csv(os.path.join(output_dir, 'cleaned_data/before_after_comparison/summary_statistics_comparison.csv'), index_col=0)
                
                html_report += """
                <table>
                    <tr>
                        <th>Column</th>
                        <th>Original Mean</th>
                        <th>Cleaned Mean</th>
                        <th>Mean Change %</th>
                        <th>Original Std</th>
                        <th>Cleaned Std</th>
                        <th>Std Change %</th>
                    </tr>
                """
                
                for col, row in summary_comparison.iterrows():
                    mean_change = row['Mean_Change_%']
                    std_change = row['Std_Change_%']
                    
                    mean_class = 'warning' if abs(mean_change) > 10 else ('improvement' if abs(mean_change) > 5 else '')
                    std_class = 'improvement' if std_change < -5 else ('warning' if std_change > 5 else '')
                    
                    html_report += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{row['Original_Mean']:.4f}</td>
                        <td>{row['Cleaned_Mean']:.4f}</td>
                        <td class="{mean_class}">{mean_change:.2f}%</td>
                        <td>{row['Original_Std']:.4f}</td>
                        <td>{row['Cleaned_Std']:.4f}</td>
                        <td class="{std_class}">{std_change:.2f}%</td>
                    </tr>
                    """
                
                html_report += """
                </table>
                """
                
                # Add interpretation of changes
                html_report += """
                <h5>Interpretation of Changes</h5>
                <ul>
                """
                
                for col, row in summary_comparison.iterrows():
                    mean_change = row['Mean_Change_%']
                    std_change = row['Std_Change_%']
                    
                    if abs(mean_change) > 5 or abs(std_change) > 5:
                        interpretation = []
                        
                        if abs(mean_change) > 10:
                            direction = "increased" if mean_change > 0 else "decreased"
                            interpretation.append(f"<strong>significant {direction} mean</strong> ({mean_change:.2f}%)")
                        elif abs(mean_change) > 5:
                            direction = "increased" if mean_change > 0 else "decreased"
                            interpretation.append(f"{direction} mean ({mean_change:.2f}%)")
                        
                        if std_change < -10:
                            interpretation.append(f"<strong>greatly reduced variability</strong> ({std_change:.2f}%)")
                        elif std_change < -5:
                            interpretation.append(f"reduced variability ({std_change:.2f}%)")
                        elif std_change > 10:
                            interpretation.append(f"<strong>increased variability</strong> ({std_change:.2f}%)")
                        elif std_change > 5:
                            interpretation.append(f"slightly increased variability ({std_change:.2f}%)")
                        
                        if interpretation:
                            html_report += f"""
                            <li><strong>{col}</strong>: {', '.join(interpretation)}</li>
                            """
                
                html_report += """
                </ul>
                """
            except Exception as e:
                html_report += f"""
                <p>Error loading summary statistics comparison: {str(e)}</p>
                """
        
        html_report += """
            </div>
        """
        
        html_report += """
            </div>
        """
    
        # Update the onload script to specifically target the first tab by ID
    if tab_ids:
        first_tab_id = tab_ids[0]
        html_report += f"""
        <script>
            // Open the first tab by default when the page loads
            document.addEventListener('DOMContentLoaded', function() {{
                // Get the first tab button and simulate a click
                var firstTab = document.querySelector('button.tablinks[onclick*="{first_tab_id}"]');
                if (firstTab) {{
                    firstTab.click();
                }}
            }});
        </script>
        """
    
    # Add recommendations section
    html_report += """
        <div class="section">
            <h2>Recommendations</h2>
            <div class="subsection">
    """
    
    # Generate recommendations based on results
    methods_used = results.get('methods', [])
    
    # Recommendations for high outlier counts
    high_outlier_columns = []
    
    # Check Z-score outliers
    if 'zscore_results' in results and 'column_outlier_counts' in results['zscore_results']:
        for col, count in results['zscore_results']['column_outlier_counts'].items():
            if count / df.shape[0] * 100 > 5:
                high_outlier_columns.append((col, 'Z-score'))
    
    # Check IQR outliers
    if 'iqr_results' in results and 'outliers_by_column' in results['iqr_results']:
        for col, info in results['iqr_results']['outliers_by_column'].items():
            if info['outlier_percentage'] > 5:
                # Only add if not already added
                if not any(c[0] == col for c in high_outlier_columns):
                    high_outlier_columns.append((col, 'IQR'))
    
    if high_outlier_columns:
        html_report += """
                <h4>High Outlier Counts</h4>
                <p>The following columns have high outlier percentages and may need special attention:</p>
                <ul>
        """
        
        for col, method in high_outlier_columns:
            html_report += f"""
                    <li><strong>{col}</strong> (detected by {method} method)</li>
            """
        
        html_report += """
                </ul>
                <p>Recommendations:</p>
                <ul>
                    <li>Consider transforming these variables (e.g., log transformation) if they have skewed distributions</li>
                    <li>Examine if these outliers represent valid but rare events or potential data errors</li>
                    <li>For machine learning applications, consider robust models or outlier removal techniques</li>
                </ul>
        """
    
    # Recommendations for consensus outliers
    if 'combined_results' in results and 'agreement_counts' in results['combined_results']:
        agreement_counts = results['combined_results']['agreement_counts']
        max_agreement = max(agreement_counts.keys()) if agreement_counts else 0
        
        if max_agreement == len(methods_used) and len(methods_used) > 1:
            # Outliers detected by all methods
            consensus_count = len(agreement_counts.get(max_agreement, []))
            
            html_report += f"""
                <h4>Strong Consensus Outliers</h4>
                <p>{consensus_count} outliers were identified by all {len(methods_used)} methods.</p>
                <p>Recommendations:</p>
                <ul>
                    <li>These are highly likely to be true outliers and should be carefully examined</li>
                    <li>Consider flagging or removing these points for modeling purposes</li>
                    <li>Investigate the root cause of these outliers in your data generation process</li>
                </ul>
            """
    
    # Add recommendations based on cleaned data analysis
    if 'cleaned_data_results' in results and 'removal_info' in results['cleaned_data_results']:
        removal_info = results['cleaned_data_results']['removal_info']
        
        if removal_info.get('outliers_removed', 0) > 0:
            html_report += """
                <h4>Cleaned Data Recommendations</h4>
                <p>Based on the analysis of data with outliers removed:</p>
                <ul>
            """
            
            # Check if there are significant changes in statistics
            significant_changes = False
            if os.path.exists(os.path.join(output_dir, 'cleaned_data/before_after_comparison/summary_statistics_comparison.csv')):
                try:
                    summary_comparison = pd.read_csv(os.path.join(output_dir, 'cleaned_data/before_after_comparison/summary_statistics_comparison.csv'), index_col=0)
                    
                    # Check for significant changes in mean or std
                    for _, row in summary_comparison.iterrows():
                        if abs(row['Mean_Change_%']) > 10 or abs(row['Std_Change_%']) > 15:
                            significant_changes = True
                            break
                except:
                    pass
            
            if significant_changes:
                html_report += """
                    <li>The outlier removal significantly changed statistical properties of your data, suggesting these outliers had a substantial impact</li>
                    <li>Consider using the cleaned dataset for modeling, as it likely provides a more representative view of your typical data</li>
                    <li>If these outliers represent important edge cases, consider creating separate models for normal vs. outlier data</li>
                """
            else:
                html_report += """
                    <li>The outlier removal did not dramatically change statistical properties, suggesting these outliers may not have had a major impact</li>
                    <li>Consider keeping the full dataset if the outliers represent valid, if rare, cases in your domain</li>
                """
            
            # Add recommendations about remaining outliers
            if 'zscore_results' in results['cleaned_data_results'] and results['cleaned_data_results']['zscore_results'].get('outlier_count', 0) > 0:
                html_report += """
                    <li>There are still outliers in the cleaned dataset. Consider:
                        <ul>
                            <li>Using a more stringent outlier removal approach if appropriate</li>
                            <li>Applying robust statistical methods that are less sensitive to remaining outliers</li>
                        </ul>
                    </li>
                """
            
            html_report += """
                </ul>
            """
    
    # General recommendations
    html_report += """
                <h4>General Recommendations</h4>
                <ul>
                    <li>Use domain knowledge to determine if identified outliers are errors or valid extreme values</li>
                    <li>For machine learning, consider:
                        <ul>
                            <li>Using robust algorithms less sensitive to outliers (e.g., Random Forest, Gradient Boosting)</li>
                            <li>Applying transformations to reduce the impact of outliers</li>
                            <li>Creating a binary flag indicating outlier status as an additional feature</li>
                        </ul>
                    </li>
                    <li>For time-series data, check if outliers occur at specific time points that might indicate special events</li>
                    <li>For multivariate outliers (PCA-based), examine the combinations of variables that contribute to their unusual nature</li>
                </ul>
    """
    
    html_report += """
            </div>
        </div>
    """
    
    # Close HTML document
    html_report += """
    </body>
    </html>
    """
    
    # Write HTML to file
    html_path = os.path.join(output_dir, 'outlier_detection_report.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    return html_path




