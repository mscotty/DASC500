import streamlit as st
import pandas as pd
import numpy as np
import io
import gc

def create_transformed_dataset(df, numeric_columns):
    """Generate and download transformed dataset"""
    st.subheader("Download Transformed Data")
    
    if not numeric_columns:
        st.warning("No numeric columns available for transformation.")
        return
        
    cols_to_transform = st.multiselect(
        "Select columns to transform:",
        numeric_columns
    )
    
    if not cols_to_transform:
        st.info("Please select at least one column to transform.")
        return
        
    transform_options = {
        "None": lambda x: x,
        "Log": lambda x: np.log(x - x.min() + 1 if x.min() <= 0 else x),
        "Square Root": lambda x: np.sqrt(x - x.min() + 0.01 if x.min() < 0 else x),
        "Square": lambda x: x ** 2,
        "Cube": lambda x: x ** 3,
        "Z-Score": lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean(),
        "Min-Max Scaling": lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    }
    
    transformed_df = df.copy()
    transformation_log = []
    
    for col in cols_to_transform:
        transform_type = st.selectbox(
            f"Transformation for {col}:",
            list(transform_options.keys()),
            key=f"transform_{col}"
        )
        
        if transform_type != "None":
            try:
                # Check for missing values and inform user
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    st.warning(f"Column {col} has {missing_count:,} missing values that will remain missing after transformation.")
                
                # Apply transformation
                new_col_name = f"{col}_{transform_type}"
                transformed_df[new_col_name] = transform_options[transform_type](df[col])
                
                # Log the transformation
                transformation_log.append(f"- Added column '{new_col_name}' using {transform_type} transformation")
                
                # Check for issues in transformed data
                if transformed_df[new_col_name].isnull().sum() > missing_count:
                    st.warning(f"Transformation created additional missing values in {new_col_name}. Check for invalid inputs like negative values for log transformation.")
                
                if np.isinf(transformed_df[new_col_name]).any():
                    st.warning(f"Transformation created infinite values in {new_col_name}. These will be replaced with NaN.")
                    transformed_df[new_col_name] = transformed_df[new_col_name].replace([np.inf, -np.inf], np.nan)
                
            except Exception as e:
                st.error(f"Error transforming {col}: {str(e)}")
    
    if transformation_log:
        st.write("### Transformation Summary")
        for log in transformation_log:
            st.write(log)
        
        # Show preview of transformed data
        st.write("### Preview of Transformed Data")
        st.dataframe(transformed_df.head())
        
        # Convert to CSV
        try:
            csv = transformed_df.to_csv(index=False)
            
            st.download_button(
                label="Download Transformed Data as CSV",
                data=csv,
                file_name="transformed_data.csv",
                mime="text/csv"
            )
            
            # Offer additional formats
            if st.checkbox("Export to Excel instead?"):
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    transformed_df.to_excel(writer, sheet_name='Transformed_Data', index=False)
                
                buffer.seek(0)
                st.download_button(
                    label="Download Transformed Data as Excel",
                    data=buffer,
                    file_name="transformed_data.xlsx",
                    mime="application/vnd.ms-excel"
                )
            
            # Option to save only selected columns
            if st.checkbox("Download only selected columns?"):
                cols_to_save = st.multiselect(
                    "Select columns to include in download:",
                    transformed_df.columns.tolist(),
                    default=transformed_df.columns.tolist()
                )
                
                if cols_to_save:
                    filtered_df = transformed_df[cols_to_save]
                    csv_filtered = filtered_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download Selected Columns as CSV",
                        data=csv_filtered,
                        file_name="transformed_data_selected_columns.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Please select at least one column to download.")
                
        except Exception as e:
            st.error(f"Error creating export file: {str(e)}")
            
        # Clean up to free memory
        del transformed_df
        gc.collect()
    else:
        st.info("No transformations were applied. Select columns and transformations to continue.")
