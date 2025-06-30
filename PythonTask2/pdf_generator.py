import pandas as pd
import os
from fpdf import FPDF

# --- Configuration ---
DATA_FILE = "sales_data.csv"
REPORT_FILE = "sales_report.pdf"


# --- Function to analyze data ---
def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic analysis: total sales per product."""
    print("Analyzing data...")
    product_sales = df.groupby('Product')['Sales'].sum().reset_index()
    product_sales.rename(columns={'Sales': 'Total Sales'}, inplace=True)
    print("Analysis complete.")
    return product_sales

# --- Function to generate PDF report ---
def generate_pdf_report(data_df: pd.DataFrame, analysis_df: pd.DataFrame, output_filename: str):
    """Generates a PDF report with raw data and analysis."""
    pdf = FPDF()
    pdf.add_page()

    # Set font for title
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 10, "Sales Report", ln=True, align='C')
    pdf.ln(10) # Line break

    # Section: Raw Data
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "1. Raw Data Overview", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 10)
    # Add table headers
    col_widths = [40, 40, 30, 30] # Adjust column widths as needed
    for col in data_df.columns:
        pdf.cell(col_widths[data_df.columns.get_loc(col)], 10, col, border=1, align='C')
    pdf.ln()

    # Add table rows
    for index, row in data_df.iterrows(): # This will return each row in df
        for i, col in enumerate(data_df.columns): # This will return column name and column index
            pdf.cell(col_widths[i], 10, str(row[col]), border=1, align='C')
        pdf.ln()
    pdf.ln(10)

    # Section: Analysis Results
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "2. Total Sales by Product", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 10)
    # Add analysis table headers
    col_widths_analysis = [60, 60]
    for col in analysis_df.columns:
        pdf.cell(col_widths_analysis[analysis_df.columns.get_loc(col)], 10, col, border=1, align='C')
    pdf.ln()

    # Add analysis table rows
    for index, row in analysis_df.iterrows():
        for i, col in enumerate(analysis_df.columns):
            pdf.cell(col_widths_analysis[i], 10, str(row[col]), border=1, align='C')
        pdf.ln()
    pdf.ln(10)

    pdf.output(output_filename)
    print(f"PDF report generated: '{output_filename}'")

# --- Main script execution ---
if __name__ == "__main__":
    # Read data
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Successfully read data from '{DATA_FILE}'.")
    except FileNotFoundError:
        print(f"Error: Data file '{DATA_FILE}' not found. Please ensure it exists.")
        exit()
    except Exception as e:
        print(f"Error reading data file: {e}")
        exit()

    # Analyze data
    analysis_results = analyze_data(df.copy()) # Use a copy to avoid modifying original df

    # Generate report
    generate_pdf_report(df, analysis_results, REPORT_FILE)

    print("\nProcess complete. Check the generated PDF report.")