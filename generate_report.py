from fpdf import FPDF
import csv
datafile = 'sales_data.csv'
reportname = 'Sales Report'
outputfile = 'salesreport.pdf'
salesdata = []
try:
    with open(datafile, 'r') as file:
        reader = csv.DictReader(file)
        for entry in reader:
            salesdata.append(entry)
except FileNotFoundError:
    print(f"Error: File '{datafile}' not found.")
    exit()
totalsales = 0
for item in salesdata:
    totalsales += int(item['Quantity']) * float(item['Price'])
report = FPDF()
report.add_page()
report.set_font("Helvetica", 'B', 16)
report.cell(200, 10, text=reportname, new_x="LMARGIN", new_y="NEXT", align='C')
report.ln(10)
report.set_font("Helvetica", 'B', 12)
for column in salesdata[0].keys():
    report.cell(40, 10, text=column.title(), border=1, align='C')
report.ln()
report.set_font("Helvetica", size=10)
for row in salesdata:
    for value in row.values():
        report.cell(40, 10, text=str(value), border=1, align='C')
    report.ln()
if totalsales > 0:
    report.ln(10)
    report.set_font("Helvetica", 'B', 12)
    report.cell(0, 10, text=f"Total Sales: {totalsales:₹.2f}", new_x="LMARGIN", new_y="NEXT", align='R')
report.output(outputfile)
print(f"Report '{outputfile}' generated successfully.")
