# The code for generating a stacked bar chart to show capacitive contribution classsification

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load Excel file (Replace with the actual file path and Sheet name)
file_path = 'Class data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Group by Year and Class and count the frequency of occurrence
grouped_data = df.groupby(['Year', 'Class']).size().unstack(fill_value=0)

# Create a horizontal stacked bar chart
grouped_data.plot(kind='barh', stacked=True, figsize=(10, 6), color=['green', '#9ACD32', '#FFA500', (239/255, 102/255, 113/255)])

# Add titles and labels
plt.xticks([])
plt.yticks(fontsize=24)
plt.xlabel('Capacitive contribution classification',fontsize=24)
plt.ylabel('Year',fontsize=24)

plt.legend(fontsize=24)

# Show the figure
plt.tight_layout()
plt.show()
