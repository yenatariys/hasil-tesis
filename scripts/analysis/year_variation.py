
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load raw datasets
app_df = pd.read_csv('data/processed/lex_labeled_review_app.csv')
play_df = pd.read_csv('data/processed/lex_labeled_review_play.csv')

# Parse date columns to datetime objects
app_df['date_parsed'] = pd.to_datetime(app_df['date'], errors='coerce')
play_df['date_parsed'] = pd.to_datetime(play_df['at'], errors='coerce')

# Extract year from parsed dates
app_df['year'] = app_df['date_parsed'].dt.year
play_df['year'] = play_df['date_parsed'].dt.year

# Prepare output directory
output_dir = 'outputs/results/year_variation/'
os.makedirs(output_dir, exist_ok=True)

# App Store yearly distribution
app_year_counts = app_df['year'].value_counts().sort_index()
app_output_path = os.path.join(output_dir, 'app_store_yearly_distribution.txt')
with open(app_output_path, 'w', encoding='utf-8') as f:
	f.write("Distribusi Tahunan App Store:\n")
	f.write(app_year_counts.to_string())
	f.write("\n")

# Play Store yearly distribution
play_year_counts = play_df['year'].value_counts().sort_index()
play_output_path = os.path.join(output_dir, 'play_store_yearly_distribution.txt')
with open(play_output_path, 'w', encoding='utf-8') as f:
	f.write("Distribusi Tahunan Play Store:\n")
	f.write(play_year_counts.to_string())
	f.write("\n")

# Also print to console for immediate feedback
print("Distribusi Tahunan App Store:")
print(app_year_counts)
print("\nDistribusi Tahunan Play Store:")
print(play_year_counts)

# --- Visualization ---
def save_bar_and_pie_chart(counts, platform_name, output_dir):
	# Bar chart
	plt.figure(figsize=(7, 4))
	ax = counts.plot(kind='bar', color='#4a90e2')
	plt.title(f'{platform_name} Ulasan per Tahun')
	plt.xlabel('Tahun')
	plt.ylabel('Jumlah Ulasan')
	# Add count labels inside each bar (rounded, easy to read)
	for i, v in enumerate(counts.values):
		ax.text(i, v * 0.5, str(int(round(v))), ha='center', va='center', fontsize=11, color='white', fontweight='bold')
	plt.tight_layout()
	bar_path = os.path.join(output_dir, f'{platform_name.lower().replace(" ", "_")}_yearly_bar.png')
	plt.savefig(bar_path)
	plt.close()

	# Pie chart
	plt.figure(figsize=(5, 5))
	def autopct_format(pct):
		return '{:.0f}%'.format(pct) if pct > 0 else ''
	counts.plot(kind='pie', autopct=autopct_format, startangle=90, colors=plt.cm.Paired.colors)
	plt.title(f'{platform_name} Distribusi Ulasan per Tahun')
	plt.ylabel('')
	plt.tight_layout()
	pie_path = os.path.join(output_dir, f'{platform_name.lower().replace(" ", "_")}_yearly_pie.png')
	plt.savefig(pie_path)
	plt.close()

# Save charts for both platforms
save_bar_and_pie_chart(app_year_counts, 'App Store', output_dir)
save_bar_and_pie_chart(play_year_counts, 'Play Store', output_dir)