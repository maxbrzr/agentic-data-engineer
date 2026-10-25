from dcat_ap_hub import Dataset

url = "/Users/maxburzer/projects/agentic-data-engineer/output/machine-failure/runs/20261025T113208120864Z__kit__kit.glm-5.3-flash__84ada12e"

ds = Dataset.from_directory(url)
files = ds.download(data_dir="./data")

print(list(files))

print(files["test.csv"].data)

files_processed = ds.process(processed_dir="./processed")
