from dcat_ap_hub import Dataset
from pathlib import Path

def create_parser_file(dataset_name, data_dir):
    output_dir = Path(data_dir) / "_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def download_dataset(url: str):
    #TODO: Wie sieht es aus mit Metadaten: https://ki-daten.hlrs.de/hub/repo/datasets/10-5281-zenodo-13808085.jsonld

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    ds = Dataset.from_url(url)
    dataset_name = ds.title  
    name = dataset_name.replace(" ", "_").replace("/", "-")
    data_dir = PROJECT_ROOT / "data" / name
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        files = ds.download(data_dir=str(data_dir))
    create_parser_file(dataset_name, data_dir)
    return dataset_name, data_dir
    
#TODO:noch implementieren
def upload_to_piveau():
    pass