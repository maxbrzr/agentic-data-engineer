from pathlib import Path
import sys

from opencode_ai import Opencode
from threading import Thread

#TODO: readme doku
#TODO: Docker ->?
#TODO: !NICHT alles Zeitreihen
# Später: -> #TODO: wie gehen wir mit Images, Tabular, Texten um... 


#TODO: mehr Datensätze Testen
#TODO: mehr Tests. 
#TODO: wenn ich für den gleichen Datensatz nochmal -> alte raus
#TODO: wie siehts mit Downloads aus -> Queue 
#TODO: wie siehts mit Uploads aus 
#TODO: genauer gucken, wie "Reasoning aussehen wird"


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.downloadingUploading import download_dataset, create_parser_file
from engine.opencode_logging import save_session_messages, watch_session_events

BASE_URL_OPENCODE = "http://127.0.0.1:54321"

client = Opencode(base_url=BASE_URL_OPENCODE)
system_prompt = (PROJECT_ROOT / ".opencode" / "agents" / "Agent.md").read_text(
    encoding="utf-8"
)

output_dir = PROJECT_ROOT / "output"
session = client.session.create()

def run_pipeline():
    #hier vielleicht sowas wie "pop-queue"....
    data_dir = download_dataset("https://ki-daten.hlrs.de/de/dataset/10-5281-zenodo-13808085")

    watcher = Thread(
        target=watch_session_events,
        args=(BASE_URL_OPENCODE, session.id, output_dir),
        daemon=True,
    )
    watcher.start()

    response = client.session.chat(
        session.id,
        provider_id="opencode",
        model_id="deepseek-v4-flash-free",
        mode="Agent",
        system=system_prompt,
        timeout= 1200,
        parts=[
            {
                "type": "text",
                "text": (
                    f"Befolge den System- und Agent-Prompt vollständig. Dein aktueller Datensatz liegt in {data_dir}. "
                    f"Schreibe alle erzeugten Dateien in {output_dir}. "
                    f"Führe den Auftrag aus: analysiere {data_dir}, schreibe {output_dir / 'parser.py'} "
                    "und validiere den Parser."
                ),
            }
        ],
    )

    print("session_id:", session.id)
    print("message_id:", response.id)
    print(response)

    watcher.join(timeout=5)

    saved_paths = save_session_messages(client, session.id, output_dir)
    print("opencode run log:", saved_paths["run_log"])
    print("opencode report:", saved_paths["report"])


run_pipeline()
