from argparse import ArgumentParser
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chatbot.graph.main_graph import ChatbotGraph

if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("--savefile_name", "-S", required=False, type=str, default="graph.png")
    args = parser.parse_args()

    graph = ChatbotGraph(
        api_key="",
        chat_model="",
        temperature=0.0,
        streaming=True,
        no_retries=0
    )
    app = graph.compile_graph()

    app.get_graph().draw_mermaid_png(output_file_path=args.savefile_name)

