from __future__ import annotations

import argparse

from backend.agent_core import build_agent, load_data_if_available


def main() -> None:
    parser = argparse.ArgumentParser(description="Shark attack QA agent (Strands).")
    parser.add_argument("--data", help="Path to the Excel/CSV file.")
    parser.add_argument("--s3", help="S3 URI to the Excel/CSV file, e.g. s3://bucket/key")
    parser.add_argument("--question", help="Single question to ask.")
    args = parser.parse_args()

    agent = build_agent()

    data_path = args.s3 or args.data
    if data_path:
        agent.tool.load_shark_attacks(path=data_path)
    else:
        load_data_if_available(agent)

    if args.question:
        response = agent(args.question)
        print(response)
        return

    print("Hi ! I'm your Shark Attack Analyst. Type a question or 'exit' to quit.")
    while True:
        user_input = input("> ").strip()
        if not user_input or user_input.lower() in {"exit", "quit"}:
            break
        response = agent(user_input)
        print(response)


if __name__ == "__main__":
    main()
