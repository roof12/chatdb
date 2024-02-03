import json


class Conversations:
    def __init__(self, file_path):
        with open(file_path, "r") as file:
            self.conversations = json.load(file)

    def convert(self):
        markdowns = []
        metadatas = []
        ids = []

        for _, conversation in enumerate(self.conversations, start=1):
            markdown_output = ""
            messages = self.get_conversation_messages(conversation)
            markdown_output += f"# {conversation['title']}\n\n"
            for message in messages:
                markdown_output += f"**{message['author']}**: {message['text']}\n\n"
            markdown_output += "\n"

            markdowns.append(markdown_output)
            metadatas.append(
                {
                    "source": "chatgpt",
                    "title": conversation["title"],
                }
            )
            ids.append(conversation["conversation_id"])

        return markdowns, metadatas, ids

    def get_conversation_messages(self, conversation):
        messages = []
        current_node = conversation["current_node"]
        while current_node is not None:
            node = conversation["mapping"][current_node]
            if (
                node.get("message")
                and node["message"].get("content")
                and node["message"]["content"].get("content_type") == "text"
                and len(node["message"]["content"].get("parts", [])) > 0
                and len(node["message"]["content"]["parts"][0]) > 0
                and (
                    node["message"]["author"]["role"] != "system"
                    or node["message"]["metadata"].get("is_user_system_message", False)
                )
            ):
                author = node["message"]["author"]["role"]
                if author == "assistant":
                    author = "ChatGPT"
                elif author == "system" and node["message"]["metadata"].get(
                    "is_user_system_message", False
                ):
                    author = "Custom user info"
                messages.append(
                    {"author": author, "text": node["message"]["content"]["parts"][0]}
                )
            current_node = node.get("parent")
        return messages[::-1]
