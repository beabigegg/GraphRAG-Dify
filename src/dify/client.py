from __future__ import annotations
from typing import Dict, Any, Optional
import requests


class DifyClient:
    def __init__(self, base_url: str, api_key: str, app_type: str = "workflow", response_mode: str = "blocking", user_id: str = "demo-user"):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.app_type = app_type
        self.response_mode = response_mode
        self.user_id = user_id
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
        })

    # --- File upload (used by Workflow / Chat) ---
    def upload_file(self, filepath: str) -> str:
        url = f"{self.base_url}/v1/files/upload"
        with open(filepath, "rb") as f:
            files = {"file": (filepath, f)}
            data = {"user": self.user_id}
            r = self.session.post(url, files=files, data=data, timeout=120)
        r.raise_for_status()
        return r.json()["id"]

    # --- Workflow trigger ---
    def run_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/workflows/run"
        payload = {
            "inputs": inputs,
            "response_mode": self.response_mode,
            "user": self.user_id,
        }
        r = self.session.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    # --- Chat messages ---
    def chat_messages(self, inputs: Dict[str, Any], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat-messages"
        payload = {
            "inputs": inputs,
            "response_mode": self.response_mode,
            "user": self.user_id,
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        r = self.session.post(url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()

    # --- Unified extraction wrapper ---
    def extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Depending on the Dify app type:
        #   - Workflow: treat {text, images, section_id, rev} as workflow inputs
        #   - Chat: embed the same content into inputs/text or fields
        if self.app_type == "workflow":
            return self.run_workflow(inputs=payload)
        return self.chat_messages(inputs={"text": payload})