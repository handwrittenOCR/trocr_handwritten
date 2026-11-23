import os
from typing import Dict, List

import requests


def _env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"Missing required env var: {name}")
    return val


class LabelStudio:
    """Thin Label Studio REST client."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.token = token
        self.h = self._resolve_auth(token)

    def _patch(self, path: str, payload: Dict) -> None:
        """PATCH ``path``; on 401 (expired short-lived JWT) renew the auth once and retry."""
        r = requests.patch(
            f"{self.url}{path}", headers=self.h, json=payload, timeout=30
        )
        if r.status_code == 401:
            self.h = self._resolve_auth(self.token)
            r = requests.patch(
                f"{self.url}{path}", headers=self.h, json=payload, timeout=30
            )
        r.raise_for_status()

    def _probe(self, headers: Dict[str, str]) -> bool:
        """True if these headers authenticate against the projects endpoint."""
        r = requests.get(
            f"{self.url}/api/projects", headers=headers, params={"page_size": 1}
        )
        return r.status_code != 401

    def _resolve_auth(self, token: str) -> Dict[str, str]:
        """
        Pick the auth scheme that Label Studio accepts.

        Tries, in order: legacy ``Token <t>``, direct ``Bearer <t>``, and finally
        exchanging a personal-access (refresh) token at /api/token/refresh for a
        short-lived JWT used as ``Bearer``.
        """
        for scheme in ("Token", "Bearer"):
            h = {"Authorization": f"{scheme} {token}"}
            if self._probe(h):
                return h
        try:
            r = requests.post(f"{self.url}/api/token/refresh", json={"refresh": token})
            access = r.json().get("access") if r.ok else None
            if access:
                h = {"Authorization": f"Bearer {access}"}
                if self._probe(h):
                    return h
        except Exception:
            pass
        raise SystemExit(
            "Label Studio rejected the token (401). Open Account & Settings and "
            "copy the token (or enable Legacy Tokens in Organization -> API Tokens "
            "Settings), then set LS_TOKEN in .env."
        )

    def list_projects(self) -> List[Dict]:
        """Return all projects."""
        r = requests.get(
            f"{self.url}/api/projects", headers=self.h, params={"page_size": 1000}
        )
        r.raise_for_status()
        return r.json().get("results", [])

    def update_config(self, project_id: int, label_config: str) -> None:
        """Replace a project's labeling config (names must stay compatible)."""
        self._patch(f"/api/projects/{project_id}", {"label_config": label_config})

    def update_task(self, task_id: int, data: Dict) -> None:
        """Replace a task's data; its annotations are kept."""
        self._patch(f"/api/tasks/{task_id}", {"data": data})

    def delete_all_tasks(self, project_id: int) -> None:
        """Delete every task (and its annotations) in a project."""
        r = requests.post(
            f"{self.url}/api/dm/actions",
            headers=self.h,
            params={"id": "delete_tasks", "project": project_id},
            json={"selectedItems": {"all": True, "excluded": []}},
        )
        r.raise_for_status()

    def get_or_create_project(self, title: str, label_config: str) -> int:
        """Return the id of the project named ``title``, creating it if absent."""
        for p in self.list_projects():
            if p["title"] == title:
                return p["id"]
        r = requests.post(
            f"{self.url}/api/projects",
            headers=self.h,
            json={"title": title, "label_config": label_config},
        )
        r.raise_for_status()
        return r.json()["id"]

    def import_tasks(self, project_id: int, items: List[Dict], batch: int = 200) -> int:
        """Import task+prediction items into a project, in batches."""
        n = 0
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            r = requests.post(
                f"{self.url}/api/projects/{project_id}/import",
                headers=self.h,
                json=chunk,
            )
            r.raise_for_status()
            n += len(chunk)
        return n

    def export_tasks(self, project_id: int, all_tasks: bool = False) -> List[Dict]:
        """Export a project's tasks with their annotations as JSON; unannotated ones too if ``all_tasks``."""
        r = requests.get(
            f"{self.url}/api/projects/{project_id}/export",
            headers=self.h,
            params={"exportType": "JSON", "download_all_tasks": str(all_tasks).lower()},
        )
        r.raise_for_status()
        return r.json()
