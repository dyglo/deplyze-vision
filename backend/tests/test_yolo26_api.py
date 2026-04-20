"""Backend API tests for YOLO26 CV Platform"""
import pytest
import requests
import os

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')

class TestHealth:
    """Health check tests"""
    def test_root_health(self):
        r = requests.get(f"{BASE_URL}/api/")
        assert r.status_code == 200
        data = r.json()
        assert "YOLO26" in data.get("message", "")

class TestModelConfigs:
    """Model config tests"""
    def test_list_model_configs_returns_5(self):
        r = requests.get(f"{BASE_URL}/api/model-configs")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5
        assert {m["family"] for m in data} == {"YOLO26"}

    def test_model_configs_have_required_fields(self):
        r = requests.get(f"{BASE_URL}/api/model-configs")
        assert r.status_code == 200
        for model in r.json():
            assert "id" in model
            assert "name" in model
            assert "task" in model
            assert "is_builtin" in model
            assert model["url"].startswith("/models/yolo26")
            assert model["url"].endswith("/model.json")

    def test_filter_by_task(self):
        r = requests.get(f"{BASE_URL}/api/model-configs?task=detect")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        for m in data:
            assert m["task"] == "detect"

class TestProjects:
    """Projects CRUD tests"""
    created_id = None

    def test_create_project(self):
        r = requests.post(f"{BASE_URL}/api/projects", json={
            "name": "TEST_Project_Alpha",
            "description": "Test project",
            "task_types": ["detect", "pose"]
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "TEST_Project_Alpha"
        assert "id" in data
        TestProjects.created_id = data["id"]

    def test_list_projects(self):
        r = requests.get(f"{BASE_URL}/api/projects")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        ids = [p["id"] for p in data]
        assert TestProjects.created_id in ids

    def test_get_project(self):
        r = requests.get(f"{BASE_URL}/api/projects/{TestProjects.created_id}")
        assert r.status_code == 200
        assert r.json()["name"] == "TEST_Project_Alpha"

    def test_delete_project(self):
        r = requests.delete(f"{BASE_URL}/api/projects/{TestProjects.created_id}")
        assert r.status_code == 200
        # Verify deleted
        r2 = requests.get(f"{BASE_URL}/api/projects/{TestProjects.created_id}")
        assert r2.status_code == 404

class TestRuns:
    """Runs CRUD tests"""
    created_id = None

    def test_create_run(self):
        r = requests.post(f"{BASE_URL}/api/runs", json={
            "task": "detect",
            "model_name": "COCO-SSD Detection",
            "source_type": "image",
            "results_count": 3,
            "detections": [{"label": "person", "confidence": 0.95}]
        })
        assert r.status_code == 200
        data = r.json()
        assert data["task"] == "detect"
        assert "id" in data
        TestRuns.created_id = data["id"]

    def test_list_runs(self):
        r = requests.get(f"{BASE_URL}/api/runs")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_list_runs_filter_task(self):
        r = requests.get(f"{BASE_URL}/api/runs?task=detect")
        assert r.status_code == 200
        for run in r.json():
            assert run["task"] == "detect"

    def test_delete_run(self):
        r = requests.delete(f"{BASE_URL}/api/runs/{TestRuns.created_id}")
        assert r.status_code == 200

class TestDatasets:
    """Dataset CRUD tests"""
    created_id = None

    def test_create_dataset(self):
        r = requests.post(f"{BASE_URL}/api/datasets", json={
            "name": "TEST_Dataset_Cars",
            "description": "Car detection dataset",
            "classes": ["car", "truck", "bus"],
            "image_count": 100
        })
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "TEST_Dataset_Cars"
        assert "car" in data["classes"]
        TestDatasets.created_id = data["id"]

    def test_list_datasets(self):
        r = requests.get(f"{BASE_URL}/api/datasets")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_delete_dataset(self):
        r = requests.delete(f"{BASE_URL}/api/datasets/{TestDatasets.created_id}")
        assert r.status_code == 200

class TestStats:
    """Stats endpoint tests"""
    def test_get_stats(self):
        r = requests.get(f"{BASE_URL}/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "projects" in data
        assert "runs" in data
        assert "datasets" in data
        assert "models" in data
        assert "task_distribution" in data
        assert isinstance(data["projects"], int)
