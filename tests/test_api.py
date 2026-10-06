from fastapi.testclient import TestClient
from backend.main import app
from pathlib import Path

def test_health():
    response=TestClient(app).get('/health')
    assert response.status_code==200 and response.json()['status']=='ok'

def test_invalid_upload():
    response=TestClient(app).post('/api/analyses',files={'video':('notes.txt',b'not video','text/plain')})
    assert response.status_code==415

def test_unknown_job():
    assert TestClient(app).get('/api/analyses/not-found').status_code==404


def test_pose_upload_generates_real_report():
    sample = Path(__file__).resolve().parents[1] / 'sample/height1_X3D.npy'
    client = TestClient(app)
    with sample.open('rb') as pose:
        response = client.post('/api/poses', files={'pose': ('X3D.npy', pose, 'application/octet-stream')}, data={'fps': '59.94'})
    assert response.status_code == 202
    analysis_id = response.json()['analysis_id']
    assert client.get(f'/api/analyses/{analysis_id}').json()['status'] == 'completed'
    report = client.get(f'/api/analyses/{analysis_id}/results').json()
    assert report['cycles']['count'] == 13
    assert len(report['charts']) == 4
    assert client.get(f'/api/analyses/{analysis_id}/charts/knee_flexion.svg').status_code == 200
