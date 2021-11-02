"""This example showcases the hello world example of Hera using kubernetes user token"""
from typing import Optional
import errno
import os

from hera.v1.input import InputFrom
from hera.v1.resources import Resources
from hera.v1.task import Task
from hera.v1.workflow import Workflow
from hera.v1.volume import Volume
from hera.v1.workflow_service import WorkflowService

from kubernetes import client, config
import base64


def get_sa_token(
    service_account: str, namespace: str = "default", config_file: Optional[str] = None
):
    """Get ServiceAccount token using kubernetes config.

     Parameters
    ----------
    service_account: str
        The service account to authenticate from.
    namespace: str = 'default'
        The K8S namespace the workflow service submits workflows to. This defaults to the `default` namespace.
    config_file: str
        The path to k8s configuration file.
    
     Raises
    ------
    FileNotFoundError
        When the config_file can not be found.
    """
    if config_file is not None and not os.path.isfile(config_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), config_file)

    config.load_kube_config(config_file=config_file)
    v1 = client.CoreV1Api()
    secret_name = (
        v1.read_namespaced_service_account(service_account, namespace).secrets[0].name
    )
    sec = v1.read_namespaced_secret(secret_name, namespace).data
    return base64.b64decode(sec["token"]).decode()


def preprocess_data(base_path: str):
    import os
    import sys

    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from pathlib import Path

    preprocess_path = Path(base_path) / 'preprocess'
    preprocess_path.mkdir(parents=True, exist_ok=True)
    (Path(base_path) / 'models').mkdir(parents=True, exist_ok=True)

    X, y = datasets.load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    x_train_path = str(preprocess_path / 'x_train.npy')
    x_test_path = str(preprocess_path / 'x_test.npy')
    y_train_path = str(preprocess_path / 'y_train.npy')
    y_test_path = str(preprocess_path / 'y_test.npy')
    model_path = str(Path(base_path) / 'models' / 'sgd_regressor.pkl')

    np.save(x_train_path, X_train)
    np.save(x_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)

    sys.stdout.flush()
    json.dump([{
        'pipeline_config': {
            'base_path': base_path,
            'x_train': x_train_path,
            'x_test': x_test_path,
            'y_train': y_train_path,
            'y_test': y_test_path,
            'model_path': model_path
        }
    }], sys.stdout)


def train_model(pipeline_config):
    import os
    import sys

    import joblib
    import numpy as np
    from sklearn.linear_model import SGDRegressor
    from pathlib import Path

    _sys_stdout = sys.stdout
    sys.stdout=open("/tmp/logs.txt","w")

    base_path = pipeline_config.get('base_path')
    sgd_regressor_path = pipeline_config.get('model_path')

    x_train = pipeline_config.get('x_train')
    y_train = pipeline_config.get('y_train')

    x_train_data = np.load(x_train)
    y_train_data = np.load(y_train)

    model = SGDRegressor(verbose=1)
    model.fit(x_train_data, y_train_data)
    
    joblib.dump(model, sgd_regressor_path)

    sys.stdout.close()
    sys.stdout = _sys_stdout
    sys.stdout.flush()
    json.dump({'pipeline_config': pipeline_config}, sys.stdout)
    with open("/tmp/logs.txt","r") as fd:
        sys.stdout.write(f"\n{'-' * 50}\n{fd.read()}")



def check_model_state(pipeline_config):
    from pathlib import Path
    import sys
    import json

    model_path = Path(pipeline_config.get('model_path'))
      
    result = 'done' if model_path.is_file() else 'unknow'
    sys.stdout.flush()
    json.dump(result, sys.stdout)


def test_model(pipeline_config):
    import os
    import sys
    
    import joblib
    import numpy as np
    from sklearn.metrics import mean_squared_error
    from pathlib import Path

    base_path = pipeline_config.get('base_path')
    summary_path = Path(base_path) / 'summary'
    summary_path.mkdir(parents=True, exist_ok=True)

    x_test = pipeline_config.get('x_test')
    y_test = pipeline_config.get('y_test')
    model_path = pipeline_config.get('model_path')

    x_test_data = np.load(x_test)
    y_test_data = np.load(y_test)

    model = joblib.load(model_path)
    y_pred = model.predict(x_test_data)

    err = mean_squared_error(y_test_data, y_pred)
    
    summary_path_result = str(summary_path / 'output.txt')
    with open(summary_path_result, 'a') as f:
        f.write(str(err))
    
    pipeline_config.update({'summary_path_result': summary_path_result})

    sys.stdout.flush()
    json.dump([{'pipeline_config': pipeline_config}], sys.stdout)


resource_path = '/pipeline'
resource_volume = Resources(volume=Volume(size='2Gi', mount_path=resource_path, storage_class_name='local-path'))


task_preprocess = Task(
    'preprocess',
    preprocess_data,
    [{'base_path': resource_path}],
    image='rodrigobaron/example:latest',
    resources=resource_volume
)

task_model_state = Task(
    'state',
    check_model_state,
    input_from=InputFrom(name='preprocess', parameters=['pipeline_config']),
    image='rodrigobaron/example:latest',
    resources=resource_volume
)

task_train_model = Task(
    'train_model',
    train_model,
    input_from=InputFrom(name='preprocess', parameters=['pipeline_config']),
    image='rodrigobaron/example:latest',
    resources=resource_volume
)

task_test_model = Task(
    'test_model',
    test_model,
    input_from=InputFrom(name='preprocess', parameters=['pipeline_config']),
    image='rodrigobaron/example:latest',
    resources=resource_volume
)


task_preprocess.next(task_model_state)
task_model_state.next_when('{{item.outputs.result}} == ["unknow"]', task_train_model)
# task_model_state.next_when('{{item.outputs.result}} == ["done"]', task_test_model)

task_model_state.next(task_test_model)
task_train_model.next(task_test_model)


namespace = "argo"
token = get_sa_token("argo-server", namespace=namespace)

ws = WorkflowService("argo.k8s.rodrigobaron.com", token, namespace=namespace)
w = Workflow("ml-pipeline", ws)
w.add_tasks(task_preprocess, task_model_state, task_train_model, task_test_model)
ws.submit(w.workflow, namespace=namespace)
