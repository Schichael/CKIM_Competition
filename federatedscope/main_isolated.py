import os
import sys


sys.path = ['/home/michael/Projects/CKIM_Competition/federatedscope', '/home/michael/Projects/CKIM_Competition', '/home/michael/Projects/CKIM_Competition/federatedscope', '/home/michael/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/223.7571.203/plugins/python/helpers/pycharm_display', '/usr/lib/python39.zip', '/usr/lib/python3.9', '/usr/lib/python3.9/lib-dynload', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/federatedscope-0.1.9-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/protobuf-3.19.4-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/Pympler-1.0.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/tensorboardX-2.5.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/tensorboard-2.10.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/wandb-0.13.4-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/iopath-0.1.10-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/fvcore-0.1.5.post20220512-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/PyYAML-6.0-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/grpcio_tools-1.50.0rc1-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/grpcio-1.50.0rc1-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/pandas-1.5.0-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/scipy-1.7.3-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/scikit_learn-1.0.2-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/numpy-1.22.4-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/Werkzeug-2.2.2-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/tensorboard_plugin_wit-1.8.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/tensorboard_data_server-0.6.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/Markdown-3.4.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/google_auth-2.12.0-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/google_auth_oauthlib-0.4.6-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/absl_py-1.3.0-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/six-1.16.0-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/shortuuid-1.0.9-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/setproctitle-1.3.2-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/sentry_sdk-1.9.10-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/psutil-5.9.2-py3.9-linux-x86_64.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/promise-2.3-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/pathtools-0.1.2-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/docker_pycreds-0.4.0-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/GitPython-3.1.29-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/click-8.1.3-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/portalocker-2.5.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/tabulate-0.9.0-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/termcolor-2.0.1-py3.9.egg', '/home/michael/Projects/CKIM_other/CIKM22_FL_Competition/venv/lib/python3.9/site-packages/yacs-0.1.8-py3.9.egg', '/home/michael/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/223.7571.203/plugins/python/helpers/pycharm_matplotlib_backend']

print(sys.path)
from federatedscope.core.cmd_args import parse_args
from federatedscope.core.auxiliaries.data_builder import get_data
from federatedscope.core.auxiliaries.utils import setup_seed, update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, get_server_cls
from federatedscope.core.configs.config import global_cfg
from federatedscope.core.fed_runner import FedRunner
from yacs.config import CfgNode

if os.environ.get('https_proxy'):
    del os.environ['https_proxy']
if os.environ.get('http_proxy'):
    del os.environ['http_proxy']

if __name__ == '__main__':
    init_cfg = global_cfg.clone()
    args = parse_args()
    print(args)

    init_cfg.merge_from_file(args.cfg_file)
    client_num = args.client
    lr = args.lr
    init_cfg.data.client = client_num
    init_cfg.train.optimizer.lr = lr

    update_logger(init_cfg)
    setup_seed(init_cfg.seed)



    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object
    data, modified_cfg = get_data(config=init_cfg.clone())
    init_cfg.merge_from_other_cfg(modified_cfg)

    init_cfg.freeze()

    # load clients' cfg file
    client_cfg = CfgNode.load_cfg(open(args.client_cfg_file,
                                       'r')) if args.client_cfg_file else None

    runner = FedRunner(data=data,
                       server_class=get_server_cls(init_cfg),
                       client_class=get_client_cls(init_cfg),
                       config=init_cfg.clone(),
                       client_config=client_cfg)
    _ = runner.run()
