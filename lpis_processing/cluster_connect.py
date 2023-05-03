import coiled
import dask.distributed

import lpis_processing.cluster_config_coiled as cluster_config_coiled
from .utils import SingleDispatcher

cluster_dispatcher = SingleDispatcher()


def connect_client(cluster):
    if cluster:
        client = dask.distributed.Client(cluster)
        print(client.dashboard_link)
        client.restart()
        return client
    return None


@cluster_dispatcher.register('coiled')
def connect_coiled_cluster():
    coiled.create_software_environment(
        name=cluster_config_coiled.software_environment_name,
        container=cluster_config_coiled.docker_image_uri,
    )
    if cluster_config_coiled.exists:
        cluster = coiled.Cluster(name=cluster_config_coiled.name)
    else:
        cluster = coiled.Cluster(name=cluster_config_coiled.name, software=cluster_config_coiled.cluster_software,
                                 n_workers=cluster_config_coiled.min_workers,
                                 worker_cpu=cluster_config_coiled.worker_cpu,
                                 worker_memory=cluster_config_coiled.worker_memory)
        cluster.adapt(minimum=cluster_config_coiled.min_workers, maximum=cluster_config_coiled.max_workers)

    return connect_client(cluster)


if __name__ == '__main__':
    cluster_dispatcher('coiled')
