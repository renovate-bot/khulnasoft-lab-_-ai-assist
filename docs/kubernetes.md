# Kubernetes cluster

This file provides an overview of the kubernetes cluster we use to deploy Code Suggestions (CS) post 
Milestone ["Code Suggestions Gated MVC"](https://gitlab.com/groups/gitlab-org/modelops/applied-ml/code-suggestions/-/epics/2).

We rely on Google Kubernetes Engine (GKE) with the GPU support enabled to deploy all CS components.
For security reasons, we have set up the `ai-assist` cluster as private, i.e. the nodes have no public IP addresses.

Cluster configuration:
- name: ai-assist
- version: 1.24.8-gke.2000
- zone: us-central1-c
- node pool \#0: node-pool-n2-cpu
  - description: node pool created for apps requiring CPU only
  - type: n2-standard-2
  - autoscaling enabled: 0-5 nodes
- node pool \#1: node-pool-a100-gpu
  - description: node pool created for apps requiring GPU+CPU
  - type: a2-highgpu-1g
  - autoscaling enabled: 0-5 nodes
  - 1 x NVIDIA A100 40GB per node
  - NVIDIA-SMI 470.141.03 
  - Driver Version: 470.141.03
  - CUDA Version: 11.7

Please note that we have two node pools created specifically for different app types. 
The CPU-only pool is for the apps that **do not** require high-performance CPU, e.g., model-gateway and nfs-server.
The GPU pool is for the Triton server and its components that **do** require high-performance CPU and GPU.

At this iteration, we set up 1 GPU per node. By default, we run 1 model replica with the option to increase 
the number of replicas up to 5 (one model per node). By increasing the number of GPUs per node, we can achieve a higher
density of models on the same number of nodes. Triton also has options to get several logical model instances per one physical GPU. 

For model storage, the cluster provides an NFS server with 500GB of available space. The `models-fauxpilot-nfs-pv` 
persistent volume created on top of the NFS has `ReadWriteMany` access mode. With follow-up iterations, we
can revisit the way to store models and update them. The current solution cannot be considered optimal for GA.
The NFS server is not linked to Gitlab CI. We may have difficulty updating model versions at one go.

Relevant links: 
- n2-standard-2 - [N2 Machine series](https://cloud.google.com/compute/docs/general-purpose-machines#n2_machines)
- a2-highgpu-1g - [NVIDIA A100 GPU](https://cloud.google.com/compute/docs/gpus#a100-gpus)
