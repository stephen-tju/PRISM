# python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf

export utilization=0.7
export model_name="meta-llama/Llama-2-7b-hf"

CUDA_VISIBLE_DEVICES=0 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8000 > ../server-8000.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8010 > ../server-8010.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8002 > ../server-8002.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8003 > ../server-8003.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8004 > ../server-8004.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8005 > ../server-8005.log 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8006 > ../server-8006.log 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup vllm serve $model_name --gpu-memory-utilization $utilization --port 8007 > ../server-8007.log 2>&1 &


