# Configuraciones de los motores de inferencia (voz05)

Parámetros y comandos exactos usados en el benchmark. Máquina **voz05** (2× RTX PRO 6000
Blackwell, sm_120, CUDA 13.0), todo en **espacio de usuario** bajo `~/llm-bench/`.
Generador en **GPU0**, embeddings (Ollama) en **GPU1**. API OpenAI-compatible en `:8000`,
consumida por la app `:8510` (modo `openai`, `OPENAI_BASE_URL=http://localhost:8000/v1`,
`OPENAI_MODEL=gemma4-awq`).

---

## 1 · vLLM

### Instalación (userspace)
- Entorno: `uv` + venv `~/llm-bench/vllm-env` (Python 3.10).
- `uv pip install vllm --torch-backend=auto` → **vLLM 0.23.0 + torch 2.11.0+cu130** (auto detecta el driver cu130).
- Toolkit CUDA en userspace (conda/miniforge en `~/llm-bench/cuda13`), necesario para los kernels JIT:
  ```
  conda create -p ~/llm-bench/cuda13 -c nvidia cuda-nvcc=13 cuda-cudart-dev=13 cuda-cccl=13
  conda install -p ~/llm-bench/cuda13 -c nvidia libcurand-dev   # lo pide el sampler de FlashInfer
  ```

### Variables de entorno (antes de `vllm serve`)
```bash
export CUDA_HOME=~/llm-bench/cuda13
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH          # cabeceras (curand.h…)
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH                     # -lcudart en compilación
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH               # .so en ejecución
```

### Comando de arranque (26B)
```bash
CUDA_VISIBLE_DEVICES=0 ~/llm-bench/vllm-env/bin/vllm serve \
  ~/llm-bench/models/hf/gemma-4-26B-A4B-it-AWQ-4bit \
  --port 8000 --served-model-name gemma4-awq \
  --gpu-memory-utilization 0.9 \
  --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":0,"video":0}' \
  --max-num-batched-tokens 16384
```
- Para el **12B** (modelo `Unified`, incluye audio): añadir `"audio":0` →
  `--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}'`.

### Significado de los parámetros
| Flag | Valor | Para qué |
|---|---|---|
| `--gpu-memory-utilization` | 0.9 | % de VRAM reservada (modelo + KV-caché). |
| `--max-model-len` | 16384 | Contexto máximo por petición (tokens). |
| `--limit-mm-per-prompt` | image/video/audio = 0 | RAG solo texto → salta el warmup multimodal (~37 s) y libera presupuesto. |
| `--max-num-batched-tokens` | 16384 | Tope de tokens por iteración (prefill troceado); ancho para prompts RAG largos. |
| `--served-model-name` | gemma4-awq | Alias en la API (la app pide ese nombre). |

### Activo por defecto (no hay que ponerlo)
- **PagedAttention** (gestión de KV-caché tipo memoria virtual) — clave del rendimiento.
- **Continuous batching**, **CUDA graphs** + **torch.compile** (requieren el `nvcc` de arriba).
- **Prefix caching** (`enable_prefix_caching=True`) — reutiliza el KV del system-prompt compartido. Desactivable con `--no-enable-prefix-caching` (se usó solo para el A/B).
- Sampler **FlashInfer**, **chunked prefill**, **async scheduling**.

### Modelos
- `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` (compressed-tensors W4A16, kernels Marlin).
- `cyankiwi/gemma-4-12B-it-AWQ-INT4`.

---

## 2 · llama.cpp

### Compilación (userspace, GPU Blackwell sm_120)
- Clonado `ggml-org/llama.cpp` en `~/llm-bench/llama.cpp`. Toolchain: `cmake`+`ninja` (en miniforge), `gcc/g++ 11`, toolkit `~/llm-bench/cuda13` (nvcc + **libcublas-dev** añadido para llama.cpp).
```bash
export CUDA_HOME=~/llm-bench/cuda13
export PATH=~/llm-bench/miniforge3/bin:$CUDA_HOME/bin:$PATH
export CPATH=$CUDA_HOME/targets/x86_64-linux/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc)
```
- Binario: `~/llm-bench/llama.cpp/build/bin/llama-server`.

### Variable de entorno (ejecución)
```bash
export LD_LIBRARY_PATH=~/llm-bench/cuda13/lib:$LD_LIBRARY_PATH   # si no, 'libcudart/libcublas not found'
```

### Comando de arranque (NP = nº de slots; CTX = 16384 × NP)
```bash
CUDA_VISIBLE_DEVICES=0 ~/llm-bench/llama.cpp/build/bin/llama-server \
  --model ~/llm-bench/models/gguf/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --alias gemma4-awq \
  --host 0.0.0.0 --port 8000 \
  -ngl 99 \
  -fa on \
  -cb \
  --parallel <NP> \
  --ctx-size <16384*NP> \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --no-webui
```
(Barrido del benchmark: NP = 1, 4, 8 → `--ctx-size` 16384 / 65536 / 131072.)

### Significado de los parámetros
| Flag | Valor | Para qué |
|---|---|---|
| `-ngl, --gpu-layers` | 99 | Descarga TODAS las capas a la GPU. |
| `-fa, --flash-attn` | on | Flash-attention (como Ollama). |
| `-cb, --cont-batching` | — | Continuous batching (varios slots a la vez). |
| `--parallel` | 1/4/8 | Nº de slots concurrentes (≈ NUM_PARALLEL de Ollama). |
| `--ctx-size` | 16384×NP | Contexto TOTAL, repartido entre slots → 16384 por slot. |
| `--chat-template-kwargs` | `enable_thinking:false` | **Crítico**: gemma-4 es híbrido y por defecto "piensa" (la salida iba a `reasoning_content`, `content` vacío). Esto fuerza respuesta directa, igual que vLLM/Ollama. |
| `--alias` | gemma4-awq | Nombre en la API (la app no cambia). |
| `--no-webui` | — | No servir la UI web. |

### Modelos (GGUF mainline)
- `ggml-org/gemma-4-26B-A4B-it-Q4_K_M.gguf` y `ggml-org/gemma-4-12B-it-Q4_K_M.gguf`.
- ⚠️ **El GGUF de Ollama NO carga** en llama.cpp mainline (`wrong number of tensors; expected 1014, got 658`): Ollama empaqueta los expertos del MoE en un layout fork-específico. Por eso se usan los GGUF oficiales de `ggml-org` (mismo modelo y cuantización Q4_K_M).

---

## 3 · Contexto compartido (para reproducir)

- **Embeddings**: servidor **Ollama** independiente en **GPU1:11435** (`qwen3-embedding:4b` / `:0.6b`), aislado por `CUDA_VISIBLE_DEVICES=1`. La app lo usa vía `OLLAMA_EMBED_HOST=http://localhost:11435`.
- **App** `:8510` (uvicorn, 4 workers) en modo `openai` apuntando al motor de turno (`:8000`). Producción (`:8503`) nunca se tocó.
- **Túneles SSH** desde signal4old: `-L 8000:localhost:8000` (generador) y `-L 11435:localhost:11435` (embeddings); voz05 bloquea inbound salvo SSH.
- **Retriever Opción A** (reutiliza vectores de CrateDB en el rerank) → pipeline *generator-bound*, comparación de motores limpia.
- **Cuantización comparada**: vLLM AWQ-4bit (compressed-tensors) vs llama.cpp/Ollama GGUF Q4_K_M — ambas 4-bit, equiparables (no idénticas; documentado).

> Referencia rápida — comando de Ollama (3er motor) para completar:
> `CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=0.0.0.0:11434 OLLAMA_NUM_PARALLEL=<NP> OLLAMA_CONTEXT_LENGTH=16384 ollama serve` (el `CONTEXT_LENGTH` evita el churn router↔generador).
